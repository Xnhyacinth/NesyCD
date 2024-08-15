import json
import contextlib
import gc
import time
import ray
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaForCausalLM
from transformers import TopPLogitsWarper, LogitsProcessorList
import pandas as pd
from tqdm import tqdm
import argparse
import re
from vllm.distributed.parallel_state import destroy_model_parallel

iterations = [] 
# n_steps = 1024
choices_per_step = 3

def calculate_percentage_differences(scores):
    differences = []
    percentage_pattern = re.compile(r'\(([\d.]+)%\)')
    for score_pair in scores:
        percentages = []
        for score in score_pair:
            # start = score.find('(') + 1
            # end = score.find('%')
            # while True:
            #     if '(' in score[start:end]:
            #         # print(score[start:end])
            #         start = score[start:end].find('(') + start
            #     else:
            #         break
            # percentage = float(score[start:end])
            match = percentage_pattern.search(score)
            if match:
                percentage = float(match.group(1))
                percentages.append(percentage)
        difference = abs(percentages[0] - percentages[1])
        differences.append(difference)
    return differences

def get_retirval(model_path, data, threshold=68, n_steps=16):
    # model_path = 'models/llama2-7b_lora_sft_bbh_std_cot_merge_r32_lr2e-4'
    system="{prefix} {task} Your response should conclude with the format \"Therefore, the answer is\".\n\n"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = 'left'
    # tokenizer.pad_token = tokenizer.unk_token
    # data = data[:32]
    # 加载LLAMA模型
    # device = torch.device('cuda')
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto')
    num, total = 0, []
    with torch.no_grad():
        for d in tqdm(data):
            input_txt = system.format(prefix=d['prefix_prompt'], task=d['task_description'])
            input_ids = tokenizer(input_txt, return_tensors="pt")["input_ids"].to(model.device)
            iterations = []
            # init_len = input_ids.shape[1]
            for _ in range(n_steps): 
                iteration = dict()
                iteration["Input"] = tokenizer.decode(input_ids[0])
                output = model(input_ids=input_ids)
                # Select logits of the first batch and the last token and apply softmax
                next_token_logits = output.logits[0, -1, :]
                next_token_probs = torch.nn.functional.softmax(next_token_logits / 0.6, dim=-1)
                next_tokens = torch.multinomial(next_token_probs, num_samples=1)
                if next_tokens[0] == tokenizer.eos_token_id:  #or len(iterations) == 5
                    break
                iteration['next_tokens'] = (next_tokens, tokenizer.batch_decode(next_tokens))
                iteration['p'] = next_token_probs[next_tokens]
                sorted_ids = torch.argsort(next_token_probs, dim=-1, descending=True)
                for choice_idx in range(choices_per_step):
                    token_id = sorted_ids[choice_idx]
                    token_prob = next_token_probs[token_id].cpu().numpy()
                    token_choice = ( f"{token_id}  {tokenizer.decode(token_id)} ({100 * token_prob:.2f}%)" )
                    iteration[f"Choice {choice_idx+1}"] = token_choice
                iterations.append(iteration)
                # input_ids = torch.cat([input_ids, sorted_ids[None, 0, None]], dim=-1)
                input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1).tile(1, 1)], dim=-1)
        
            scores = [(i['Choice 1'], i['Choice 2']) for i in iterations[:]]
            scores = np.mean(calculate_percentage_differences(scores))
            total.append(scores)
            # print(scores)
            if scores > threshold:
                d['score'] = 'right'
                num += 1
            else:
                d['score'] = 'error'
    print(f'\nAvg scores: {np.mean(total)}         Right Ratio: {num / len(data)}\n')
    del model
    del tokenizer
    destroy_model_parallel()
    with contextlib.suppress(AssertionError):
        torch.distributed.destroy_process_group()
    gc.collect()
    torch.cuda.empty_cache()
    ray.shutdown()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_cached()
    return data