# Inspired by: https://github.com/hendrycks/test/blob/master/evaluate_flan.py
import inspect
import json
import os
import re
from typing import Any, Dict, List, Optional
from ..extras.misc import get_device_count, infer_optim_dtype
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm, trange
from transformers.utils import cached_file
from transformers import GenerationConfig
from ..data import get_template_and_fix_tokenizer
from ..extras.constants import CHOICES, SUBJECTS
from ..hparams import get_eval_args
from ..model import load_model, load_tokenizer, load_config
from .template import get_eval_template
from .eval_csqa import eval_csqa, eval_csqa_chat, support_sets
from ..extras.packages import is_vllm_available
from .inference import generate_completions, get_results, generate, get_results_bbh
import random
from .retrieval import add_knowledge
if is_vllm_available():
    from vllm import LLM

def load_model_vllm(model_args, tokenizer):
    config = load_config(model_args)  # may download model from ms hub
    infer_dtype = infer_optim_dtype(model_dtype=getattr(config, "torch_dtype", None))
    infer_dtype = str(infer_dtype).split(".")[-1]
    engine_args = {
        "model": model_args.model_name_or_path,
        "trust_remote_code": True,
        "dtype": infer_dtype,
        "tensor_parallel_size": get_device_count() or 1,
        "gpu_memory_utilization": model_args.vllm_gpu_util,
        "enforce_eager": model_args.vllm_enforce_eager,
        "max_model_len": 4096 if "70B" in model_args.model_name_or_path else config.max_position_embeddings
    }
    return LLM(**engine_args)

class Evaluator:
    def __init__(self, args: Optional[Dict[str, Any]] = None) -> None:
        self.model_args, self.data_args, self.eval_args, finetuning_args, self.generating_args = get_eval_args(args)
        self.tokenizer = load_tokenizer(self.model_args)["tokenizer"]
        self.tokenizer.padding_side = "right"  # avoid overflow issue in batched inference for llama2
        self.template = get_template_and_fix_tokenizer(self.tokenizer, self.data_args.template)
        if not self.eval_args.vllm:
            self.model = load_model(self.tokenizer, self.model_args, finetuning_args)
        elif not self.eval_args.retrieval:
            self.model = load_model_vllm(self.model_args, self.tokenizer)
        self.eval_template = get_eval_template(self.eval_args.lang)
        choices = CHOICES if "commonsenseqa" not in self.eval_args.task else ["A", "B", "C", "D", "E"]
        self.choice_inputs = [
            self.tokenizer.encode(self.eval_template.prefix + ch, add_special_tokens=False)[-1] for ch in choices
        ]

    @torch.inference_mode()
    def batch_inference(self, batch_input: Dict[str, torch.Tensor]) -> List[str]:
        logits = self.model(**batch_input).logits
        lengths = torch.sum(batch_input["attention_mask"], dim=-1)
        word_probs = torch.stack([logits[i, lengths[i] - 1] for i in range(len(lengths))], dim=0)
        choice_probs = torch.nn.functional.softmax(word_probs[:, self.choice_inputs], dim=-1).detach()
        return [chr(ord("A") + offset.item()) for offset in torch.argmax(choice_probs, dim=-1)]

    def eval(self) -> None:
        mapping = cached_file(
            path_or_repo_id=os.path.join(self.eval_args.task_dir, self.eval_args.task),
            filename="mapping.json",
            cache_dir=self.model_args.cache_dir,
            token=self.model_args.hf_hub_token,
        )

        with open(mapping, "r", encoding="utf-8") as f:
            categorys: Dict[str, Dict[str, str]] = json.load(f)

        category_corrects = {subj: np.array([], dtype="bool") for subj in SUBJECTS}
        pbar = tqdm(categorys.keys(), desc="Processing subjects", position=0)
        results = {}
        for subject in pbar:
            if "trust_remote_code" in inspect.signature(load_dataset).parameters:  # for datasets==2.16.0
                kwargs = {"trust_remote_code": True}
            else:
                kwargs = {}

            dataset = load_dataset(
                path=os.path.join(self.eval_args.task_dir, self.eval_args.task),
                name=subject,
                cache_dir=self.model_args.cache_dir,
                download_mode=self.eval_args.download_mode,
                token=self.model_args.hf_hub_token,
                **kwargs,
            )
            pbar.set_postfix_str(categorys[subject]["name"])
            inputs, outputs, labels = [], [], []
            for i in trange(len(dataset[self.data_args.split]), desc="Formatting batches", position=1, leave=False):
                support_set = (
                    dataset["train"].shuffle().select(range(min(self.eval_args.n_shot, len(dataset["train"]))))
                )
                messages = self.eval_template.format_example(
                    target_data=dataset[self.data_args.split][i],
                    support_set=support_set,
                    subject_name=categorys[subject]["name"],
                )

                input_ids, _ = self.template.encode_oneturn(tokenizer=self.tokenizer, messages=messages)
                inputs.append({"input_ids": input_ids, "attention_mask": [1] * len(input_ids)})
                labels.append(messages[-1]["content"])

            for i in trange(
                0, len(inputs), self.eval_args.batch_size, desc="Predicting batches", position=1, leave=False
            ):
                batch_input = self.tokenizer.pad(
                    inputs[i : i + self.eval_args.batch_size], return_attention_mask=True, return_tensors="pt"
                ).to(self.model.device)
                preds = self.batch_inference(batch_input)
                outputs += preds

            corrects = np.array(outputs) == np.array(labels)
            category_name = categorys[subject]["category"]
            category_corrects[category_name] = np.concatenate([category_corrects[category_name], corrects], axis=0)
            category_corrects["Average"] = np.concatenate([category_corrects["Average"], corrects], axis=0)
            results[subject] = {str(i): outputs[i] for i in range(len(outputs))}

        pbar.close()
        self._save_results(category_corrects, results)
        
    def my_eval(self) -> None:
        results = {}
        if "trust_remote_code" in inspect.signature(load_dataset).parameters:  # for datasets==2.16.0
            kwargs = {"trust_remote_code": True}
        else:
            kwargs = {}

        file = open(f'{self.eval_args.task_dir}/{self.eval_args.task}/{self.data_args.split}.json', 'r', encoding='utf-8')
        dataset = file.read()
        dataset = json.loads(dataset)
        file.close()
        
        inputs, outputs, labels = [], [], []
        for i in trange(len(dataset), desc="Formatting batches", position=1, leave=False):
            messages = self.eval_template.format_example(
                target_data=dataset[i],
                support_set=[],
                subject_name=self.eval_args.task.split('/')[-1],
            )

            input_ids, _ = self.template.encode_oneturn(tokenizer=self.tokenizer, messages=messages)
            inputs.append({"input_ids": input_ids, "attention_mask": [1] * len(input_ids)})
            labels.append(messages[-1]["content"])

        for i in trange(
            0, len(inputs), self.eval_args.batch_size, desc="Predicting batches", position=1, leave=False
        ):
            batch_input = self.tokenizer.pad(
                inputs[i : i + self.eval_args.batch_size], return_attention_mask=True, return_tensors="pt"
            ).to(self.model.device)
            preds = self.batch_inference(batch_input)
            outputs += preds
        corrects = np.array(outputs) == np.array(labels)
        results = {str(i): outputs[i] for i in range(len(outputs))}
        score_info = "{:>15}: {:.2f}".format(self.eval_args.task.split('/')[-1], 100 * np.mean(corrects))

        print(score_info)
        
        if self.eval_args.save_dir is not None:
            os.makedirs(self.eval_args.save_dir, exist_ok=False)
            with open(os.path.join(self.eval_args.save_dir, "results.json"), "w", encoding="utf-8", newline="\n") as f:
                json.dump(results, f, indent=4)

            with open(os.path.join(self.eval_args.save_dir, "results.log"), "w", encoding="utf-8", newline="\n") as f:
                f.write(score_info)
        
    def _save_results(self, category_corrects: Dict[str, np.ndarray], results: Dict[str, Dict[int, str]]) -> None:
        score_info = "\n".join(
            [
                "{:>15}: {:.2f}".format(category_name, 100 * np.mean(category_correct))
                for category_name, category_correct in category_corrects.items()
                if len(category_correct)
            ]
        )
        print(score_info)
        if self.eval_args.save_dir is not None:
            os.makedirs(self.eval_args.save_dir, exist_ok=False)
            with open(os.path.join(self.eval_args.save_dir, "results.json"), "w", encoding="utf-8", newline="\n") as f:
                json.dump(results, f, indent=4)

            with open(os.path.join(self.eval_args.save_dir, "results.log"), "w", encoding="utf-8", newline="\n") as f:
                f.write(score_info)

    def eval_gen(self):
        file = open(f'{self.eval_args.task_dir}/{self.eval_args.task}/{self.data_args.split}.json', 'r', encoding='utf-8')
        dataset = file.read()
        dataset = json.loads(dataset)
        file.close()

        inputs = []
        support_name = self.eval_args.task.split('/')[-1] + '_temp' if 'temp' in self.eval_args.lang else self.eval_args.task.split('/')[-1]
        if 'full' in self.eval_args.lang and 'temp' in self.eval_args.lang:
            support_name += '_full'
        supports_set = support_sets.get(support_name, [])
        
        if self.eval_args.num_knowledge is not None:
            dataset = add_knowledge(self.eval_args, dataset, self.model_args.model_name_or_path)
            if self.eval_args.retrieval:
                self.model = load_model_vllm(self.model_args, self.tokenizer)
        mt = None
        if 'mt' in self.eval_args.lang:
            match = re.findall(r'([^_]+)-([^_]+)', self.eval_args.save_dir)
            for mat in match:
                if mat[0] == 'mt':
                    mt = "{}-{}".format(*mat)
            print(f'\nMethod: {mt}')
        for i in trange(len(dataset), desc="Formatting batches", position=1, leave=False):
            if 'bbh' in self.eval_args.lang:
                supports = supports_set.get(dataset[i]['task_name'], [])
            else:
                supports = supports_set
            random.shuffle(supports)
            k = min(self.eval_args.n_shot, len(supports))
            messages = self.eval_template.format_example(
                target_data=dataset[i],
                support_set=supports[:k],
                subject_name=self.eval_args.task.split('/')[-1] if 'task_name' not in dataset[i].keys() else dataset[i]['task_name'],
                mt=mt,
                num_knowledge=self.eval_args.num_knowledge
            )

            input_ids, _ = self.template.encode_oneturn(tokenizer=self.tokenizer, messages=messages)
            inputs.append({"input_ids": input_ids, "attention_mask": [1] * len(input_ids)})
            dataset[i]['label'] = chr(ord("A") + dataset[i]["choices"].index(dataset[i]['answer'])) if 'output' not in dataset[i].keys() else dataset[i]['output']
            dataset[i]['model_inputs'] = self.tokenizer.decode(input_ids, skip_special_tokens=True)
        
        print('\n')
        print('*' * 200)
        print(self.tokenizer.decode(inputs[-1]['input_ids']))
        print('*' * 200)
        print(messages)
        print(dataset[-1])
        print('*' * 200)
        
        if self.eval_args.gen_chat:
            print('\nEval gen chat... Use vllm...')
            outputs = generate(self.model, self.tokenizer, inputs, self.generating_args, self.eval_args)
            # outputs = eval_csqa_chat(self.model, self.tokenizer, inputs,
            #                         self.generating_args, self.eval_args, self.eval_template)
            if mt == 'mt-cas':
                print('\nmt-cas  stage 2...')
                system_prompt = "{prefix} {task}\n\nQuestion: {question}\nRational: {rational}\nTherefore, the answer is "
                user_prompts = [system_prompt.format(prefix=x['prefix_prompt'], task=x['task_description'], question=x["instruction"], rational=y) for x, y in zip(dataset, outputs)]
                print(f'\n{user_prompts[0]}')
                outputs = generate(self.model, self.tokenizer, user_prompts, self.generating_args, self.eval_args, mt_cas=True)
        else:
            outputs = []
            for i in trange(
                0, len(inputs), self.eval_args.batch_size, desc="Predicting batches", position=1, leave=False
            ):
                batch_input = self.tokenizer.pad(
                    inputs[i : i + self.eval_args.batch_size], return_attention_mask=True, return_tensors="pt"
                ).to(self.model.device)
                outputs += eval_csqa(self.model, self.tokenizer, batch_input, self.generating_args)

        if 'temp' in self.eval_args.lang:
            for sample, output in zip(dataset, outputs):
                sample["summary"] = output
            if self.eval_args.save_dir is not None:
                os.makedirs(self.eval_args.save_dir, exist_ok=False)
                with open(os.path.join(self.eval_args.save_dir, "results.json"), "w", encoding="utf-8", newline="\n") as f:
                    json.dump(dataset, f, indent=4)
        elif 'bbh' in self.eval_args.lang:
            macro_res, results = get_results_bbh(dataset, outputs)
            print("macro_res:", macro_res)
            if self.eval_args.save_dir is not None:
                os.makedirs(self.eval_args.save_dir, exist_ok=False)
                with open(os.path.join(self.eval_args.save_dir, "results.json"), "w", encoding="utf-8", newline="\n") as f:
                    json.dump(results, f, ensure_ascii=False, indent=4)
                with open(os.path.join(self.eval_args.save_dir, "res.json"), "w", encoding="utf-8", newline="\n") as f:
                    json.dump(macro_res, f, ensure_ascii=False, indent=4)
        else:
            results, fail_items, right = get_results(dataset, outputs)
                
            score_info = "{:>15}: {:.2f}".format(self.eval_args.task.split('/')[-1], 100 * right / len(dataset))
            print(score_info)
            
            if self.eval_args.save_dir is not None:
                os.makedirs(self.eval_args.save_dir, exist_ok=False)
                with open(os.path.join(self.eval_args.save_dir, "results.json"), "w", encoding="utf-8", newline="\n") as f:
                    json.dump(results, f, indent=4)
                with open(os.path.join(self.eval_args.save_dir, "fail.json"), "w", encoding="utf-8", newline="\n") as f:
                    json.dump(fail_items, f, indent=4)
                with open(os.path.join(self.eval_args.save_dir, "results.log"), "w", encoding="utf-8", newline="\n") as f:
                    f.write(score_info)
        
        
        
def run_eval() -> None:
    evaluator = Evaluator()
    if "gen" in evaluator.eval_args.lang:
        print("\nRunning eval_gen()...")
        evaluator.eval_gen()
    elif "know" in evaluator.eval_args.task:
        print("\nRunning my_eval()...")
        evaluator.my_eval()
    else:
        evaluator.eval()
