import torch
import numpy as np
from .inference import generate_completions, get_results
import json
import os
import re
import string
from collections import Counter
from .eval_bbh import bbh_temp, bbh, bbh_temp_full
from transformers import StoppingCriteria, GenerationConfig
import warnings


class KeyWordsCriteria(StoppingCriteria):
    def __init__(self, stop_id_sequences):
        assert isinstance(stop_id_sequences[0], list), "stop_id_sequences should be a list of list of ids"
        self.stop_sequences = stop_id_sequences

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        sequences_should_be_stopped = []
        for i in range(input_ids.shape[0]):
            sequence_should_be_stopped = False
            for stop_sequence in self.stop_sequences:
                if input_ids[i][-len(stop_sequence):].tolist() == stop_sequence:
                    sequence_should_be_stopped = True
                    break
            sequences_should_be_stopped.append(sequence_should_be_stopped)
        return all(sequences_should_be_stopped)

eval_batch_size = 16

class Metric:
    def __init__(self, name, **kwargs):
        self.name = name
        self.store_individual_scores = False

    def __call__(self, predictions, references, questions=None, ids=None):
        raise NotImplementedError()

    @classmethod
    def _normalize_text(cls, text):
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        text = text.lower()
        text = "".join(char for char in text if char not in set(string.punctuation))
        text = re.sub(regex, " ", text)
        text = " ".join(text.split())
        return text

    def _get_tokens(self, text):
        if not text:
            return []
        return self._normalize_text(text).split()

class F1(Metric):
    """Computes average F1 score between a list of predictions and a list of
    list of references.

    Code taken from: https://github.com/McGill-NLP/topiocqa
    """

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

    def __call__(self, predictions, references, questions=None, ids=None):
        scores = [
            self._f1(prediction, reference)
            for prediction, reference in zip(predictions, references)
        ]
        return {"f1": np.mean(scores)}

    def _f1(self, prediction, references):
        """Computes F1 score between a prediction and a list of references.
        Take the max F1 score if there are multiple references.
        """

        f1_scores = [self._f1_score(prediction, reference) for reference in references]
        return max(f1_scores)

    def _f1_score(self, prediction, reference):
        reference_tokens = self._get_tokens(reference)
        prediction_tokens = self._get_tokens(prediction)

        common_tokens = Counter(reference_tokens) & Counter(prediction_tokens)

        num_common = sum(common_tokens.values())

        if len(reference_tokens) == 0 or len(prediction_tokens) == 0:
            # If either is empty, then F1 is 1 if they agree, 0 otherwise.
            return int(reference_tokens == prediction_tokens)

        if num_common == 0:
            return 0

        precision = 1.0 * num_common / len(prediction_tokens)
        recall = 1.0 * num_common / len(reference_tokens)
        f1 = (2 * precision * recall) / (precision + recall)

        return f1

class EM(Metric):
    """Computes average exact match score between a list of predictions and a
    list of list of references.
    """

    def __init__(self, name, **kwargs):
        super().__init__(name, **kwargs)

    def __call__(self, predictions, references, questions=None, ids=None):
        scores = [
            self._exact_match(prediction, reference)
            for prediction, reference in zip(predictions, references)
        ]
        return {"em": np.mean(scores)}

    def _exact_match(self, prediction, references):
        """Computes exact match score between a prediction and a list of
        references. Take the max EM score if there are multiple references.
        """

        em_scores = [
            self._exact_match_score(prediction, reference) for reference in references
        ]
        return max(em_scores)

    def _exact_match_score(self, prediction, reference):
        reference_tokens = self._get_tokens(reference)
        prediction_tokens = self._get_tokens(prediction)

        return int(reference_tokens == prediction_tokens)

few_prompt = """
Question: The sanctions against the school were a punishing blow, and they seemed to what the efforts the school had made to change?\nA. ignore\nB. enforce\nC. authoritarian\nD. yell at\nE. avoid
Answer: Let's think step by step. To determine the correct answer, we need to evaluate how sanctions against the school would relate to the efforts the school had made to change. Let's analyze each option:
A. Ignore - Sanctions could be seen as ignoring the efforts the school had made to change, as they imply the efforts were not considered sufficient or relevant.
B. Enforce - This does not fit well with the context since sanctions are a form of punishment, not enforcement of efforts.
C. Authoritarian - This is an adjective and does not grammatically fit the sentence.
D. Yell at - This is not appropriate in the context of sanctions.
E. Avoid - Avoiding efforts does not make sense in the context of imposing sanctions.
Given these considerations, the most logical answer is A. ignore.
Therefore, the answer is \\boxed{A}.

Question: Sammy wanted to go to where the people were.  Where might he go?\nA. race track\nB. populated areas\nC. the desert\nD. apartment\nE. roadblock
Answer: Let's think step by step. To determine where Sammy might go to find where the people are, we need to evaluate each option based on common sense:
A. Race track - A race track can have a lot of people, especially during events.
B. Populated areas - By definition, populated areas have a lot of people.
C. The desert - The desert is typically sparsely populated and not where people commonly gather.
D. Apartment - While an apartment building may have people, it is not necessarily a place where people are gathered in large numbers.
E. Roadblock - A roadblock is not a place where people gather; it is usually a temporary setup to control traffic.
Given these considerations, the most logical answer is B. populated areas. 
Therefore, the answer is \\boxed{B}.

Question: Google Maps and other highway and street GPS services have replaced what?\nA. united states\nB. mexico\nC. countryside\nD. atlas\nE. oceans
Answer: Let's think step by step. To determine what Google Maps and other highway and street GPS services have replaced, we need to evaluate each option based on common sense and the functionality of these services:
A. United States - This is a country and not something that can be replaced by GPS services.
B. Mexico - Similar to the United States, this is a country and not relevant to the context.
C. Countryside - The countryside is a type of area and cannot be replaced by GPS services.
D. Atlas - An atlas is a collection of maps and is directly related to navigation, which GPS services have indeed replaced.
E. Oceans - Oceans are large bodies of water and cannot be replaced by GPS services.
Given these considerations, the most logical answer is D. atlas.
Therefore, the answer is \\boxed{D}.
"""

support_sets = {
    'commonsenseqa': [
    {
        'question': "Question: The sanctions against the school were a punishing blow, and they seemed to what the efforts the school had made to change?\nA. ignore\nB. enforce\nC. authoritarian\nD. yell at\nE. avoid\nAnswer: ",
        'rational': '''Let's think step by step. To determine the correct answer, we need to evaluate how sanctions against the school would relate to the efforts the school had made to change. Let's analyze each option:\
\nA. Ignore - Sanctions could be seen as ignoring the efforts the school had made to change, as they imply the efforts were not considered sufficient or relevant.\
\nB. Enforce - This does not fit well with the context since sanctions are a form of punishment, not enforcement of efforts.\
\nC. Authoritarian - This is an adjective and does not grammatically fit the sentence.\
\nD. Yell at - This is not appropriate in the context of sanctions.\
\nE. Avoid - Avoiding efforts does not make sense in the context of imposing sanctions.\
\nGiven these considerations, the most logical answer is A. ignore.\
\nTherefore, the answer is \\boxed{A}.''',
        'answer': '\\boxed{A}'
    },
    {
        'question': "Question: Sammy wanted to go to where the people were.  Where might he go?\nA. race track\nB. populated areas\nC. the desert\nD. apartment\nE. roadblock\nAnswer: ",
        'rational': '''Let's think step by step. To determine where Sammy might go to find where the people are, we need to evaluate each option based on common sense:\
\nA. Race track - A race track can have a lot of people, especially during events.\
\nB. Populated areas - By definition, populated areas have a lot of people.\
\nC. The desert - The desert is typically sparsely populated and not where people commonly gather.\
\nD. Apartment - While an apartment building may have people, it is not necessarily a place where people are gathered in large numbers.\
\nE. Roadblock - A roadblock is not a place where people gather; it is usually a temporary setup to control traffic.\
\nGiven these considerations, the most logical answer is B. populated areas.\
\nTherefore, the answer is \\boxed{B}.''',
        'answer': '\\boxed{B}'
    },
    {
        'question': "Question: Google Maps and other highway and street GPS services have replaced what?\nA. united states\nB. mexico\nC. countryside\nD. atlas\nE. oceans\nAnswer: ",
        'rational': '''Let's think step by step. To determine what Google Maps and other highway and street GPS services have replaced, we need to evaluate each option based on common sense and the functionality of these services:\
\nA. United States - This is a country and not something that can be replaced by GPS services.\
\nB. Mexico - Similar to the United States, this is a country and not relevant to the context.\
\nC. Countryside - The countryside is a type of area and cannot be replaced by GPS services.\
\nD. Atlas - An atlas is a collection of maps and is directly related to navigation, which GPS services have indeed replaced.\
\nE. Oceans - Oceans are large bodies of water and cannot be replaced by GPS services.\
\nGiven these considerations, the most logical answer is D. atlas.\
\nTherefore, the answer is \\boxed{D}.''',
        'answer': '\\boxed{D}'
    },
],
    'bbh_temp': bbh_temp,
    'bbh_temp_full': bbh_temp_full,
    'bbh': bbh,
    'bb': {},
    'agieval': {},
    'gsm8k': {},
    'gsm8k+': {},
    'arc-c': {},
    'arc-e': {}
    
}
@torch.no_grad()
def eval_csqa_chat(model, tokenizer, inputs, generation_args, eval_args, use_prompt=True):
    tokenizer.padding_side = 'left'
    prompts = []
    questions = []
    answers = []
    labels = []
    choice_tem = "\n{choice}. {content}"
    generation_kwargs = {
        "do_sample": generation_args.do_sample,
        "temperature": generation_args.temperature,
        "top_p": generation_args.top_p,
        "max_length": generation_args.max_length,
        "max_new_tokens": generation_args.max_new_tokens,
        "repetition_penalty": generation_args.repetition_penalty,
        "num_beams": generation_args.num_beams,
        "length_penalty": generation_args.length_penalty,
    }    
    
    # for sample in dataset:
    #     question = sample['question']
    #     prompt = few_prompt + 'Question: {}\n'.format(question)
    #     candidates = [choice_tem.format(choice=chr(ord("A") + id), content=ch) for id, ch in enumerate(sample["choices"])]
    #     prompt += "".join(candidates)
    #     if use_prompt:
    #         prompt = "The following are multiple choice questions (with answers) about {}. "\
    #                  "Choose a correct answer that appears in the candidate answers. You need to explain your "\
    #                  "reasoning first. Your final answer should be in the form \\\boxed{{answer}} at the "\
    #                  "end of your response.\n\n".format(subject) + prompt
    #     messages = [{"role": "user", "content": prompt}]
    #     prompt = tokenizer.apply_chat_template(messages,
    #                                            tokenize=False,
    #                                            add_generation_prompt=True)
    #     prompt += "\nAnwser:"
    #     prompts.append(prompt)
    #     answers.append(sample['answer'])
    #     labels.append(chr(ord("A") + sample["choices"].index(sample["answer"])))
    #     sample['label'] = chr(ord("A") + sample["choices"].index(sample["answer"]))
    #     questions.append(sample)

    terminators = [
        [tokenizer.eos_token_id],
        [tokenizer.convert_tokens_to_ids("<|eot_id|>")],
    ]

    if eval_args.vllm:
        from vllm import LLM, SamplingParams
        sampling_params = SamplingParams(temperature=generation_args.temperature, 
                                         max_tokens=generation_args.max_new_tokens,
                                         top_p=generation_args.top_p,
                                         stop_token_ids=[tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
                                         )
        outputs = model.generate(
            prompt_token_ids=prompts,
            sampling_params=sampling_params,
        )
        outputs = [output.outputs[0].text for output in outputs]
    else: 
        outputs = generate_completions(
            model=model,
            tokenizer=tokenizer,
            prompts=prompts,
            batch_size=eval_args.batch_size,
            stop_id_sequences=terminators,
            **generation_kwargs
        )
    return outputs


@torch.no_grad()
def eval_csqa(model, tokenizer, batch_input, generation_args):
    generation_kwargs = {
        "do_sample": generation_args.do_sample,
        "temperature": generation_args.temperature,
        "top_p": generation_args.top_p,
        "max_length": generation_args.max_length,
        "max_new_tokens": generation_args.max_new_tokens,
        "repetition_penalty": generation_args.repetition_penalty,
        "num_beams": generation_args.num_beams,
        "length_penalty": generation_args.length_penalty,
    }
    # print(generation_kwargs)
    tokenizer.padding_side = 'left'

    stop_id_sequences = [
        [tokenizer.eos_token_id],
        [tokenizer.convert_tokens_to_ids("<|eot_id|>")],
    ]
    num_return_sequences = generation_kwargs.get("num_return_sequences", 1)
    batch_input_ids = batch_input['input_ids']
    try:
        batch_outputs = model.generate(
            **batch_input,
            eos_token_id=tokenizer.eos_token_id,
            stopping_criteria=[KeyWordsCriteria(stop_id_sequences)] if stop_id_sequences else None,
            **generation_kwargs
        )

        # the stopping criteria is applied at batch level, so if other examples are not stopped, the entire batch will continue to generate.
        # so some outputs still have the stop sequence, which we need to remove.
        if stop_id_sequences:
            for output_idx in range(batch_outputs.shape[0]):
                for token_idx in range(batch_input_ids.shape[1], batch_outputs.shape[1]):
                    if any(batch_outputs[output_idx,
                            token_idx: token_idx + len(stop_sequence)].tolist() == stop_sequence for stop_sequence in
                            stop_id_sequences):
                        batch_outputs[output_idx, token_idx:] = tokenizer.pad_token_id
                        break

        # remove the prompt from the output
        # we need to re-encode the prompt because we need to make sure the special tokens are treated the same way as in the outputs.
        # we changed our previous way of truncating the output token ids dicrectly because some tokenizer (e.g., llama) won't add space token before the first token.
        # space is important for some tasks (e.g., code completion).
        batch_outputs = tokenizer.batch_decode(batch_outputs, skip_special_tokens=True)
        batch_prompts = tokenizer.batch_decode(batch_input_ids, skip_special_tokens=True)
        # duplicate the prompts to match the number of return sequences
        batch_prompts = [prompt for prompt in batch_prompts for _ in range(num_return_sequences)]
        batch_generations = [
            output[len(prompt):] for prompt, output in zip(batch_prompts, batch_outputs)
        ]
    except Exception as e:
        print("Error when generating completions for batch:")
        print(batch_prompts)
        print("Error message:")
        print(e)
        print("Use empty string as the completion.")
        batch_generations = [""] * len(batch_prompts) * num_return_sequences

        # for prompt, generation in zip(batch_prompts, batch_generations):
        #     print("========")
        #     print(prompt)
        #     print("--------")
        #     print(generation)


    return batch_generations