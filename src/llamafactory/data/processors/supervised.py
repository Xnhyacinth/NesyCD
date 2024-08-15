from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ...extras.constants import IGNORE_INDEX
from ...extras.logging import get_logger
from .mm_utils import get_paligemma_token_type_ids, get_pixel_values
from torch.nn.utils.rnn import pad_sequence

if TYPE_CHECKING:
    from transformers import ProcessorMixin
    from transformers.tokenization_utils import PreTrainedTokenizer

    from ...hparams import DataArguments
    from ..template import Template


logger = get_logger(__name__)


def preprocess_supervised_dataset(
    examples: Dict[str, List[Any]],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    data_args: "DataArguments",
) -> Dict[str, List[List[int]]]:
    # build inputs with format `<bos> X Y <eos>` and labels with format `<ignore> ... <ignore> Y <eos>`
    # for multiturn examples, we only mask the prompt part in each prompt-response pair.
    if 'mt-' in data_args.mt:
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": [], "rationale_input_ids": [], "rationale_attention_mask": [], "rationale_labels": []}
    else:
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
    if processor is not None:
        model_inputs["pixel_values"] = []
        if hasattr(processor, "image_seq_length"):  # paligemma models
            model_inputs["token_type_ids"] = []

    for i in range(len(examples["prompt"])):
        if len(examples["prompt"][i]) % 2 != 1 or len(examples["response"][i]) != 1:
            logger.warning("Dropped invalid example: {}".format(examples["prompt"][i] + examples["response"][i]))
            continue

        if processor is not None and not hasattr(processor, "image_seq_length"):  # llava-like models
            examples["prompt"][i][0]["content"] = template.image_token + examples["prompt"][i][0]["content"]

        if 'mt-' in data_args.mt:
            input_ids, labels = [], []
            rationale_input_ids, rationale_labels = [], []
            response = examples["response"][i][0]['content']
            prompt = examples["prompt"][i][0]['content'].split('Your response should conclude with the format')[0].strip()
            question = examples["prompt"][i][0]['content'].split('\n\nQuestion: ')[1].split('\nAnswer: ')[0].strip()
            rationale = response.split('Therefore, the answer is')[0].strip()
            prompt += "\n\nQuestion: " + question
            answer = response.split('Therefore, the answer is')[1].strip()
            if data_args.mt == 'mt-re':
                instruction_rationale = prompt + '\nRationale:'
                instruction_answer = prompt + '\nAnswer:'
            elif data_args.mt == 'mt-ra':
                instruction_rationale = prompt + '\n[Explanation Generation]:'
                instruction_answer = prompt + '\n[Answer Prediction]:'
                rationale = 'The answer is ' + answer + ' Explanation: ' + rationale
            elif data_args.mt == 'mt-cot':
                instruction_rationale = prompt + '\n[Explanation Generation]:'
                instruction_answer = prompt + '\n[Answer Prediction]:'
                rationale = response
            elif data_args.mt == 'mt-cas':
                instruction_rationale = prompt + '\nRationale:'
                instruction_answer = prompt + ' ' + rationale + '\nTherefore, the answer is'
                
            messages_rationale = [{'role': 'user', 'content': instruction_rationale}] + [{'role': 'assistant', 'content': rationale}]
            messages = [{'role': 'user', 'content': instruction_answer}] + [{'role': 'assistant', 'content': answer}]
            for turn_idx, (source_ids, target_ids) in enumerate(
                template.encode_multiturn(
                    tokenizer,
                    messages_rationale,
                    examples["system"][i],
                    examples["tools"][i],
                    data_args.cutoff_len,
                    data_args.reserved_label_len,
                )
            ):
                if data_args.train_on_prompt:
                    source_mask = source_ids
                elif turn_idx != 0 and template.efficient_eos:
                    source_mask = [tokenizer.eos_token_id] + [IGNORE_INDEX] * (len(source_ids) - 1)
                else:
                    source_mask = [IGNORE_INDEX] * len(source_ids)

                rationale_input_ids += source_ids + target_ids
                rationale_labels += source_mask + target_ids
                
            for turn_idx, (source_ids, target_ids) in enumerate(
                template.encode_multiturn(
                    tokenizer,
                    messages,
                    examples["system"][i],
                    examples["tools"][i],
                    data_args.cutoff_len,
                    data_args.reserved_label_len,
                )
            ):
                if data_args.train_on_prompt:
                    source_mask = source_ids
                elif turn_idx != 0 and template.efficient_eos:
                    source_mask = [tokenizer.eos_token_id] + [IGNORE_INDEX] * (len(source_ids) - 1)
                else:
                    source_mask = [IGNORE_INDEX] * len(source_ids)

                input_ids += source_ids + target_ids
                labels += source_mask + target_ids
        else:
            messages = examples["prompt"][i] + examples["response"][i]
            input_ids, labels = [], []

            if processor is not None and hasattr(processor, "image_seq_length"):  # paligemma models
                image_token_id = tokenizer.convert_tokens_to_ids(template.image_token)
                input_ids += [image_token_id] * getattr(processor, "image_seq_length")
                labels += [IGNORE_INDEX] * getattr(processor, "image_seq_length")

            for turn_idx, (source_ids, target_ids) in enumerate(
                template.encode_multiturn(
                    tokenizer,
                    messages,
                    examples["system"][i],
                    examples["tools"][i],
                    data_args.cutoff_len,
                    data_args.reserved_label_len,
                )
            ):
                if data_args.train_on_prompt:
                    source_mask = source_ids
                elif turn_idx != 0 and template.efficient_eos:
                    source_mask = [tokenizer.eos_token_id] + [IGNORE_INDEX] * (len(source_ids) - 1)
                else:
                    source_mask = [IGNORE_INDEX] * len(source_ids)

                input_ids += source_ids + target_ids
                labels += source_mask + target_ids

        if template.efficient_eos:
            input_ids += [tokenizer.eos_token_id]
            labels += [tokenizer.eos_token_id]
            if 'mt-' in data_args.mt:
                rationale_input_ids += [tokenizer.eos_token_id]
                rationale_labels += [tokenizer.eos_token_id]
        
        if 'mt-' in data_args.mt:
            if tokenizer.name_or_path != 'meta-llama/Llama-2-7b-hf':
                if data_args.mt == 'mt-cas':
                    rationale_input_ids += [tokenizer.pad_token_id] * (len(input_ids) - len(rationale_input_ids))
                    rationale_labels += [IGNORE_INDEX] * (len(labels) - len(rationale_labels))
                else:
                    input_ids += [tokenizer.pad_token_id] * (len(rationale_input_ids) - len(input_ids))
                    labels += [IGNORE_INDEX] * (len(rationale_labels) - len(labels))
            model_inputs["rationale_input_ids"].append(rationale_input_ids)
            model_inputs["rationale_attention_mask"].append([1] * len(rationale_input_ids))
            model_inputs["rationale_labels"].append(rationale_labels)
            
        model_inputs["input_ids"].append(input_ids)
        model_inputs["attention_mask"].append([1] * len(input_ids))
        model_inputs["labels"].append(labels)
        
        if processor is not None:
            model_inputs["pixel_values"].append(get_pixel_values(examples["images"][i], processor))
            if hasattr(processor, "image_seq_length"):  # paligemma models
                model_inputs["token_type_ids"].append(get_paligemma_token_type_ids(len(input_ids), processor))

    return model_inputs


def preprocess_packed_supervised_dataset(
    examples: Dict[str, List[Any]],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    data_args: "DataArguments",
) -> Dict[str, List[List[int]]]:
    # build inputs with format `<bos> X1 Y1 <eos> <bos> X2 Y2 <eos>`
    # and labels with format `<ignore> ... <ignore> Y1 <eos> <ignore> ... <ignore> Y2 <eos>`
    model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
    input_ids, labels = [], []
    for i in range(len(examples["prompt"])):
        if len(examples["prompt"][i]) % 2 != 1 or len(examples["response"][i]) != 1:
            logger.warning("Dropped invalid example: {}".format(examples["prompt"][i] + examples["response"][i]))
            continue

        messages = examples["prompt"][i] + examples["response"][i]
        for source_ids, target_ids in template.encode_multiturn(
            tokenizer, messages, examples["system"][i], examples["tools"][i]
        ):
            if data_args.train_on_prompt:
                source_mask = source_ids
            elif len(input_ids) != 0 and template.efficient_eos:
                source_mask = [tokenizer.eos_token_id] + [IGNORE_INDEX] * (len(source_ids) - 1)
            else:
                source_mask = [IGNORE_INDEX] * len(source_ids)

            input_ids += source_ids + target_ids
            labels += source_mask + target_ids

    if template.efficient_eos:
        input_ids += [tokenizer.eos_token_id]
        labels += [tokenizer.eos_token_id]    

    total_length = len(input_ids)
    block_size = data_args.cutoff_len
    # we drop the small remainder, and if the total_length < block_size, we exclude this batch
    total_length = (total_length // block_size) * block_size
    # split by chunks of cutoff_len
    for i in range(0, total_length, block_size):
        if not all(label == IGNORE_INDEX for label in labels[i : i + block_size]):
            model_inputs["input_ids"].append(input_ids[i : i + block_size])
            model_inputs["attention_mask"].append([1] * block_size)
            model_inputs["labels"].append(labels[i : i + block_size])

    return model_inputs


def print_supervised_dataset_example(example: Dict[str, List[int]], tokenizer: "PreTrainedTokenizer") -> None:
    valid_labels = list(filter(lambda x: x != IGNORE_INDEX, example["labels"]))
    print("input_ids:\n{}".format(example["input_ids"]))
    print("inputs:\n{}".format(tokenizer.decode(example["input_ids"], skip_special_tokens=False)))
    print("label_ids:\n{}".format(example["labels"]))
    print("labels:\n{}".format(tokenizer.decode(valid_labels, skip_special_tokens=False)))
