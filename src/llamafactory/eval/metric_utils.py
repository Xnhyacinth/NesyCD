import re
from rouge import Rouge
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import statistics
import copy
from bert_score import score
import sys
sys.path.append('../..')
from tqdm import tqdm
import torch
# nltk.download('punkt') 

def find_last_innermost_parentheses(text):
    last_open_parenthesis = text.rfind('(')
    
    if last_open_parenthesis == -1:
        return None 
    
    last_close_parenthesis = text.rfind(')', last_open_parenthesis)
    
    if last_close_parenthesis == -1:
        return None 
    
    innermost_content = text[last_open_parenthesis + 1:last_close_parenthesis]
    return innermost_content

def extract_answer(s):
    _PAT_LAST_DIGIT = re.compile(
        r"([+-])?(?=([0-9]|\.[0-9]))(0|([1-9](\d{0,2}(,\d{3})*)|\d*))?(\.\d*)?(?=\D|$)"
    )
    match = list(_PAT_LAST_DIGIT.finditer(s))
    if match:
        last_digit = match[-1].group().replace(",", "").replace("+", "").strip()
    else:
        last_digit = 'none'
        print(f"No digits found in {s!r}", flush=True)
    return 'none', last_digit.rstrip('.')

def extract_answers_for_model(model_output):
    if model_output is None:
        return 'none', 'none'
    model_output = model_output.strip()
    model_output = model_output.rstrip('.')
    model_output = model_output.lower()
    model_output = model_output.replace(u')', ')').replace(u'(', '(')

    if len(model_output) == 1:
        md_choice = f'({model_output})'
        md_content = 'none'
        # if re.search(r'[0-9]', model_output):
        md_content = model_output
    else:
        pattern = r'\([a-z]\)'
        match = re.search(pattern, model_output)
        if match:
            md_choice = match.group(0)
            sp_list = model_output.split(')')
            if len(sp_list) == 1:
                md_content = 'none'
            else:
                md_content = sp_list[1]
                md_content = md_content.lstrip('.')
                md_content = md_content.rstrip('.')
                md_content = md_content.strip()
        else:
            pattern = r'\([1-9]\)'
            match = re.search(pattern, model_output)
            if match:
                md_choice = match.group(0)
                sp_list = model_output.split(')')
                if len(sp_list) == 1:
                    md_content = 'none'
                else:
                    md_content = sp_list[1]
                    md_content = md_content.lstrip('.')
                    md_content = md_content.rstrip('.')
                    md_content = md_content.strip()
            else:
                if '(' not in model_output and ')' not in model_output:
                    pattern = r'option [a-z]'
                    match = re.search(pattern, model_output)
                    if match:
                        md_choice = match.group(0)
                        sp_list = model_output.split(')')
                        if len(sp_list) == 1:
                            md_content = 'none'
                        else:
                            md_content = sp_list[1]
                            md_content = md_content.lstrip('.')
                            md_content = md_content.rstrip('.')
                            md_content = md_content.strip()
                    else:
                        md_choice = 'none'
                        md_content = model_output.lstrip('.')
                        md_content = md_content.rstrip('.')
                        md_content = md_content.strip()
                else:
                    pattern = r'\(?[a-z]\)?'
                    match = re.search(pattern, model_output)
                    if match:
                        choice = re.search(r'[a-z]', match[0])
                        md_choice = f'({choice.group(0)})'
                        sp_list = model_output.split(')')
                        if len(sp_list) == 1:
                            md_content = 'none'
                        else:
                            md_content = sp_list[1]
                            md_content = md_content.lstrip('.')
                            md_content = md_content.rstrip('.')
                            md_content = md_content.strip()
                    else:
                        # print(model_output)
                        md_choice = 'none'
                        md_content = model_output.lstrip('.')
                        md_content = md_content.rstrip('.')
                        md_content = md_content.strip()
                    

    return md_choice, md_content

def extract_answers_for_gt(original_output):
    original_output = original_output.strip()
    original_output = original_output.rstrip('.')
    original_output = original_output.lower()
    original_output = original_output.replace(u')', ')').replace(u'(', '(')

    pattern = r'\([a-z]\)'
    match = re.search(pattern, original_output)
    if match:
        gt_choice = match.group(0)
        sp_list = original_output.split(')')
        if len(sp_list) == 1:
            gt_content = 'none'
        else:
            gt_content = sp_list[1]
            gt_content = gt_content.lstrip('.')
            gt_content = gt_content.rstrip('.')
            gt_content = gt_content.strip()
    else:
        pattern = r'\([1-9]\)'
        match = re.search(pattern, original_output)
        if match:
            gt_choice = match.group(0)
            sp_list = original_output.split(')')
            if len(sp_list) == 1:
                gt_content = 'none'
            else:
                gt_content = sp_list[1]
                gt_content = gt_content.lstrip('.')
                gt_content = gt_content.rstrip('.')
                gt_content = gt_content.strip()
        else:
            gt_choice = 'none'
            gt_content = original_output.lstrip('.')
            gt_content = gt_content.rstrip('.')
            gt_content = gt_content.strip()

    return gt_choice, gt_content

def decide(ground_truth: str, model_answer: str):
    # for formal_fallacies task check

    if 'invalid' in ground_truth:
        if 'invalid' in model_answer:
            return True
        else:
            return False
    else:
        if 'invalid' in model_answer:
            return False
        else:
            return True

def compute_metrics(task_descs, task_names, teacher_responses, model_inputs, original_inputs, original_outputs, model_outputs, eval_preds):
    
    stu_responses = copy.deepcopy(eval_preds)
    for i, x in enumerate(stu_responses):
        stu_responses[i] = x.split(original_inputs[i])[-1].lstrip('\n').lstrip('Answer:').strip().lstrip('Rationale:').strip()
    
    # Unused, you can load model to compute the reward score of responses of students and teachers
    stu_reward_scores = []
    tea_reward_scores = []

    cnt = 0
    while cnt < 20:
        torch.cuda.empty_cache()
        cnt += 1

    task_name_map = {}
    for i in task_names:
        if i not in task_name_map.keys():
            task_name_map[i] = 1
        else:
            task_name_map[i] += 1

    sub_task_acc_dict = copy.deepcopy(task_name_map)
    for i in sub_task_acc_dict.keys():
        sub_task_acc_dict[i] = 0

    if 'gsm8k' not in task_names[0]:
        for i, x in enumerate(model_outputs):
            # print(model_outputs[i])
            x = x.split('Question: ')[0]
            # pattern = r'\([A-Z]\)'
            # if re.search(pattern, x.split('\n\n')[0]):
            #     model_outputs[i] = x.split('\n\n')[0]
            if 'Therefore, the answer is' in x:
                model_outputs[i] = x.split('Therefore, the answer is')[1].split('.')[0].split('\n\n')[0].strip()
            elif '[Answer Prediction]:' in x:
                model_outputs[i] = x.split('[Answer Prediction]:')[1].split('.')[0].strip()
            elif 'the correct answer is option' in x:
                model_outputs[i] = x.split('the correct answer is option')[1].split('.')[0].split('\n\n')[0].strip()
            elif 'the correct answer is' in x:
                model_outputs[i] = x.split('the correct answer is')[1].split('.')[0].split('\n\n')[0].strip()
            elif 'The correct answer is' in x:
                model_outputs[i] = x.split('The correct answer is')[1].split('.')[0].split('\n\n')[0].strip()
            elif 'The answer is' in x:
                model_outputs[i] = x.split('The answer is')[1].split('.')[0].split('\n\n')[0].strip()
            elif 'Answer:' in x and 'Explanation' not in x:
                model_outputs[i] = x.split('Answer:')[1].split('.')[0].split('\n\n')[0].strip()
            elif 'A:' in x and 'Explanation' not in x:
                model_outputs[i] = x.split('A:')[1].split('.')[0].split('\n\n')[0].strip()
            elif 'Answer:' in x and 'Explanation' in x:
                model_outputs[i] = x.split('Explanation')[0].split('Answer:')[-1].split('.')[0].strip()
            elif 'A:' in x and 'Explanation' in x:
                model_outputs[i] = x.split('Explanation')[0].split('A:')[-1].split('.')[0].strip()
            elif 'The answer is' in x and 'Explanation' in x:
                model_outputs[i] = x.split('Explanation')[0].split('The answer is')[-1].split('.')[0].strip()
            elif '\n\nAnswer:' in x:
                model_outputs[i] = x.split('\n\nAnswer:')[1].split('.')[0].split('\n\n')[0].strip()
            elif '\n\nA:' in x:
                model_outputs[i] = x.split('\n\nA:')[1].split('.')[0].split('\n\n')[0].strip()
            elif '### Response:' in x:
                model_outputs[i] = x.split('### Response:')[1].split('.')[0].strip()
            elif 'the answer is' in x:
                model_outputs[i] = x.split('the answer is')[1].split('.')[0].split('\n\n')[0].strip()
            elif 'answer is' in x:
                model_outputs[i] = x.split('answer is')[1].split('.')[0].split('\n\n')[0].strip()
            else:
                model_outputs[i] = x.split('.')[0].strip()
        # print(model_outputs[i])
    correct_count = 0
    formatted_gt_answers = []
    original_extracted_model_answers = []
    formatted_extracted_model_answers = []
    right_scores = []
    for i, x in enumerate(model_outputs):
        original_extracted_model_answers.append(x)
        if task_names[i] != 'formal_fallacies':
            # simple check
            # if original_outputs[i] in x or x in original_outputs[i]:
            #     correct_count += 1
            #     sub_task_acc_dict[task_names[i]] += 1

            # more hard check
            if 'gsm8k' in task_names[i] or 'GSM' in task_names[i] or 'MATH' in task_names[i]:
                gt_choice, gt_content = extract_answer(copy.deepcopy(original_outputs[i]))
                md_choice, md_content = extract_answer(copy.deepcopy(x))
            else:
                gt_choice, gt_content = extract_answers_for_gt(copy.deepcopy(original_outputs[i]))
                md_choice, md_content = extract_answers_for_model(copy.deepcopy(x))

            # note: gt choice must be parthed with '()', like '(A)'
            # but md choice are not required, likr 'A'

            if md_choice and md_choice != 'none' and gt_choice and gt_choice != 'none':
                if md_choice in gt_choice:
                    # check choice
                    right_scores.append(1)
                    correct_count += 1
                    sub_task_acc_dict[task_names[i]] += 1
                else:
                    right_scores.append(0)
            else:
                # check content
                if md_content == 'none' or md_content == '':
                    right_scores.append(0)  
                elif gt_content == md_content:
                    right_scores.append(1)
                    correct_count += 1
                    sub_task_acc_dict[task_names[i]] += 1
                # elif gt_content in md_content and len(gt_content) * 10 > len(md_content):
                #     right_scores.append(1)
                #     correct_count += 1
                #     sub_task_acc_dict[task_names[i]] += 1
                else:
                    right_scores.append(0)

        else:
            gt_choice = 'none'
            gt_content = original_outputs[i].lower().strip()
            md_choice = 'none'
            md_content = x.lower().strip()
            
            if decide(ground_truth=gt_content, model_answer=md_content):
                right_scores.append(1)
                correct_count += 1
                sub_task_acc_dict[task_names[i]] += 1
            else:
                right_scores.append(0)
        
        formatted_gt_answers.append(f'Choice:{gt_choice}, Content:{gt_content}')
        formatted_extracted_model_answers.append(f'Choice:{md_choice}, Content:{md_content}')
    
    
    for i in sub_task_acc_dict.keys():
        sub_task_acc_dict[i] = sub_task_acc_dict[i] / task_name_map[i]

    total_num = len(original_outputs)
    accuracy = correct_count / total_num

    rouge_scores = {}
    bleu_mean = 0.0
    
    bert_scores = {
        'p': 'null',
        'r': 'null',
        'f': 'null'
    }

    hmean = 0.0

    results = {
        'description': 'macro test results',
        "right_num": correct_count,
        "total_num": total_num,
        "total_accuracy": accuracy,
        "sub task accuracy": sub_task_acc_dict,
        "bleu_scores": bleu_mean,
        "rouge_scores": rouge_scores,
        "bert_score": bert_scores,
        "hmean": hmean
    }

    details = []
    for i, x in enumerate(original_inputs):
        details.append({
            "task_description": task_descs[i],
            "task_name": task_names[i],
            "instruction": original_inputs[i],
            "input": "",
            "output": original_outputs[i] if '(' not in original_outputs[i] and ')' not in original_outputs[i] else original_outputs[i].split(')')[0] + ')',
            "formatted_gt_answer": formatted_gt_answers[i],
            "formatted_model_answer": formatted_extracted_model_answers[i],
            "original_model_answer": original_extracted_model_answers[i],
            "original_model_predict": eval_preds[i],
            "stu_right_score": right_scores[i],
            "question": model_inputs[i],
            "stu_response": stu_responses[i],
            "tea_response": teacher_responses[i], 
        })

    return results, details

def compute_acc(model_outputs, original_outputs):
    extracted_model_answers, model_match_num, model_not_match_num, model_not_match_list = extract_answers_for_model(model_outputs)
    extracted_original_answers, original_match_num, original_not_match_num, _ = extract_answers_for_gt(original_outputs)

    correct_count = sum(1 for model_answer, original_answer in zip(extracted_model_answers, extracted_original_answers)
                         if model_answer == original_answer)
    not_right_outputs = []
    for i, x in enumerate(extracted_model_answers):
        if extracted_model_answers[i] != extracted_original_answers[i]:
            not_right_outputs.append(model_outputs[i])
    total_num = len(original_outputs)
    accuracy = correct_count / total_num
    results = {
        'model_not_match_list': model_not_match_list,
        "model_match_num": model_match_num,
        "model_not_match_num": model_not_match_num,
        "original_match_num": original_match_num,
        "original_not_match_num": original_not_match_num,
        'not_right_outputs': not_right_outputs,
        "total_num": total_num,
        "accuracy": accuracy,
    }
    print(results)

    return results