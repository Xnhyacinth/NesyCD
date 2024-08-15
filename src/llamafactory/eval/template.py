from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Tuple, Union
from ..data import Role
from ..extras.constants import CHOICES


@dataclass
class EvalTemplate:
    name: str 
    system: str
    choice: str
    answer: str
    prefix: str

    def _parse_example(self, example: Dict[str, str], mt=None) -> Tuple[str, str]:
        r"""
        input: a dict with keys {"question", "A", "B", "C", "D", "answer"}
        output: a tuple of (prompt, response)
        """
        if 'instruction' in example.keys():
            if 'temp' in self.name:
                return "".join(["Question: " + example["instruction"] + "\nAnswer: " + example["output"]] + [self.answer]), example["output"]
            else:
                if mt == 'mt-re':
                    return "".join(["Question: " + example["instruction"]] + [self.answer + 'Answer: ']).replace('Question: Question: ', 'Question: '), example["output"]
                elif mt == 'mt-ra' or mt == 'mt-cot':
                    return "".join(["Question: " + example["instruction"]] + [self.answer + '[Answer Prediction]: ']).replace('Question: Question: ', 'Question: '), example["output"]
                elif mt == 'mt-cas':
                    return "".join(["Question: " + example["instruction"]] + [self.answer + 'Rationale: ']).replace('Question: Question: ', 'Question: '), example["output"]
                else:
                    return "".join(["Question: " + example["instruction"]] + [self.answer]).replace('Question: Question: ', 'Question: '), example["output"]
        elif 'rational' in example.keys():
            if 'deep' in self.name:
                return example['question'].replace('Question: Question: ', 'Question: '), example["rational_deep"]
            else:
                return example['question'].replace('Question: Question: ', 'Question: '), 'Therefore, the answer is ' + example["answer"] + '.' if 'cot' not in self.name else example['rational']
        elif "A" not in example.keys():
            candidates = [self.choice.format(choice=chr(ord("A") + id), content=ch) for id, ch in enumerate(example["choices"])]
            return "".join(["Question: " + example["question"]] + candidates + [self.answer]), chr(ord("A") + example["choices"].index(example["answer"]))
        else:
            candidates = [self.choice.format(choice=ch, content=example[ch]) for ch in CHOICES if ch in example]
            return "".join([example["question"]] + candidates + [self.answer]), example["answer"]
    
    def format_example(
        self, target_data: Dict[str, str], support_set: Sequence[Dict[str, str]], subject_name: str, mt=None, num_knowledge=1
    ) -> List[Dict[str, str]]:
        r"""
        Converts dataset examples to messages.
        """
        messages = []
        for k in range(len(support_set)):
            prompt, response = self._parse_example(support_set[k])
            messages.append({"role": Role.USER.value, "content": prompt})
            messages.append({"role": Role.ASSISTANT.value, "content": response})

        prompt, response = self._parse_example(target_data, mt)
        messages.append({"role": Role.USER.value, "content": prompt})
        messages.append({"role": Role.ASSISTANT.value, "content": response})

        if 'prefix' in self.name and 'sum' in self.name:
            messages[0]["content"] = self.system.format(prefix=target_data['prefix_prompt'], task=target_data['task_description'], summary=target_data['summary']) + messages[0]["content"]
        elif 'kard' in self.name or 'kaft' in  self.name:
            knowledge = ''
            for id, know in enumerate(target_data["knowledge"][:3]):
                knowledge += f'{id + 1}. {" ".join(know.split(" . ")[1].split(" ")[: 128]).rstrip(".").strip()}.\n'
            messages[0]["content"] = self.system.format(prefix=target_data['prefix_prompt'], task=target_data['task_description'], knowledge=knowledge[:-2]) + messages[0]["content"]
        elif 'prefix' in self.name:
            if 'knowledge' in target_data.keys() and target_data['knowledge'][0]['knowledge'] != 'No need summary.':
                summary = [know['knowledge'] for know in target_data['knowledge'][:num_knowledge]]
                knowledge = "You can utilize the general learning summary and supplementary knowledge proposed by the expert teacher to address this question.\n{knowledge}\n\n".format(knowledge='\n'.join(summary))
                messages[0]["content"] = self.system.format(prefix=target_data['prefix_prompt'], task=target_data['task_description']) + knowledge + messages[0]["content"]
            else:
                messages[0]["content"] = self.system.format(prefix=target_data['prefix_prompt'], task=target_data['task_description']) + messages[0]["content"]
        elif 'sum' in self.name:
            messages[0]["content"] = self.system.format(subject=subject_name, task=target_data['task_description'], summary=target_data['summary']) + messages[0]["content"]
        elif 'bbh' in self.name or 'temp' in self.name:
            messages[0]["content"] = self.system.format(subject=subject_name, task=target_data['task_description']) + messages[0]["content"]
        else:
            messages[0]["content"] = self.system.format(subject=subject_name) + messages[0]["content"]
        return messages


eval_templates: Dict[str, "EvalTemplate"] = {}


def _register_eval_template(name: str, system: str, choice: str, answer: str, prefix: str) -> None:
    eval_templates[name] = EvalTemplate(
        name=name, system=system, choice=choice, answer=answer, prefix=prefix)


def get_eval_template(name: str) -> "EvalTemplate":
    eval_template = eval_templates.get(name, None)  
    assert eval_template is not None, "Template {} does not exist.".format(
        name)
    return eval_template


_register_eval_template(
    name="en",
    system="The following are multiple choice questions (with answers) about {subject}.\n\n",
    choice="\n{choice}. {content}",
    answer="\nAnswer: Therefore, the answer is ",
    prefix=" ",
)
# Take a deep breath and work on this problem step-by-step.


_register_eval_template(
    name="gen_bbh",
    system="The following are multiple choice questions (with answers) about {subject}. {task} Your response should contain only a one-sentence answer with the format \"Therefore, the answer is\".\n\n",
    choice="\n{choice}. {content}",
    answer="\nAnswer: ",
    prefix=" ",
)
_register_eval_template(
    name="gen_bbh_prefix",
    system="{prefix} {task} Your response should contain only a one-sentence answer with the format \"Therefore, the answer is\".\n\n",
    choice="\n{choice}. {content}",
    answer="\nAnswer: ",
    prefix=" ",
)
_register_eval_template(
    name="gen_bbh_sum",
    system="The following are multiple choice questions (with answers) about {subject}. {task} Your response should contain only a one-sentence answer with the format \"Therefore, the answer is\".\n\nYou can follow the general method proposed by the expert teacher to address this question.\n{summary}\n\n",
    choice="\n{choice}. {content}",
    answer="\nAnswer: ",
    prefix=" ",
)
_register_eval_template(
    name="gen_bbh_sum_full",
    system="The following are multiple choice questions (with answers) about {subject}. {task} Your response should contain only a one-sentence answer with the format \"Therefore, the answer is\".\n\nYou can utilize the general learning summary and supplementary knowledge proposed by the expert teacher to address this question.\n{summary}\n\n",
    choice="\n{choice}. {content}",
    answer="\nAnswer: ",
    prefix=" ",
)
_register_eval_template(
    name="gen_bbh_sum_prefix",
    system="{prefix} {task} Your response should contain only a one-sentence answer with the format \"Therefore, the answer is\".\n\nYou can follow the general method proposed by the expert teacher to address this question.\n{summary}\n\n",
    choice="\n{choice}. {content}",
    answer="\nAnswer: ",
    prefix=" ",
)
_register_eval_template(
    name="gen_bbh_sum_prefix_full",
    system="{prefix} {task} Your response should contain only a one-sentence answer with the format \"Therefore, the answer is\".\n\nYou can utilize the general learning summary and supplementary knowledge proposed by the expert teacher to address this question.\n{summary}\n\n",
    choice="\n{choice}. {content}",
    answer="\nAnswer: ",
    prefix=" ",
)


_register_eval_template(
    name="gen_bbh_cot",
    system="The following are multiple choice questions (with answers) about {subject}. {task} Your response should conclude with the format \"Therefore, the answer is\".\n\n",
    choice="\n{choice}. {content}",
    answer="\nAnswer: Let's think step by step. ",
    prefix=" ",
)
_register_eval_template(
    name="gen_bbh_cot_prefix",
    system="{prefix} {task} Your response should conclude with the format \"Therefore, the answer is\".\n\n",
    choice="\n{choice}. {content}",
    answer="\nAnswer: Let's think step by step. ",
    prefix=" ",
)
_register_eval_template(
    name="gen_bbh_cot_prefix_kard",
    system="{prefix} {task} Your response should conclude with the format \"Therefore, the answer is\".\n\nYou can try using some of the world knowledge below to help you complete the task:\n{knowledge}\n\n",
    choice="\n{choice}. {content}",
    answer="\nAnswer: Let's think step by step. ",
    prefix=" ",
)
_register_eval_template(
    name="gen_bbh_cot_prefix_kaft",
    system="{prefix} {task}\n\nYou can try using some of the world knowledge below to help you complete the task:\n{knowledge}\n\n",
    choice="\n{choice}. {content}",
    answer="\nAnswer: ",
    prefix=" ",
)
_register_eval_template(
    name="gen_bbh_cot_prefix_ft",
    system="{prefix} {task}\n\n",
    choice="\n{choice}. {content}",
    answer="\nAnswer: ",
    prefix=" ",
)
_register_eval_template(
    name="gen_bbh_cot_prefix_mt-cas",
    system="{prefix} {task}\n\n",
    choice="\n{choice}. {content}",
    answer="\n",
    prefix=" ",
)
_register_eval_template(
    name="gen_bbh_cot_prefix_mt",
    system="{prefix} {task}\n\n",
    choice="\n{choice}. {content}",
    answer="\n",
    prefix=" ",
)
_register_eval_template(
    name="gen_bbh_cot_sum",
    system="The following are multiple choice questions (with answers) about {subject}. {task} Your response should conclude with the format \"Therefore, the answer is\".\n\nYou can follow the general method proposed by the expert teacher to address this question.\n{summary}\n\n",
    choice="\n{choice}. {content}",
    answer="\nAnswer: Let's think step by step. ",
    prefix=" ",
)
_register_eval_template(
    name="gen_bbh_cot_sum_full",
    system="The following are multiple choice questions (with answers) about {subject}. {task} Your response should conclude with the format \"Therefore, the answer is\".\n\nYou can utilize the general learning summary and supplementary knowledge proposed by the expert teacher to address this question.\n{summary}\n\n",
    choice="\n{choice}. {content}",
    answer="\nAnswer: Let's think step by step. ",
    prefix=" ",
)
_register_eval_template(
    name="gen_bbh_cot_sum_sim",
    system="{task} Your response should conclude with the format \"Therefore, the answer is\".\n\nYou can follow the general method proposed by the expert teacher to address this question.\n{summary}\n\n",
    choice="\n{choice}. {content}",
    answer="\nAnswer: ",
    prefix=" ",
)
_register_eval_template(
    name="gen_bbh_cot_sum_deep",
    system="The following are multiple choice questions (with answers) about {subject}. {task} Your response should conclude with the format \"Therefore, the answer is\".\n\nYou can follow the general method proposed by the expert teacher to address this question.\n{summary}\n\n",
    choice="\n{choice}. {content}",
    answer="\nAnswer: Take a deep breath and work on this problem step-by-step. ",
    prefix=" ",
)

_register_eval_template(
    name="gen_bbh_cot_sum_prefix",
    system="{prefix} {task} Your response should conclude with the format \"Therefore, the answer is\".\n\nYou can follow the general method proposed by the expert teacher to address this question.\n{summary}\n\n",
    choice="\n{choice}. {content}",
    answer="\nAnswer: Let's think step by step. ",
    prefix=" ",
)
_register_eval_template(
    name="gen_bbh_cot_sum_prefix_deep",
    system="{prefix} {task} Your response should conclude with the format \"Therefore, the answer is\".\n\nYou can follow the general method proposed by the expert teacher to address this question.\n{summary}\n\n",
    choice="\n{choice}. {content}",
    answer="\nAnswer: Take a deep breath and work on this question step-by-step. ",
    prefix=" ",
)
_register_eval_template(
    name="gen_bbh_cot_sum_prefix_full",
    system="{prefix} {task} Your response should conclude with the format \"Therefore, the answer is\".\n\nYou can utilize the general learning summary and supplementary knowledge proposed by the expert teacher to address this question.\n{summary}\n\n",
    choice="\n{choice}. {content}",
    answer="\nAnswer: Let's think step by step. ",
    prefix=" ",
)


_register_eval_template(
    name="gen_temp_cot_full",
    system="As an expert assistant teacher, you are tasked with generating a learning summary and knowledge to aid students in successfully completing the same task in the future. The following are tasks about {subject}. Based on a thorough understanding and explanation of the question, relevant background knowledge, fundamental concepts, and empirical conclusions, please generate a learning summary in a numbered list format and supplementary knowledge in a numbered list format. {task} Please be as concise, clear, and effective as possible.\n\n",
    choice="\n{choice}. {content}",
    answer="\n",
    prefix=" ",
)

_register_eval_template(
    name="gen_temp_cot",
    system="You are an expert assistant teacher. The following are tasks about {subject}. Based on an understanding and explanation of the problem, along with relevant background knowledge, fundamental concepts, and empirical conclusions, please generate a learning summary in a numbered list format that will help students successfully complete the same task in the future. {task} Please be as concise, clear, and effective as possible.\n\n",
    choice="\n{choice}. {content}",
    answer="\nSummary of learning as a numbered list: ",
    prefix=" ",
)


_register_eval_template(
    name="gen_choice_cot",
    system="The following are multiple choice questions (with answers) about {subject}. Choose a correct answer that appears in the candidate answers. Your response should conclude with the format \"Therefore, the answer is \\boxed{{answer}}\".\n\n",
    choice="\n{choice}. {content}",
    answer="\nAnswer: Let's think step by step. ",
    prefix=" ",
)

_register_eval_template(
    name="gen_choice",
    system="The following are multiple choice questions (with answers) about {subject}. Choose a correct answer that appears in the candidate answers. Your response should conclude with the format '\\boxed{{answer}}'.\n\n",
    choice="\n{choice}. {content}",
    answer="\nAnswer: ",
    prefix=" ",
)

_register_eval_template(
    name="gen_cot",
    system="Below is an instruction that describes a task about {subject}, paired with an input that provides further context.\nYou need to think step by step to give an explanation for reasoning. Your final answer should be in the form \\boxed{{answer}} at the end of your response.\n\n",
    choice="\n{choice}. {content}",
    answer="\nAnswer: ",
    prefix=" ",
)

_register_eval_template(
    name="gen",
    system="Below is an instruction that describes a task about {subject}, paired with an input that provides further context.\nYour final answer should be in the form \\boxed{{answer}} at the end of your response. Please answer the question simply.\n\n",
    choice="\n{choice}. {content}",
    answer="\nAnswer: ",
    prefix=" ",
)

_register_eval_template(
    name="zh",
    system="以下是中国关于{subject}考试的单项选择题，请选出其中的正确答案。\n\n",
    choice="\n{choice}. {content}",
    answer="\n答案：",
    prefix=" ",
)
