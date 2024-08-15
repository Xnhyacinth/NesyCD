import torch
import numpy as np
from .inference import generate_completions, get_results
import json
import os
import re
import string
from collections import Counter

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
Question: Find a movie similar to Interstellar, Monsters, Inc, Back to the Future, Inception:\nOptions:\n(A) My Fellow Americans\n(B) Shadow of the Vampire\n(C) Star Wars Episode VII - The Force Awakens\n(D) Psycho II\nA: (C).

Summary of learning as a numbered list:
1: Identify Key Characteristics: Look for central themes, genres, and elements in the movies.
2: Find Common Themes: Determine the shared themes and genres among the given movies.
3: Evaluate Options: Compare each option against these common themes and genres.
4: Select the Best Fit: Choose the movie that best matches the identified key elements

Q: Select the humorous edit that 'ruins' the input movie or musical artist name. Which of the following is a humorous edit of this artist or movie name: 'radiohead'?\nOptions:\n(A) radio head\n(B) rawdiohead\n(C) radiodead\n(D) radiohesd\nA:(C)

Summary of learning as a numbered list:
Step 1: Understand the Task: Ensure you know the goal is to find a humorous alteration.
Step 2: Identify the Original Name: Clearly recognize the name you are altering.
Step 3: Analyze Each Option: Consider how each option changes the original name and assess its humor.
Step 4: Select the Most Humorous Edit: Choose the option that changes the meaning in a clever and funny way.

Q: Given a series of navigation instructions, determine whether one would end up back at the starting point. If you follow these instructions, do you return to the starting point? Take 3 steps. Take 6 steps. Take 8 steps. Take 5 steps. Take 6 steps. Take 2 steps. Take 1 step. Take 2 steps.\nOptions:\n- Yes\n- No\nA: No
"""
      
with open('evaluation/know/bbh/fewshot.json', 'r') as json_file:
    bbh = json.load(json_file)
bbh_temp = [
    {
        'question': "Question: Find a movie similar to Interstellar, Monsters, Inc, Back to the Future, Inception:\nOptions:\n(A) My Fellow Americans\n(B) Shadow of the Vampire\n(C) Star Wars Episode VII - The Force Awakens\n(D) Psycho II\nAnswer: (C).\n\nSummary of learning as a numbered list:",
        'rational': '''
1: Identify Key Characteristics: Look for central themes, genres, and elements in the movies.
2: Find Common Themes: Determine the shared themes and genres among the given movies.
3: Evaluate Options: Compare each option against these common themes and genres.
4: Select the Best Fit: Choose the movie that best matches the identified key elements.''',
        'answer': '(C)'
    },
    {
        'question': "Question: Select the humorous edit that 'ruins' the input movie or musical artist name. Which of the following is a humorous edit of this artist or movie name: 'radiohead'?\nOptions:\n(A) radio head\n(B) rawdiohead\n(C) radiodead\n(D) radiohesd\nAnswer: (C).\n\nSummary of learning as a numbered list:",
        'rational': '''
1: Understand the Task: Ensure you know the goal is to find a humorous alteration.
2: Identify the Original Name: Clearly recognize the name you are altering.
3: Analyze Each Option: Consider how each option changes the original name and assess its humor.
4: Select the Most Humorous Edit: Choose the option that changes the meaning in a clever and funny way.''',
        'answer': '(C)'
    },
    {
        'question': "Question: Given a series of navigation instructions, determine whether one would end up back at the starting point. If you follow these instructions, do you return to the starting point? Take 3 steps. Take 6 steps. Take 8 steps. Take 5 steps. Take 6 steps. Take 2 steps. Take 1 step. Take 2 steps.\nOptions:\n- Yes\n- No\nAnswer: No.\n\nSummary of learning as a numbered list:",
        'rational': '''
1.Understand the Task: Recognize that the goal is to determine if the instructions lead back to the starting point.
2.Analyze Each Step: Carefully read through each navigation instruction.
3.Track Movement: Keep a running tally or map of the steps taken, noting direction and distance.
4.Calculate Net Movement: Determine the overall movement by summing up the steps in each direction.
5.Evaluate Result: Compare the net movement to the starting point to see if they align.
6.Determine Outcome: Decide whether the total movements result in returning to the starting point or not.''',
        'answer': 'No'
    },
]

bbh_temp_full = [
    {
        'question': "Question: Find a movie similar to Interstellar, Monsters, Inc, Back to the Future, Inception:\nOptions:\n(A) My Fellow Americans\n(B) Shadow of the Vampire\n(C) Star Wars Episode VII - The Force Awakens\n(D) Psycho II\nAnswer: (C).\n\n",
        'rational': '''
Summary of Learning:

1. Understand the Question:
   - Recognize that the task is to find a movie similar to the given set of movies based on common themes or genres.

2. Identify Key Characteristics:
   - Look for central themes, genres, and elements in the given movies: "Interstellar," "Monsters, Inc," "Back to the Future," and "Inception."

3. Find Common Themes:
   - Determine the shared themes and genres among the given movies (e.g., science fiction, fantasy, adventure, time manipulation).

4. Evaluate Options:
   - Compare each option against these common themes and genres to see which one aligns best.

5. Select the Best Fit:
   - Choose the movie that best matches the identified key elements and common themes.

Supplementary Knowledge:

1. Background Information on Given Movies:
   - Interstellar: A sci-fi film focused on space exploration, time manipulation, and the survival of humanity.
   - Monsters, Inc.: An animated fantasy movie about monsters who generate energy by scaring children, featuring alternate dimensions.
   - Back to the Future: A sci-fi film centered on time travel and the adventures that arise from altering the past and future.
   - Inception: A sci-fi movie that explores the concept of entering and manipulating dreams to influence the real world.

2. Genre Definitions:
   - Science Fiction (Sci-fi): Fiction based on imagined future scientific or technological advances and major social or environmental changes.
   - Fantasy: A genre that uses magic and other supernatural elements as a primary plot element, theme, or setting.
   - Adventure: A genre characterized by exciting and unusual experiences or journeys.

3. Analyzing Movie Elements:
   - Look at the key elements of the plot, setting, and characters in movies to determine their genre and themes.
   - Consider how movies make viewers feel (e.g., excited, thoughtful) and how this relates to the themes and genres.

4. Critical Thinking in Choices:
   - Elimination Strategy: Discard options that clearly do not fit the identified common themes and genres.
   - Comparison: Compare the remaining options based on a detailed understanding of the themes and genres discussed.

5. Example Analysis:
   - My Fellow Americans: A political comedy, not fitting the sci-fi/fantasy/adventure themes.
   - Shadow of the Vampire: A horror film, not fitting the sci-fi/fantasy/adventure themes.
   - Star Wars Episode VII - The Force Awakens: A sci-fi/fantasy adventure film, fitting the themes of space exploration and adventure.
   - Psycho II: A horror/thriller, not fitting the sci-fi/fantasy/adventure themes.

   - Conclusion: Option (C) "Star Wars Episode VII - The Force Awakens" aligns best with the common themes and genres of the given movies.''',
        'answer': '(C)'
    },
    {
        'question': "Question: Select the humorous edit that 'ruins' the input movie or musical artist name. Which of the following is a humorous edit of this artist or movie name: 'radiohead'?\nOptions:\n(A) radio head\n(B) rawdiohead\n(C) radiodead\n(D) radiohesd\nAnswer: (C).\n\n",
        'rational': '''
Summary of Learning:

1. Understand the Question:
   - Recognize that the task is to select a humorous edit that alters the original name while retaining a recognizable connection to it.

2. Identify the Original Name:
   - Clearly identify the original name provided (in this case, "radiohead").

3. Evaluate Each Option:
   - Assess each option to see how it humorously changes the original name.

4. Select the Funniest Edit:
   - Choose the option that is most humorous while still recognizable as a play on the original name.

Supplementary Knowledge:

1. Humor in Wordplay:
   - Puns: A form of wordplay that exploits multiple meanings of a term or similar-sounding words.
   - Phonetic Alterations: Changing the sounds of the original word to create humor.

2. Criteria for a Good Humorous Edit:
   - Recognizability: The altered name should still be recognizable as a variant of the original name.
   - Humor: The alteration should add a humorous twist, often by implying something funny or absurd.

3. Types of Humorous Edits:
   - Literal Translations: Making a literal translation or interpretation of part of the name (e.g., "radio head").
   - Phonetic Play: Changing the sound slightly to create a humorous effect (e.g., "radiohesd").
   - Conceptual Twist: Introducing a concept that twists the meaning humorously (e.g., "radiodead" suggests the opposite of "radiohead").

4. Critical Thinking Strategies:
   - Recognize Patterns: Notice common humorous patterns in altered names.
   - Assess Humor Impact: Evaluate which option is the funniest based on common humor principles.

5. Example Analysis:
   - Given Options: 
     - (A) radio head: A literal and slightly humorous change.
     - (B) rawdiohead: A phonetic change, less recognizable.
     - (C) radiodead: A humorous twist, suggesting the opposite of "radiohead."
     - (D) radiohesd: A minor phonetic change, less impactful.
   - Select (C) "radiodead" as it introduces a humorous and easily recognizable twist.''',
        'answer': '(C)'
    },
    {
        'question': "Question: Given a series of navigation instructions, determine whether one would end up back at the starting point. If you follow these instructions, do you return to the starting point? Take 3 steps. Take 6 steps. Take 8 steps. Take 5 steps. Take 6 steps. Take 2 steps. Take 1 step. Take 2 steps.\nOptions:\n- Yes\n- No\nAnswer: No.\n\n",
        'rational': '''
Summary of Learning:

1. Understand the Question:
   - Clearly understand what the question is asking. In this case, determine if the navigation instructions lead back to the starting point.

2. Analyze Each Instruction:
   - Break down each step given in the instructions to understand the movement.

3. Track Position:
   - Keep track of your position as you follow each instruction, either mentally or by writing it down.

4. Determine Final Position:
   - Compare the final position with the starting point to see if they match.

Supplementary Knowledge:

1. Understanding Navigation Instructions:
   - Steps in One Direction: Ensure all steps are considered in a straight line or in a specified direction.
   - Returning to the Starting Point: Sum of all steps in opposite directions should equal zero for a return to the starting point.

2. Types of Navigation Movements:
   - Linear Movements: Movements that happen in a straight line without turning.
   - Directional Movements: Movements that involve turns or changes in direction (e.g., left, right).

3. Basic Math Concepts:
   - Addition and Subtraction: Understanding how to add and subtract steps to find the final position.
   - Net Movement: Concept of net movement to determine overall change in position.

4. Critical Thinking Strategies:
   - Step-by-Step Analysis: Carefully follow each step one by one without skipping any.
   - Summarize Movements: After tracking all steps, summarize the total movement to easily determine the end position.
   - Decision Making: Based on the analysis, conclude whether the answer is "Yes" or "No."
   
5. Example Analysis:
   - Given Instructions: Take 3 steps. Take 6 steps. Take 8 steps. Take 5 steps. Take 6 steps. Take 2 steps. Take 1 step. Take 2 steps.
   - Net Calculation: 
     - Sum of steps: \(3 + 6 + 8 + 5 + 6 + 2 + 1 + 2 = 33\)
     - Since there are no specific directions given, assume all steps are in the same direction.
     - Net movement is 33 steps in one direction, indicating that the final position is not the starting point.''',
        'answer': 'No'
    },
]

### Summary of Learning:

# 1. Understand the Task:
#    - Identify that the task is to determine if a series of navigation instructions lead back to the starting point.

# 2. Identify Key Elements:
#    - Recognize the instructions and understand the direction and magnitude of each step.

# 3. Track Movement:
#    - Accurately follow each step and keep track of the net movement in each direction.

# 4. Evaluate the Final Position:
#    - Determine if the final position is the same as the starting point.

# ### Supplementary Knowledge:

# 1. Understanding Navigation Instructions:
#    - Steps: Each instruction indicates a movement in a specific direction.
#    - Starting Point: The initial position before following any instructions.

# 2. Tracking Movement:
#    - Net Movement Calculation: Sum the steps in each direction to determine the overall displacement.
#    - Directional Balance: Ensure that movements in opposite directions cancel each other out to return to the starting point.

# 3. Example Analysis:
#    - Given Instructions: Take 3 steps. Take 6 steps. Take 8 steps. Take 5 steps. Take 6 steps. Take 2 steps. Take 1 step. Take 2 steps.
#    - Net Calculation: 
#      - Sum of steps: \(3 + 6 + 8 + 5 + 6 + 2 + 1 + 2 = 33\)
#      - Since there are no specific directions given, assume all steps are in the same direction.
#      - Net movement is 33 steps in one direction, indicating that the final position is not the starting point.

# 4. Critical Thinking in Choices:
#    - Evaluation: Analyze the net movement to determine if it leads back to the starting point.
#    - Decision Making: Based on the analysis, conclude whether the answer is "Yes" or "No."

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