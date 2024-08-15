import collections
import openai
import os
import time
import threading
import json
import _thread
import random
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import contextmanager
from collections import defaultdict
from openai import OpenAI
from metric_utils import extract_answer, extract_answers_for_model, extract_answers_for_gt
import copy

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

bbh_error = [
    {
        'question': "The student was given the following question:\nThe following are multi-step reasoning math questions about gsm8k. Utilize mathematical and logical abilities to solve multi-step basic mathematical reasoning problems. Your response should conclude with the format \"Therefore, the answer is\". Question: DeShaun always wins the award in English class for reading the most books over the summer. This year is no different. Summer break is 80 days long. When he arrives, he notifies the teacher that he read 60 books over the summer. Each book averaged 320 pages long. The person closest to him read 75% as much as he did. How many pages did the second person read on average each day of break?\n",
        'answer': "The student’s wrong response is:\n60*320 = <<60*320=19200>>19200 pages\n75% of 19200 = <<75*19200=14000>>14000 pages\nHe read 14000/80 = <<14000/80=175>>175 pages a day.\nTherefore, the answer is 175.\n",
        'rational':'''
Learning Summary:
The student made an error in calculating 75% of the total number of pages DeShaun read. The incorrect calculation led to an erroneous conclusion about the average number of pages read per day by the second person. To correct this, the student needs to follow these steps:

1. Calculate the total number of pages DeShaun read: Multiply the number of books by the average number of pages per book.
2. Calculate 75% of the total pages: Apply the percentage correctly.
3. Calculate the average number of pages per day for the second person: Divide the total pages read by the number of days.

Supplementary Knowledge:
1. Percentage Calculation: 
   - Understanding that 75% means 75 out of 100 or 0.75 as a decimal. 
   - To find 75% of a number, multiply the number by 0.75.

2. Steps to Correct the Error:
   - Step 1: Calculate the total pages DeShaun read: \( 60 \text{ books} \times 320 \text{ pages/book} = 19200 \text{ pages} \).
   - Step 2: Calculate 75% of 19200 pages: \( 19200 \times 0.75 = 14400 \text{ pages} \).
   - Step 3: Calculate the average number of pages read per day: \( 14400 \text{ pages} / 80 \text{ days} = 180 \text{ pages/day} \).

3. Logical Reasoning: 
   - Verify each step to ensure the calculations are accurate.
   - Understanding how to break down a multi-step problem into manageable parts helps avoid errors and ensures the solution process is clear and logical.

Correct Solution:
1. Calculate the total number of pages DeShaun read:
   \[
   60 \text{ books} \times 320 \text{ pages/book} = 19200 \text{ pages}
   \]
2. Calculate 75% of 19200 pages:
   \[
   19200 \times 0.75 = 14400 \text{ pages}
   \]
3. Calculate the average number of pages read per day by the second person:
   \[
   14400 \text{ pages} / 80 \text{ days} = 180 \text{ pages/day}
   \]
Therefore, the answer is 180.'''
    },
    {
        'question': 'The student was given the following question:\nThe following are multiple choice questions (with answers) about logical_deduction_three_objects. A logical deduction task which requires deducing the order of a sequence of objects. Your response should conclude with the format \"Therefore, the answer is\".\n\nQuestion: The following paragraphs each describe a set of three objects arranged in a fixed order. The statements are logically consistent within each paragraph. A fruit stand sells three fruits: peaches, pears, and mangoes. The mangoes are less expensive than the peaches. The mangoes are more expensive than the pears.\nOptions:\n(A) The peaches are the second-most expensive\n(B) The pears are the second-most expensive\n(C) The mangoes are the second-most expensive\n',
        'answer': 'The student’s wrong response is:\n1. The mangoes are less expensive than the peaches: \"mangoes < peaches\".\n2. The mangoes are more expensive than the pears: \"mangoes > pears\".\nFrom these two statements, we can conclude that \"pears < mangoes < peaches\".\nAccording to this ordering, the second-most expensive fruit is the pears.\nThe pears are the second-most expensive. Therefore, the answer is (B).\n',
        'rational': '''
Learning Summary:
The student made an error in determining the second-most expensive fruit by misinterpreting the logical relationships given in the problem. The correct approach involves properly understanding the comparative prices of the fruits based on the provided statements. The key is to correctly order the fruits by price and identify the second-most expensive one.

Supplementary Knowledge:
1. Logical Deduction:
   - When faced with comparative statements, carefully align each object in a sequence that satisfies all given conditions.
   - Ensure that each step logically follows from the previous one and that all given relationships are accounted for.

2. Steps to Correct the Error:
   - Step 1: Identify and write down the relationships from the statements:
     - Mangoes are less expensive than peaches: \( \text{mangoes} < \text{peaches} \).
     - Mangoes are more expensive than pears: \( \text{mangoes} > \text{pears} \).
   - Step 2: Combine these relationships to form a complete order:
     - From the given relationships: \( \text{pears} < \text{mangoes} < \text{peaches} \).
   - Step 3: Determine the second-most expensive fruit from the ordered sequence:
     - The sequence is \( \text{pears} < \text{mangoes} < \text{peaches} \).
     - The second-most expensive fruit is the mangoes.

3. Critical Thinking:
   - It is essential to double-check the logical consistency of the ordering.
   - Visualizing or writing down the sequence can help avoid misinterpretation.

Correct Solution:
1. From the statements, we know:
   - The mangoes are less expensive than the peaches: \( \text{mangoes} < \text{peaches} \).
   - The mangoes are more expensive than the pears: \( \text{mangoes} > \text{pears} \).

2. Combining these two pieces of information:
   \[
   \text{pears} < \text{mangoes} < \text{peaches}
   \]

3. From the ordered sequence, the second-most expensive fruit is:
   \[
   \text{mangoes}
   \]

Therefore, the answer is (C).'''
    }
]

bbh_err = [
    {
        'question': "The student was given the following question:\nThe following are multi-step reasoning math questions about gsm8k. Utilize mathematical and logical abilities to solve multi-step basic mathematical reasoning problems. Your response should conclude with the format \"Therefore, the answer is\". Question: DeShaun always wins the award in English class for reading the most books over the summer. This year is no different. Summer break is 80 days long. When he arrives, he notifies the teacher that he read 60 books over the summer. Each book averaged 320 pages long. The person closest to him read 75% as much as he did. How many pages did the second person read on average each day of break?\n",
        'answer': "The student’s wrong response is:\n60*320 = <<60*320=19200>>19200 pages\n75% of 19200 = <<75*19200=14000>>14000 pages\nHe read 14000/80 = <<14000/80=175>>175 pages a day.\nTherefore, the answer is 175.\n",
        'rational':'Learning Summary:\n1. Check Calculations Thoroughly: Always double-check your arithmetic operations to ensure accuracy. A single error in multiplication or division can lead to an incorrect answer.\n2. Understand Percentage Calculations: When dealing with percentages, ensure you understand how to correctly convert percentages to decimals and perform the necessary operations.\n3. Units Consistency: Keep track of units throughout your calculations. Ensure you are consistent in using and converting units where necessary.\n4. Logical Flow: Break down the problem into clear, logical steps. Ensure each step follows from the previous one and leads logically to the next.\n5. Final Answer Format: Conclude your answer with a clear statement in the format “Therefore, the answer is [answer]” to indicate completion.\n\nSupplementary Knowledge:\n1. Basic Arithmetic Operations:\n   - Multiplication: Understand how to multiply numbers correctly, including large numbers.\n   - Division: Be able to divide numbers accurately, ensuring the division makes sense within the context.\n2. Percentage Calculations:\n   - Understanding Percentages: Know how to convert percentages to decimals (e.g., 75% = 0.75) and apply them in calculations.\n   - Finding Percentages of a Number: Be able to calculate a percentage of a given number (e.g., 75% of 19200).\n3. Averages:\n   - Calculating Averages: Understand how to find the average of a set of numbers, including dividing the total by the number of units (e.g., total pages read divided by the number of days).\n4. Units of Measurement:\n   - Consistency in Units: Keep track of units (e.g., pages, days) throughout your calculations to ensure consistency and correctness.'
    },
    {
        'question': 'The student was given the following question:\nThe following are multiple choice questions (with answers) about logical_deduction_three_objects. A logical deduction task which requires deducing the order of a sequence of objects. Your response should conclude with the format \"Therefore, the answer is\".\n\nQuestion: The following paragraphs each describe a set of three objects arranged in a fixed order. The statements are logically consistent within each paragraph. A fruit stand sells three fruits: peaches, pears, and mangoes. The mangoes are less expensive than the peaches. The mangoes are more expensive than the pears.\nOptions:\n(A) The peaches are the second-most expensive\n(B) The pears are the second-most expensive\n(C) The mangoes are the second-most expensive\n',
        'answer': 'The student’s wrong response is:\n1. The mangoes are less expensive than the peaches: \"mangoes < peaches\".\n2. The mangoes are more expensive than the pears: \"mangoes > pears\".\nFrom these two statements, we can conclude that \"pears < mangoes < peaches\".\nAccording to this ordering, the second-most expensive fruit is the pears.\nThe pears are the second-most expensive. Therefore, the answer is (B).\n',
        'rational': 'Learning Summary:\n1. Understand SVG Path Commands: Familiarize yourself with SVG path commands such as M (move to) and L (line to). Know what each command does and how they combine to form shapes.\n2. Identify Shape Properties: Recognize the properties of basic geometric shapes, such as the number of sides, equal-length sides, and internal angles.\n3. Process of Elimination: Use the process of elimination effectively by comparing the given options and ruling out those that do not fit the criteria derived from the problem.\n4. Double-Check Logic: Ensure the logical steps taken in identifying the shape are sound and based on the properties and definitions of the shapes.\n5. Conclude Clearly: Always conclude with a clear statement in the format “Therefore, the answer is [answer]” to indicate the final decision.\n\nSupplementary Knowledge:\n1. SVG Path Commands:\n   - Move To (M): The M command moves the current point to the specified coordinates (x, y).\n   - Line To (L): The L command draws a straight line from the current point to the specified coordinates (x, y).\n2. Basic Geometric Shapes:\n   - Triangle: A shape with three sides and three angles.\n   - Rectangle: A four-sided shape with opposite sides equal and all angles 90 degrees.\n   - Kite: A four-sided shape with two pairs of adjacent sides that are equal.\n   - Pentagon: A shape with five sides and five angles.\n   - Hexagon: A shape with six sides and six angles.\n   - Heptagon: A shape with seven sides and seven angles.\n   - Octagon: A shape with eight sides and eight angles.\n   - Circle: A round shape with all points equidistant from the center.\n   - Line: A straight path that extends in both directions with no endpoints.\n   - Sector: A portion of a circle, defined by two radii and the connecting arc.\n3. Properties of Shapes:\n   - Number of Sides: Knowing how many sides a shape has can help identify it.\n   - Side Lengths: Understanding if sides are equal or different in length is crucial for classification.\n   - Angles: Knowing the internal angles can help differentiate between shapes, especially quadrilaterals.\n\nBy mastering these concepts and methods, students can improve their ability to solve similar multiple-choice questions about geometric shapes effectively.'
    }
]

bbh_err_new = [
    {
        'question': "The student was given the following question:\nThe following are multiple choice questions (with answers) about elementary_mathematics.  Your response should conclude with the format \"Therefore, the answer is\".\n\nQuestion: Carmen saved 592 pennies. Her sister saved 128 pennies. Together, they put 250 pennies in wrappers and took them to the bank. Whatis the total number of pennies, rounded to the nearest hundred, Carmen and her sister have left?\nA. 300 \nB. 500 \nC. 700 \nD. 1,000\n",
        'answer': "The student’s wrong response is:\nFirst, let's calculate the total number of pennies Carmen and her sister have left after putting 250 pennies in wrappers and taking them to the bank. \nCarmen saved 592 pennies, her sister saved 128 pennies, and together, they put 250 pennies in wrappers, giving a total of 592 + 128 + 250 = 970 pennies.\n\nNext, we need to round this total number of pennies to the nearest hundred. Since 970 is less than 1,000, we round down to 300 as the total number of pennies Carmen and her sister have left.\n\nTherefore, the answer is A. 300.\n",
        'rational':'Learning Summary:\n1. Careful Reading and Question Comprehension: Always read the question thoroughly to understand what is being asked, including any actions already taken (e.g., pennies put in wrappers).\n2. Correct Mathematical Operations: Ensure that addition or subtraction operations reflect the sequence of events described in the question.\n3. Rounding Rules: Round numbers according to the provided guidelines. For rounding to the nearest hundred, if the tens digit is 5 or more, round up; if less than 5, round down.\n4. Verification of Calculation: Double-check arithmetic operations to ensure accuracy, especially with multiple steps.\n\nSupplementary Knowledge:\n1. Basic Arithmetic Operations: Addition and subtraction are fundamental for solving such problems. Adding amounts to find a total and subtracting what has been used or taken away is crucial.\n2. Rounding Numbers: Understand how to round numbers to the nearest ten, hundred, thousand, etc. If the digit immediately following the rounding position is 5 or greater, round up; if less than 5, round down.\n3. Handling Currency: Recognize currency denominations such as pennies and understand their accumulation for managing money-related problems.\n4. Problem-Solving Strategy: Use a step-by-step approach to problem-solving, including understanding the question, performing calculations, checking the work, and presenting the answer clearly.'
    },
    {
        'question': 'The student was given the following question:\nThe following are multi-step reasoning math questions about gsm8k. Utilize mathematical and logical abilities to solve multi-step basic mathematical reasoning problems. Your response should conclude with the format \"Therefore, the answer is\".\n\nQuestion: Mary and Jimmy start running from the same spot, but in opposite directions. Mary runs at 5 miles per hour and Jimmy runs at 4 miles per hour. What is the distance between them after 1 hour?\n',
        'answer': 'The student’s wrong response is:\n1 hour is 60 minutes.\nMary moves 5 miles/hour*60 minutes=<<5*60=300>>300 miles in 1 hour.\nJimmy moves 4 miles/hour*60 minutes=<<4*60=240>>240 miles in 1 hour.\nThe distance between them is 300 miles-240 miles=<<300-240=60>>60 miles.\nTherefore, the answer is 60.\n',
        'rational': "Learning Summary:\n1. Speed and Direction Consideration: When dealing with two objects moving in opposite directions, correctly incorporate their speeds and directions to determine their relative distance.\n2. Unit Conversion Accuracy: Be precise in converting units (e.g., hours to minutes) to match the given parameters and ensure consistency throughout the calculation.\n3. Determine the Direction: Based on the question, decide whether to subtract distances for the same direction or add them for opposite directions.\n4. Check Final Result: Always review your final calculation to ensure it aligns with the question's requirements and makes logical sense.\n5. Accuracy in Arithmetic Operations: Maintain accuracy in mathematical operations, especially when subtracting or comparing values.\n\nSupplementary Knowledge: \n1. Speed and Distance Relationships: Know the formula for calculating distance (distance = speed × time). For objects starting from the same point and moving in the same direction, the difference in their distances is the difference in their speeds multiplied by the time. For objects moving in opposite directions, the total distance is the sum of their speeds multiplied by the time.\n2. Time Conversions: 1 hour equals 60 minutes, and 1 minute equals 60 seconds. Convert time units as needed based on the problem.\n3. Problem Solving Strategy: Use a systematic approach to problem-solving by breaking down tasks into manageable steps for efficient and accurate solutions."
    }
]

class TimeoutException(Exception):
    def __init__(self, msg=''):
        self.msg = msg


@contextmanager
def time_limit(seconds, msg=''):

    timer = threading.Timer(seconds, lambda: _thread.interrupt_main())
    timer.start()
    try:
        yield
    except KeyboardInterrupt:
        raise TimeoutException("Timed out for operation {}".format(msg))
    finally:
        # if the action ends in specified time, timer is canceled
        timer.cancel()


def add_prompt(item, prompt):

    def rmreturn(s):
        s = s.replace('\n\n', ' ')
        s = s.replace('\n', ' ')
        return s.strip()
    choice = "\n{choice}. {content}"
    query = item['question']
    candidates = ' '.join([choice.format(choice=chr(ord("A") + id), content=ch) for id, ch in enumerate(item["choices"])])
    prompt = prompt.replace('{query}', query + candidates)

    if item.get('output'):  # background info
        backinfo = rmreturn(item['output'][0])
        prompt = prompt.replace('{background}', backinfo)

    return prompt


def clustering_prompt(items, prompt):

    def rmreturn(s):
        s = s.replace('\n\n', ' ')
        s = s.replace('\n', ' ')
        return s.strip()

    cluster_prompts = []
    for item in items:
        query = item['question']
        backinfo = rmreturn(item['output'][0])
        item_prompt = prompt.replace('{query}', query)
        item_prompt += f' {backinfo}'
        cluster_prompts.append(item_prompt)

    cluster_prompts.append(prompt)
    return ' \n\n\n\n '.join(cluster_prompts)


def run_embeddings(input_text, engine='text-similarity-davinci-001'):

    texts = [t.replace('\n', '') for t in input_text]
    outputs = openai.Embedding.create(input=texts, model=engine)['data']
    embeddings = [o['embedding'] for o in outputs]

    return embeddings


def run_inference(inputs_with_prompts, engine, max_tokens, num_sequence=1, temp=0):

    completions = {"choices": []}
    # for _ in range(200):
    #     try:
    #         with time_limit(20, 'run gpt-4'):
    #             completions = openai.chat.completions.create(
    #                 model="gpt-4",  # use gpt-4
    #                 messages=[
    #                     {"role": "system", "content": "You are a helpful assistant."},
    #                     {"role": "user", "content": inputs_with_prompts}
    #                 ]
    #                 )
    #             break
    #     except:
    #         time.sleep(2)
    for _ in range(5):
        try:
            completions = openai.chat.completions.create(
                model="gpt-4",  # use gpt4
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": inputs_with_prompts}
                ]
            )
            break
        except Exception as e:
            print(e)
            print(inputs_with_prompts)
            time.sleep(2)
    if len(completions['choices']) == 0:
        return None
    outputs = completions.choices[0].message.content
    return outputs


def predict(params):
    prompt, query, engine, datatype, n, full = params
    label = 1
    if datatype == 'temp':
        question = query['instruction']
        # query['task_name'] = 'multi-step_math'
        # query['task_description'] = query['task_discription']
        # del query['task_discription']
        fewshot = ''
        if full == 1:
            bbh = bbh_temp_full
        else:
            bbh = bbh_temp
        for ex in bbh[:n]:
            fewshot += ex['question'] + ex['rational'] + '\n\n'
        input_with_prompt = prompt.format(prefix=query['prefix_prompt'], task=query['task_description'], query=question, fewshot=fewshot, answer=query['output'])
        # query['input_with_prompt'] = input_with_prompt
    elif datatype == 'temp_error':
        question = query['question']
        # query['task_name'] = 'multi-step_math'
        # query['task_description'] = query['task_discription']
        # del query['task_discription']
        
        bbh = bbh_err_new
        if n == 0:
            fewshot = '\n\n'
        else:
            fewshot=''
        for i, ex in enumerate(bbh[:n]):
            fewshot += f'\n\nExample {i}' + ex['question'] + '\n' + ex['answer'] + '\n' + ex['rational']
        if n != 0:
            fewshot += '\n\n'
        question = question.split('\nAnswer: ')[0]
        response = query['stu_response']
        if '? ? ? ? ? ? ? ? ? ? ?' in response:
            response = response.split('? ? ? ? ? ? ? ? ? ? ?')[0]
        if '| | | | | | | | | | | | |' in response:
            response = response.split('| | | | | | | | | | | | |')[0]
        if 'per day per day per day per day' in response:
            response = response.split('per day per day per day per day')[0]
        if '\t\t\t\t\t\t\t\t\t\t\t\t\t\t' in response:
            response = response.split('\t\t\t\t\t\t\t\t\t\t\t\t\t\t')[0]
        if '+ True + True + True + True' in response:
            response = response.split('+ True + True + True + True')[0]
        if '888888888' in response:
            response = response.split('888888888')[0]
        if '0000000000000000000' in response:
            response = response.split('0000000000000000000')[0]
        if '0, 0, 0, 0, 0, 0, 0, 0, 0, 0' in response:
            response = response.split('0, 0, 0, 0, 0, 0, 0, 0, 0, 0')[0]
        if '\ufffd\ufffd\ufffd\ufffd\ufffd\ufffd\ufffd' in response:
            response = response.split('\ufffd\ufffd\ufffd\ufffd\ufffd\ufffd\ufffd')[0]
        if 'not not not not not not not' in response:
            response = response.split('not not not not not not not')[0]
        response = ' '.join(response.split(' ')[:256])
        input_with_prompt = prompt.format(question=question, fewshot=fewshot, response=response)
    else:
        # input_with_prompt = add_prompt(query, prompt)
        question = query['instruction']
        # choice = "\n{choice}. {content}"
        # candidates = ' '.join([choice.format(choice=chr(ord("A") + id), content=query[chr(ord("A") + id)]) for id in range(4)])
        # input_with_prompt = prompt.format(subject=query['subject'], query=question)
        input_with_prompt = prompt.format(query=question)
        label = 1
        # query['instruction'] = question + candidates
        # query['task_name'] = query['subject']
        # query['task_description'] = ''
        # query['output'] = query['answer']
        # query['input'] = ''
    
    retry_count = 15
    retry_interval = 5
    if 'yi' in engine:
        API_KEY = ""
        API_BASE = "https://api.lingyiwanwu.com/v1"
    elif 'gpt' in engine:
        API_KEY = ""
        # openai.base_url = url
        API_BASE = ''
    elif 'glm' in engine:
        API_KEY = ""
        # openai.base_url = url
        API_BASE = 'https://open.bigmodel.cn/api/paas/v4/'
    else:
        API_KEY = ""
        API_BASE = f'https://cloud.infini-ai.com/maas/{engine}/nvidia/'
    client = OpenAI(
        api_key=API_KEY,
        base_url=API_BASE
    )
    if label == 0:
        if 'Therefore, the answer is' not in query['response']:
            label = 1
        else:
            response = query['response'].split('Therefore, the answer is')[1].split('.')[0].strip().rstrip('.')
            if response[0] != query['output']:
                label = 1
    if label:
        for _ in range(retry_count):
            try:
                response = client.chat.completions.create(
                    model=engine,  # use gpt4
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": input_with_prompt}
                    ],
                    temperature=0.8,
                    # stream=True
                )
                msg = response.choices[0].message.content.strip()
                query['summary'] = msg
                return question, query

            except TimeoutError:
                print("Timeout", question)
                retry_count += 1
                # retry_interval *= 2
                time.sleep(retry_interval)

            except Exception as e:
                print(e)
                # print(question)
                retry_count += 1
                # retry_interval *= 2
                time.sleep(retry_interval)
        query['summary'] = 'api defeat'
    else:
        query['summary'] = query['response']
    return question, query


def run_main(inlines, outfile, engine, prompt, datatype, n=0, temp=0, full=0):

    if os.path.exists(outfile):
        outs = open(outfile, 'a', encoding='utf8')
        num_lines = len(open(outfile, 'r').readlines())
        inlines = inlines[num_lines:]
    else:  # not os.path.exists(outfile)
        outs = open(outfile, 'a', encoding='utf8')
        # outs.write(json.dumps({"prompt": prompt}) + '\n')
    # inlines = inlines[:4]
    # inlines = random.sample(inlines, min(len(inlines), 4000))
    pbar = tqdm(total=len(inlines))
    index = 0
    pbar.update(index)
    # predict((prompt, inlines[0], engine, datatype, n, full))
    with ProcessPoolExecutor(max_workers=20) as executor:
        futures = [executor.submit(predict, (prompt, query, engine, datatype, n, full)) for query in inlines]
        query2res = collections.defaultdict(int)
        results = []

        for job in as_completed(futures):
            query, res = job.result(timeout=None)
            # res['response'] = res['summary']
            # del res['summary']
            results.append(res)
                
            # query2res[query] = res
            pbar.update(1)
    # results = []
    # for k, v in query2res.items():
    #     results.append(v)
    # with open(os.path.join(self.eval_args.save_dir, "results.json"), "w", encoding="utf-8", newline="\n") as f:
    
    json.dump(results, outs, indent=4)
    
    pbar.close()
    outs.close()
    
    if datatype == 'question_answering':
        with open(outfile, 'r') as f:
            results = json.load(f) 
        

        correct_count = 0
        for res in results:
            answer = res['summary']
            if 'gsm8k' in res['task_name'] or 'GSM' in res['task_name'] or 'MATH' in res['task_name']:
                pass
            else:
                if 'the answer is:\n\n' in answer:
                    answer = answer.split('the answer is:\n\n')[-1].split('.')[0].split('\n\n')[0].strip()
                elif 'the answer is:' in answer:
                    answer = answer.split('the answer is:')[-1].split('.')[0].split('\n\n')[0].strip()
                elif 'the answer is' in answer:
                    answer = answer.split('the answer is')[-1].split('.')[0].split('\n\n')[0].strip()
                elif 'answer is' in answer:
                    answer = answer.split('answer is')[-1].split('.')[0].split('\n\n')[0].strip()
                elif 'is that' in answer:
                    answer = answer.split('is that')[-1].split('.')[0].split('\n\n')[0].strip()
                elif 'is:\n' in answer:
                    answer = answer.split('is:\n')[-1].split('.')[0].split('\n\n')[0].strip()
                elif 'is:' in answer:
                    answer = answer.split('is:')[-1].split('.')[0].split('\n\n')[0].strip()
                elif ' be ' in answer:
                    answer = answer.split(' be ')[-1].split('.')[0].split('\n\n')[0].strip()
                elif ' is ' in answer:
                    answer = answer.split(' is ')[-1].split('.')[0].split('\n\n')[0].strip()
                elif 'option' in answer:
                    answer = answer.split('option')[-1].split('.')[0].split(' ')[0].split(':')[0].split('\n\n')[0].strip()
                elif 'is' in answer:
                    answer = answer.split('is')[-1].split('.')[0].split('\n\n')[0].strip()
                else:
                    answer = answer.split('answer is')[-1].split('.')[0].split('\n\n')[0].strip()
                
            if res['task_name'] != 'formal_fallacies':
                if 'gsm8k' in res['task_name'] or 'GSM' in res['task_name'] or 'MATH' in res['task_name']:
                    gt_choice, gt_content = extract_answer(copy.deepcopy(res['output']))
                    md_choice, md_content = extract_answer(copy.deepcopy(answer))
                else:
                    gt_choice, gt_content = extract_answers_for_gt(copy.deepcopy(res['output']))
                    md_choice, md_content = extract_answers_for_model(copy.deepcopy(answer))
                if md_choice and md_choice != 'none' and gt_choice and gt_choice != 'none':
                    if md_choice in gt_choice:
                        # check choice
                        right_scores = 1
                        correct_count += 1
                    else:
                        right_scores = 0
                else:
                    # check content
                    if md_content == 'none' or md_content == '':
                        right_scores = 0  
                    elif gt_content == md_content:
                        right_scores = 1
                        correct_count += 1
                    else:
                        right_scores = 0

            else:
                gt_choice = 'none'
                gt_content = res['output'].lower().strip()
                md_choice = 'none'
                md_content = answer.lower().strip()
                
                if decide(ground_truth=gt_content, model_answer=md_content):
                    right_scores = 1
                    correct_count += 1
                else:
                    right_scores = 0
            
            res['formatted_gt_answer'] = f'Choice:{gt_choice}, Content:{gt_content}'
            res['formatted_extracted_model_answers'] = f'Choice:{md_choice}, Content:{md_content}'
            res['stu_right_score'] = right_scores
            
        total_num = len(results)
        metricss = {
                'description': 'macro test results',
                "right_num": correct_count,
                "total_num": total_num,
                "total_accuracy": correct_count / total_num,
            }
        xxxx = outfile.split('.json')[0]
        with open(f'{xxxx}_res.json', 'w') as f:
            json.dump([metricss, results], f, indent=4)

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
    
        # question, query = k, v
        # outs.write(json.dumps(query) + '\n')

    # import pandas as pd
    # df = pd.DataFrame({'query': list(query2res.keys()), 'infer_result': list(query2res.values())})
    # df.to_excel('./chatgpt_infer_result.xlsx', index=False)

    # outs.write(json.dumps({
    #         'question': inputs[0],
    #         'answer': answers[0],
    #         'label': label,
    #         'output': outputs})
    #         + '\n')

    # while index < len(inlines):
    #     inputs, answers = [], []
    #     inputs_with_prompts = []
    #     # for _ in range(1):
    #     #     if index >= len(inlines): break
    #     #     input_with_prompt = add_prompt(inlines[index], prompt)
    #     #     inputs.append(inlines[index]['question']) ## a string
    #     #     answers.append(inlines[index]['answer']) ## a list of strings
    #     #     inputs_with_prompts.append(input_with_prompt)
    #     #     index += 1
    #     input_with_prompt = add_prompt(inlines[index], prompt)
    #     inputs.append(input_with_prompt)  # a string
    #     answers.append(inlines[index]['answer'])  # a list of strings
    #     label = chr(ord("A") + inlines[index]
    #                 ['choices'].index(inlines[index]['answer']))

    #     # samples = defaultdict(list)

    #     outputs = run_inference(input_with_prompt,
    #                             engine, max_tokens, n, temp)
    #     if not outputs:
    #         continue
    #     # for j, output in enumerate(outputs):
    #     #     samples[j//n].append(output)

    #     outs.write(json.dumps({
    #         'question': inputs[0],
    #         'answer': answers[0],
    #         'label': label,
    #         'output': outputs})
    #         + '\n')
    #     index += 1
    #     pbar.update(1)

    
