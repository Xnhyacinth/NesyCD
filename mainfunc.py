import argparse
import os
import json

from inference import run_main


def readfiles(infile):

    if infile.endswith('json'): 
        lines = json.load(open(infile, 'r', encoding='utf8'))
    elif infile.endswith('jsonl'): 
        lines = open(infile, 'r', encoding='utf8').readlines()
        lines = [json.loads(l) for l in lines]
    else:
        raise NotImplementedError

    if len(lines[0]) == 1 and lines[0].get('prompt'): 
        lines = lines[1:] ## skip prompt line

    return lines


def step1(dataset, datatype, split, max_tokens, engine, prompt, pid, n, temp, full):
    inputfile = f'evaluation/know/{dataset}/{split}.json'
    with open(inputfile, 'r') as fin:
        inlines = json.load(fin)
    # inlines = readfiles(inputfile)

    if (temp is None) or (temp == 0):
        outputfolder = f'{engine}/{dataset}'
    else: # tempature > 0
        outputfolder = f'backgrounds-sample(n={n},temp={temp})-{engine}/{dataset}'
    os.makedirs(outputfolder, exist_ok=True)
    outputfile = f'{outputfolder}/{dataset}-{split}-p{pid}_{n}shot_full{full}.json'
    
    run_main(inlines, outputfile, engine, prompt, datatype, n, temp, full)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    
    # Required parameters
    parser.add_argument("--dataset", default=None, type=str, required=True,
        help="dataset name: [commonsenseqa]",
    )
    parser.add_argument("--task", default=None, type=str, required=True,
        help="task name: [step1, step2], should be either 1 or 2",
    )
    parser.add_argument("--split", default=None, type=str, required=True,
        help="dataset split: [train, dev, test]",
    )
    parser.add_argument("--engine", default='gpt-4', type=str, required=False,
        help="gpt4 (used in our experiments)",
    )
    parser.add_argument("--datatype", default='temp', type=str, required=False,
        help="gpt4 (used in our experiments)",
    )
    parser.add_argument("--full", default=0, type=int)
    parser.add_argument("--pid", default=1, type=int, required=False)
    parser.add_argument("--n", default=1, type=int, required=False)
    parser.add_argument("--temperature", default=0, type=float, required=False)

    args = parser.parse_args()

    # if args.dataset in ['commonsenseqa']:
    #     datatype = 'question answering'
    # elif args.dataset in ['bbh', 'agieval', 'bb', 'arc-c', 'arc-e', 'gsm8k', 'mmlu']:
    #     datatype = 'temp'
    # else: # other task type?
    #     # raise NotImplementedError
    #     datatype = 'question answering'
    datatype = args.datatype
    if args.task == 'step1':
        max_tokens = 300

    promptfile = 'regular'
    promptlines = open(f'{promptfile}.jsonl', 'r').readlines()

    for line in promptlines:
        line = json.loads(line)

        if line['type'] == datatype and line['task'] == args.task and line['pid'] == args.pid:
            prompt = line['prompt']

            if args.task == 'step1':
                outputs = step1(args.dataset, datatype, args.split, max_tokens, args.engine, 
                    prompt, args.pid, args.n, args.temperature, args.full)

            else:  ## should be either 1 or 2
                raise NotImplementedError
            
            if promptfile == 'regular':
                break ## only use the first prompt 