from tqdm import tqdm
import json
import os
from .gen import get_retirval
# facebook/contriever-msmarco facebook/dpr-question_encoder-multiset-base


def parse_knowledge(args, hits):
    knowledge = []
    # errors = jsonlines.open(f'{args.index_path}/index_errors.jsonl', 'r')
    # errors = []
    with open(f'{args.index_path}/index_errors.json', 'r', encoding="utf-8") as f:
        errors =json.load(f)

    for hit in hits[:3]:
        knowledge.append({
            'docid': int(hit.docid),
            'score': float(hit.score),
            'knowledge': errors[int(hit.docid)]['summary']
        })
    
    return knowledge

'indexes/std_merge_index',
'facebook/contriever-msmarco'
def add_knowledge(args, test_dataset, model_name):
    task = args.task.split('/')[1]
    if args.retrieval:
        name = model_name.split('/')[-1].strip()
        task += f'/{name}'
        filename = 'index_retrieval'
    else:
        filename = 'index_knowledge'
        # if os.path.exists(f'{args.index_path}/{task}/index_retrieval.json'):
        #     print(f'Exist index_retrieval_knowledge...   Load from {args.index_path}/{task}/index_retrieval.json...')
        #     with open(f'{args.index_path}/{task}/index_retrieval.json', 'r', encoding="utf-8") as f:
        #         new_test_dataset = json.load(f)
    
    if args.retriever != 'contriever':
        filename += f'_{args.retriever}'
    num_knowledge = 3
    if num_knowledge != 1:
        filename += '_3'
        
    # threshold = 68
    if args.threshold != 68.0:  
        print(f'threshold: {args.threshold}')
        filename += f'_{args.threshold}'
    #     args.save_dir += f'_threshold{threshold}'
    
    if os.path.exists(f'{args.index_path}/{task}/{filename}.json'):
        print(f'\nExist {filename}...   Load from {args.index_path}/{task}/{filename}.json...\n')
        with open(f'{args.index_path}/{task}/{filename}.json', 'r', encoding="utf-8") as f:
            new_test_dataset = json.load(f)
    else:
        if args.retriever == 'contriever':
            from pyserini.search import FaissSearcher
            searcher = FaissSearcher(
                args.index_path,
                'facebook/contriever-msmarco'
            )
        elif args.retriever == 'bm25':
            from pyserini.search.lucene import LuceneSearcher
            searcher = LuceneSearcher(args.index_path)
        elif args.retriever == 'dpr':
            from pyserini.search import FaissSearcher
            searcher = FaissSearcher(
                args.index_path,
                'facebook/dpr-question_encoder-multiset-base'
            )
            
        print(f'\nMiss {filename}...   Add {filename}.json...\n')
        print(searcher)
        print(args.retriever)
        new_test_dataset = []
        
        if args.retrieval:
            test_dataset = get_retirval(model_name, test_dataset, args.threshold)
        for data in tqdm(test_dataset, desc="Append knowledge to test dataset..."):
            question = data['task_description'] + ' ' + data['instruction']
            question = ' '.join(question.split(' ')[:512])
            if args.retrieval:
                if data['score'] == 'error':
                    hits = searcher.search(question) # Retrieve the data with silver reasoning

                    knowledge = parse_knowledge(args, hits)
                    data["knowledge"] = knowledge
                else:
                    data['knowledge'] = [{'knowledge': 'No need summary.'}]
            else:
                hits = searcher.search(question) # Retrieve the data with silver reasoning

                knowledge = parse_knowledge(args, hits)
                data["knowledge"] = knowledge
            new_test_dataset.append(data)
        
        if not os.path.exists(f'{args.index_path}/{task}'):
            os.makedirs(f'{args.index_path}/{task}', exist_ok=True)
        
        with open(f'{args.index_path}/{task}/{filename}.json', 'w', encoding="utf-8") as f:
            json.dump(new_test_dataset, f, indent=4, ensure_ascii=False)
    
    return new_test_dataset