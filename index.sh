# BM25

python -m pyserini.index.lucene \
  --collection JsonCollection \
  --input gpt-3.5-turbo/bbh/index_errors.json \
  --index indexes/index_errors \
  --generator DefaultLuceneDocumentGenerator \
  --threads 1 \
  --storePositions --storeDocvectors --storeRaw


# Dense

CUDA_VISIBLE_DEVICES=8,9 python -m pyserini.encode \
  input   --corpus gpt-3.5-turbo/bbh/index_errors_qwen2-05b.jsonl \
          --fields text \
          --delimiter "\n\n\n" \
          --shard-id 0 \
          --shard-num 1 \
  output  --embeddings indexes/index_errors_qwen2-05b \
          --to-faiss \
  encoder --encoder facebook/contriever-msmarco \
          --fields text \
          --batch 32 \
          --fp16 
