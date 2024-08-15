# NesyCD

## 🛠 Requirements

Install LLaMA-Factory following [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory).

```bash
cd NesyCD
pip install -e ".[torch,metrics]"
```

Install pyserini following [pyserini](https://github.com/castorini/pyserini).

## 💡 Data

- Download the datasets from official websites.

- From Google drive: (we unified the formats of the above datasets). [Link]()

- Please put them into `data` folder and `evaluation/know`.

## 👨‍💻 Training

### Generate CoT

```bash
python mainfunc.py --dataset ${d} --task step1 --split test --pid 3 --engine ${model} --datatype question_answering --n 0
```

### Generate Specialized Knowledge

```bash
python mainfunc.py --dataset bbh --task step1 --split errors_llama --pid 2 --engine ${model} --n 1 --full 0 --datatype temp_error
```

### Baselines

```bash
bash config/train.sh 2 0,1 {model} {method} lora 10 2e-4 16 fewshot 2 1e-10 32 all

model=llama2-7b, llama3-8b, tinyllama, qwen2-7b, qwen2-0.5b, qwen2-1.5b
method=bbh_ft, bbh_kaft, bbh_mt-re, bbh_mt-ra, bbh_mt-cot, bbh_mt-cas, bbh_std_cot
```

### NesyCD

index the specialized knowledge:

```bash
bash index.sh
```
