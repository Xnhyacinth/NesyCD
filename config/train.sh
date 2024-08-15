
# export WANDB_API_KEY=WANDB_API_KEY
num_gpus=${1:-"1"}
echo "GPU counts: ${num_gpus}"
gpus=${2:-"8"}
echo "GPU: ${gpus}"
model=${3:-"llama3-8b"}
dataset=${4:-"identity"}
finetuning_type=${5:-"lora"}
epoch=${6:-"3"}
lr=${7:-"1e-4"}
bs=${8:-"8"}
template=${9:-"fewshot"}
deepspeed=${10:-"-1"}
eval=${11:-"1e-10"}
r=${12:-"8"}
mod=${13:-"all"}
max_samples=${14:-"1000000"}
lang=${15:-"gen_bbh_cot_prefix"}
eval_data=${16:-"bbh"}
split_task=${17:-"test"}
index_path=${18:-"0"}
retriever=${19:-"0"}
num_knowledge=${20:-"1"}
retrieval=${21:-"0"}
threshold=${22:-"68"}
extra_args=""
save_steps=100
cutoff_len=2048
train_mod=no
eval_mod=no
merge_mod=no
gradient_accumulation_steps=1

if [ "$bs" = "4" ];then
    gradient_accumulation_steps=2
fi


if [ "$mod" = "all" ];then
    train_mod=train
    eval_mod=eval
fi
if [ "$mod" = "train" ];then
    train_mod=train
fi
if [ "$mod" = "eval" ];then
    eval_mod=eval
fi
if [ "$mod" = "merge" ];then
    merge_mod=merge
fi
if [[ $mod == *merge* ]];then
    merge_mod=merge
fi
if [[ $mod == *eval* ]];then
    eval_mod=eval
fi

echo ""
echo "train_mod: ${train_mod}"
echo "eval_mod: ${eval_mod}"
echo "merge_mod: ${merge_mod}"

model_name_or_path=${model}
model="${model_name_or_path##*/}"
if [ "$model" = "llama3-8b" ];then
    model_name_or_path=meta-llama/Meta-Llama-3-8B
    cutoff_len=4096
fi
if [ "$model" = "llama3-8b-inst" ];then
    model_name_or_path=meta-llama/Meta-Llama-3-8B-Instruct
    template=llama3
fi
if [ "$model" = "llama2-7b" ];then
    model_name_or_path=meta-llama/Llama-2-7b-hf
    cutoff_len=4096
    # extra_args="$extra_args --fp16"
fi
if [ "$model" = "qwen2-0.5b" ];then
    model_name_or_path=Qwen/Qwen2-0.5B
    cutoff_len=4096
    # extra_args="$extra_args --fp16"
fi
if [ "$model" = "qwen2-1.5b" ];then
    model_name_or_path=Qwen/Qwen2-1.5B
    cutoff_len=4096
    # extra_args="$extra_args --fp16"
fi
if [ "$model" = "qwen2-7b" ];then
    model_name_or_path=Qwen/Qwen2-7B
    cutoff_len=4096
    # extra_args="$extra_args --fp16"
fi
if [ "$model" = "llama2-7b-chat" ];then
    model_name_or_path=meta-llama/Llama-2-7b-chat-hf
    template=llama2
fi
if [ "$model" = "tinyllama" ];then
    model_name_or_path=TinyLlama/TinyLlama-1.1B-intermediate-step-1431k-3T
fi
if [ "$model" = "tinyllama-chat" ];then
    model_name_or_path=TinyLlama/TinyLlama-1.1B-Chat-v1.0
fi
if [ "$model" = "llama3-70b-inst" ];then
    model_name_or_path=meta-llama/Meta-Llama-3-70B-Instruct
    template=llama3
fi

save_path=saves/${model}/${finetuning_type}/sft/${dataset}
run_name=LLM/${model}/${finetuning_type}/sft/${dataset}
merge_path=models/${model}_${finetuning_type}_sft_${dataset}
eval_path=saves/${model}/${finetuning_type}/eval/${dataset}

save_path=${save_path//\,/_}
run_name=${run_name//\,/_}
merge_path=${merge_path//\,/_}
eval_path=${eval_path//\,/_}

if [ "$finetuning_type" = "lora" ];then
    lora_rank=${r}
    lora_dropout=0.0
    lora_target=all
    if [ "$train_mod" = "train" ];then
        merge_mod=merge
    fi
    
    # lora_alpha=16
    # "Name(s) of target modules to apply LoRA. "
    # "Use commas to separate multiple modules. "
    # "Use `all` to specify all the linear modules. "
    # "LLaMA choices: [`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`], "
    # "BLOOM & Falcon & ChatGLM choices: [`query_key_value`, `dense`, `dense_h_to_4h`, `dense_4h_to_h`], "
    # "Baichuan choices: [`W_pack`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`], "
    # "Qwen choices: [`c_attn`, `attn.c_proj`, `w1`, `w2`, `mlp.c_proj`], "
    # "InternLM2 choices: [`wqkv`, `wo`, `w1`, `w2`, `w3`], "
    # "Others choices: the same as LLaMA."
    extra_args="$extra_args --lora_rank ${lora_rank} --lora_dropout ${lora_dropout} --lora_target ${lora_target}"
    if [ "$lora_rank" != "8" ];then
        save_path="${save_path}_r${lora_rank}"
        run_name="${run_name}_r${lora_rank}"
        merge_path="${merge_path}_r${lora_rank}"
        eval_path="${eval_path}_r${lora_rank}"
    fi
fi

if [ "$deepspeed" != "-1" ];then
    extra_args="$extra_args --deepspeed examples/deepspeed/ds_z${deepspeed}_config.json"
    save_path="${save_path}_ds${deepspeed}"
    run_name="${run_name}_ds${deepspeed}"
    merge_path="${merge_path}_ds${deepspeed}"
    eval_path="${eval_path}_ds${deepspeed}"
fi

if [ "$eval" != "1e-10" ];then
    let eval_bs=bs*4
    extra_args="$extra_args --per_device_eval_batch_size ${eval_bs} --val_size ${eval} --eval_steps ${save_steps} --evaluation_strategy steps --load_best_model_at_end"
    save_path="${save_path}_eval"
    run_name="${run_name}_eval"
    merge_path="${merge_path}_eval"
    eval_path="${eval_path}_eval"
fi

if [ "$lr" != "1e-4" ];then
    save_path="${save_path}_lr${lr}"
    run_name="${run_name}_lr${lr}"
    merge_path="${merge_path}_lr${lr}"
    eval_path="${eval_path}_lr${lr}"
fi

if [ "$epoch" != "10" ];then
    save_path="${save_path}_epoch${epoch}"
    run_name="${run_name}_epoch${epoch}"
    merge_path="${merge_path}_epoch${epoch}"
    eval_path="${eval_path}_epoch${epoch}"
fi

if [ "$max_samples" != "1000000" ];then
    save_path="${save_path}_${max_samples}"
    run_name="${run_name}_${max_samples}"
    merge_path="${merge_path}_${max_samples}"
    eval_path="${eval_path}_${max_samples}"
    if [ "$max_samples" == "100" ];then
        let epoch=epoch*2
        echo "epoch: ${epoch}"
    fi
fi

# Train
if [ "$train_mod" = "train" ];then
    echo ""
    for (( i=1; i<=50; i++ ))
    do
        echo -n "*"
    done
    echo ""

    echo "model_name_or_path: ${model_name_or_path}"
    echo "template: ${template}"
    echo "save_path: ${save_path}"
    echo "bs: ${bs}"
    echo "gradient_accumulation_steps: ${gradient_accumulation_steps}"

    CUDA_VISIBLE_DEVICES=${gpus} llamafactory-cli train \
        --stage mysft \
        --do_train \
        --model_name_or_path ${model_name_or_path} \
        --dataset ${dataset} \
        --dataset_dir ./data \
        --template ${template} \
        --finetuning_type ${finetuning_type} \
        --output_dir ${save_path} \
        --overwrite_cache \
        --overwrite_output_dir \
        --cutoff_len ${cutoff_len} \
        --preprocessing_num_workers 16 \
        --per_device_train_batch_size ${bs} \
        --gradient_accumulation_steps ${gradient_accumulation_steps} \
        --lr_scheduler_type cosine \
        --logging_steps 10 \
        --warmup_ratio 0.1 \
        --save_steps ${save_steps} \
        --save_total_limit 1 \
        --learning_rate ${lr} \
        --num_train_epochs ${epoch} \
        --max_samples ${max_samples} \
        --ddp_timeout 180000000 \
        --plot_loss \
        --report_to wandb \
        --remove_unused_columns False \
        --run_name ${run_name} \
        ${extra_args} #--fp16 \
fi

## Merge
if [ "$merge_mod" = "merge" ];then
    echo ""
    for (( i=1; i<=50; i++ ))
    do
        echo -n "*"
    done
    echo ""
    echo "merge_path: ${merge_path}"
    CUDA_VISIBLE_DEVICES=0 llamafactory-cli export \
        --model_name_or_path ${model_name_or_path} \
        --adapter_name_or_path ${save_path}  \
        --template ${template} \
        --finetuning_type ${finetuning_type} \
        --export_dir ${merge_path} \
        --export_size 2 \
        --export_device cpu \
        --export_legacy_format False
fi

## Eval
if [ "$eval_mod" = "eval" ];then
    extra_args=""
    echo ""
    for (( i=1; i<=50; i++ ))
    do
        echo -n "*"
    done
    echo ""
    if [ "$train_mod" = "train" ];then
        split_task=test
    fi
    # lang=gen_bbh_cot_sum_prefix
    if [[ $lang == *sum* ]]
    then
        if [[ $dataset == *full* ]]
        then
            lang="${lang}_full"
        fi
    fi
    
    if [[ $dataset == *glm4* ]]
    then
        split_task="${split_task}_glm4"
    fi
    
    if [[ $index_path != 0 ]];then
        # eval_path="${eval_path}_${retriever}"
        # if [[ $retriever == "contriever" ]];then
        #     retriever=facebook/contriever-msmarco
        # fi
        extra_args="${extra_args} ${index_path} ${retriever} ${num_knowledge} ${retrieval} ${threshold}"
    fi

    echo "eval_path: ${eval_path}"

    if [ "$merge_mod" = "merge" ];then
        testdataset=(bbh bb agieval gsm8k gsm8k+ arc-c arc-e)
        lang=gen_bbh_cot_prefix
        if [[ $dataset == *mt-* ]]
        then
            lang="${lang}_mt"
        fi
        if [[ $dataset == *kard* ]]
        then
            lang="${lang}_kard"
        fi
        if [[ $dataset == bbh_ft ]]
        then
            lang="${lang}_ft"
        fi
        if [[ $dataset == *kaft* ]]
        then
            lang="${lang}_kaft"
        fi
        for eval_data in ${testdataset[@]}
        do
            bash config/eval.sh ${num_gpus} ${gpus} ${model} ${lang} 0 fewshot ${split_task} ${eval_data} 0.6 ${merge_path} ${eval_path} ${extra_args}
        done
        # bash config/eval.sh ${num_gpus} ${gpus} ${model} en 0 fewshot ${split_task} mmlu 0.6 ${merge_path} ${eval_path}
    fi
    if [ "$mod" = "eval_all" ];then
        testdataset=(bbh bb agieval gsm8k gsm8k+ arc-c arc-e)
        # lang=gen_bbh_cot_prefix
        if [[ $dataset == *mt-* ]]
        then
            lang="${lang}_mt"
        fi
        if [[ $dataset == *kard* ]]
        then
            lang="${lang}_kard"
        fi
        if [[ $dataset == bbh_ft ]]
        then
            lang="${lang}_ft"
        fi
        if [[ $dataset == *kaft* ]]
        then
            lang="${lang}_kaft"
        fi
        for eval_data in ${testdataset[@]}
        do
            bash config/eval.sh ${num_gpus} ${gpus} ${model} ${lang} 0 fewshot ${split_task} ${eval_data} 0.6 ${merge_path} ${eval_path} ${extra_args}
        done
    fi
    if [ "$mod" = "eval" ];then
        bash config/eval.sh ${num_gpus} ${gpus} ${model} ${lang} 0 fewshot ${split_task} ${eval_data} 0.6 ${merge_path} ${eval_path} ${extra_args}
    fi
fi