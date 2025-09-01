dataset="chair"
method="greedy"
max_new_tokens=512
num_chunks=8
seed=0
GPUS=(0 1 2 3 4 5 6 7)

# -------------------- Functions --------------------
run_inference() {
    local model_dir=$1
    local gamma=$2
    local layer=$3
    local seed=$4

    local model_path="/path/to/model_cache/$model_dir"  #TODO: update model path
    local arr=(${model_dir//--/ })
    local model_name=${arr[2]}

    output_dir="results/$model_name/CHAIR/"
    output_file="SSL-$dataset-$method-$max_new_tokens-$seed.json"
    full_output_path="$output_dir/$output_file"

    if [[ ! -f "$full_output_path" ]]; then
        mkdir -p "$output_dir"

        for i in $(seq 0 $((num_chunks-1))); do
            gpu_id=${GPUS[$i]}
            python main.py --model_path $model_path \
                            --dataset $dataset \
                            --output_file "$output_dir/$i-$output_file" \
                            --num_chunks $num_chunks \
                            --max_new_tokens $max_new_tokens \
                            --seed $seed \
                            --device cuda:$gpu_id \
                            --chunk_idx $i \
                            --layer $layer \
                            --gamma $gamma &
        done

        wait

        > "$full_output_path"
        for i in $(seq 0 $((num_chunks-1))); do
            cat "$output_dir/$i-$output_file" >> "$full_output_path"
            rm "$output_dir/$i-$output_file"
        done
    fi
}

run_eval() {
    local model_dir=$1
    local gamma=$2
    local layer=$3
    local seed=$4

    local arr=(${model_dir//--/ })
    local model_name=${arr[2]}
    local output_dir="results/$model_name/CHAIR/"
    local output_file="SSL-$dataset-$method-$max_new_tokens-$seed.json"
    local result_file="$output_dir/$output_file"

    python eval/eval.py \
        --eval_chair \
        --result-file "$result_file" 
    echo "Results above from $result_file"
}

# -------------------- Start Inference --------------------
# run_inference "models--llava-hf--llama3-llava-next-8b-hf/" 0.6 "15" $seed
# wait
# run_eval "models--llava-hf--llama3-llava-next-8b-hf/" 0.6 "15" $seed


# run_inference "models--llava-hf--llava-1.5-7b-hf/" 0.8 "30" $seed
# wait
# run_eval "models--llava-hf--llava-1.5-7b-hf/" 0.8 "30" $seed


run_inference "models--salesforce--instructblip-vicuna-7b/" 0.10 "7" $seed
wait
run_eval "models--salesforce--instructblip-vicuna-7b/" 0.10 "7" $seed


# run_inference "models--meta-llama--Llama-3.2-11B-Vision-Instruct/" 0.4 "31" $seed
# wait
# run_eval "models--meta-llama--Llama-3.2-11B-Vision-Instruct/" 0.4 "31" $seed