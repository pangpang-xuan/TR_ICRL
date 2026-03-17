
models=${1:-"Llama-3.1-8B-Instruct"} # Qwen2.5-7B-Instruct Qwen3-8B Llama-3.1-8B-Instruct DeepSeek-R1-0528-Qwen3-8B
datasets=${2:-"medxpertqa"}  # math500,amc,gsm8k,aime2024,aime2025,medqa,medxpertqa
tasks=${3:-"text"}

method=${4:-"tr_icrl"} # tr_icrl zero_shot
prompting_type=${5:-"cot"} # cot ao
temperature=${6:-0.6}
top_p=${7:-0.8}

rollout=${8:-8}
reward=${9:-True}
majority_vote=${10:-True}
unlabel=${11:-True}
steps=${12:-1,2,3,4,5,6,7,8,9,10} # 1,2,3,4,5,6,7,8,9,10
sequence=${13:-"upper"} # "upper", "descending", "random"
retrieval=${14:-True}
similar=${15:-least} # most least another

IFS=","

pids=()

for model in $models; do
    for dataset in $datasets; do
        for task in $tasks; do
            for step in $steps; do
                date +"%Y-%m-%d %H:%M:%S"
                echo "Model: ${model}"
                echo "Dataset: ${dataset}"
                echo "Task: ${task}"
                echo "Method: ${method}"
                echo "Prompting Type: ${prompting_type}"
                echo "Rollout Number: ${rollout}"
                echo "Retrieval: ${retrieval}"
                echo "Majority Vote: ${majority_vote}"
                echo "Reward: ${reward}"
                echo "Unlabel: ${unlabel}"
                echo "Step: ${step}"
                echo "Sequence: ${sequence}"
                echo "Similar: ${similar}"
                
                log_dir="outputs/${model}/${dataset}/${method}_logs/${prompting_type}/${similar}/${retrieval}_retrieval/${majority_vote}_majority_vote/${reward}_reward/${unlabel}_unlabel/${sequence}/steps_${step}/logs"
                if [ ! -d "${log_dir}" ]; then
                    mkdir -p "${log_dir}"
                fi
                log_file="${log_dir}/run-$(date +'%y%m%d_%H%M_%S')-${dataset}-${task}_rollout${rollout}_temp${temperature}.log"
                
                cp "${BASH_SOURCE[0]}" "${log_dir}/run.sh"
                cp main.py "${log_dir}/main.py"
                cp utils.py "${log_dir}/utils.py"
                cp model/api_agent.py "${log_dir}/api_agent.py"
                cp config/prompt_templates.py "${log_dir}/prompt_templates.py"
                nohup python main.py --model "${model}" \
                        --dataset "${dataset}" \
                        --task "${task}" \
                        --method "${method}" \
                        --prompting-type "${prompting_type}" \
                        --temperature "${temperature}" \
                        --top_p "${top_p}" \
                        --retrieval "${retrieval}" \
                        --rollout "${rollout}" \
                        --reward "${reward}" \
                        --majority_vote "${majority_vote}" \
                        --unlabel "${unlabel}" \
                        --step "${step}" \
                        --sequence "${sequence}" \
                        --similar "${similar}" \
                        > "${log_file}" 2>&1 &
                
            done
        done
    done
done



# bash scripts/tr_icrl.sh