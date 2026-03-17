# TR-ICRL

## 📚Overview

We propose Test-Time Rethinking for In-Context Reinforcement Learning (TR-ICRL), a novel ICRL framework designed for both reasoning and knowledge-intensive tasks. TR-ICRL operates by first retrieving the most relevant instances from an unlabeled evaluation set for a given query.
During each ICRL iteration, LLM generates a set of candidate answers for every retrieved instance. Next, a pseudo-label is derived from this set through majority voting.
This label then serves as a proxy to give reward messages and generate formative feedbacks, guiding LLM through iterative refinement.
In the end, this synthesized contextual information is integrated with the original query to form a comprehensive prompt, with the answer determining through a final round of majority voting.

## 🔧Usage

1. Install Dependencies:

```
conda create --name TR-ICRL python=3.10
conda activate TR-ICRL
pip install -r requirements.txt
```

2. 🚀Launch the reasoning model and embedding model

```
bash embedding_vllm.sh # launch embedding model
bash reasoning_vllm.sh # launch reasoning model
```

We use VLLM Online inference and the port for embedding model is 8080 and the port for reasoning model in 8848.

3. 🤔Start

```
bash scripts/tt_icrl.sh
```

4. Meaning of some parameter variables:
   + models=${1:-"Qwen2.5-7B-Instruct"}: The name of the base language model to be evaluated. The model must be registered in config/model_info.json, and its corresponding inference interface must be implemented in model/api_agent.py.
   + datasets=${2:-"medqa"}: The benchmark dataset used for evaluation.
   + tasks=${3:-"text"}: The task modality type, indicating the input format of the benchmark: `text`: text-only reasoning tasks
   + method=${4:-"tr_icrl"}:  The reasoning or enhancement method applied during inference, such as:
     - `zero_shot`: Standard zero-shot inference without additional strategies
   + prompting_type=${5:-"cot"}: The prompting strategy used to guide model inference:
     - `cot`: Chain-of-Thought prompting
     - `ao`: Answer-Only prompting
   + temperature=${6:-0.6}
   + top_p=${7:-0.8}
   + rollout=${8:-8}: The number of independent rollouts generated per retrieval question.
   + majority_vote=${9:-True}
   + unlabel=${10:-True}
   + steps=${11:-1,2,3,4,5,6,7,8,9,10} # ICRL steps
   + sequence=${12:-"upper"} # contextual coherence
   + retrieval=${13:-True}
   + similar=${14:-least} # retrieved question distribution
