import os
import json
from tqdm import tqdm
from setup import *
from utils import *
import argparse
import traceback
from config import PromptTemplates
from concurrent.futures import ThreadPoolExecutor, as_completed
from embedding import get_embedding, retrieve_most_similar, retrieve_less_similar
import logging
from copy import deepcopy
import re
import copy
import random

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logging.getLogger('openai').setLevel(logging.WARNING)


def init_file_if_needed(file_path, remove_cache):
    if os.path.exists(file_path) and remove_cache:
        os.remove(file_path)
    if not os.path.exists(file_path):
        with open(file_path, 'w', encoding='utf-8') as f:
            pass

def write_to_file(file_path, data):
    with open(file_path, 'a+', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def zero_shot_ao_r1(prompt_templates, input_sample, llm_agent, output, dataset, task, messages):
    question = input_sample['question'].strip()
    prompt = prompt_templates.zero_shot_ao_prompt.format(question=question)
    output['prompt'] = prompt
    messages.append({
        "role": "user",
        "content": prompt
    })
    response, conf = llm_agent.get_response(messages)
    response = response.strip()
    # Append the assistant's response to messages
    messages.append({"role": "assistant", "content": response})
    prediction = extract_boxed_answer_r1(response)
    # Log the interaction
    logging.debug(f"【User】{prompt}\n\n")
    logging.debug(f"【Assistant】{response}\n\n")
    return response, prediction, conf, messages

def zero_shot_cot_r1(args, prompt_templates, input_sample, llm_agent, output, dataset, task, messages):
    prompt = prompt_templates.zero_shot_cot_prompt_r1.format(question=input_sample['question'].strip())
    output['prompt'] = prompt
    messages.append({"role": "user", "content": prompt})
    response, conf = llm_agent.get_response(messages)
    response = response.strip()
    # Append the assistant's response to messages
    messages.append({"role": "assistant", "content": response})
    prediction = extract_boxed_answer_r1(response)
    # Log the interaction
    logging.info(f"【User】{prompt}\n\n")
    logging.info(f"Reproduction【Assistant】{response}\n\n")
    return response, prediction, conf, messages


def extract_boxed_answer_r1(text):
    if text is None or len(text) == 0:
        return None
    if len(text) == 1:
        return text
    match = re.search(r'\\boxed{((?:[^{}]|\{[^{}]*\})*)}', text)
    if match:
        inner_text = match.group(1)
        if len(inner_text) == 0:
            return None
        elif len(inner_text) != 1:
            text_match = re.search(r'\\text\{([A-Za-z])\}', inner_text)
            if text_match:
                return text_match.group(1)
            else:
                return inner_text
        else:
            return inner_text
    else:
        match = re.search(r'\\boxed{(.*)}', text)
        if match:
            inner_text = match.group(1)
            if inner_text.startswith('(') and inner_text.endswith(')'):
                inner_text = inner_text[1:-1]
            return inner_text
    answer_match = re.search(
        r'(?:Final\s+)?Answer\s*:\s*\(?([A-Z])\)?',
        text,
        re.IGNORECASE
    )
    if answer_match:
        return answer_match.group(1)
    return None

def set_correctness(label, prediction, dataset, input_sample):
    if prediction is None:
        return False
    if label==prediction:
        return True
    else:
        return False

def positive_messages(prompt_templates, llm_agent, message, response):
    positive_message = deepcopy(message)
    positive_message.append({"role": "assistant", "content": response})
    positive_message.append({"role": "user", "content": prompt_templates.positive_feedback})
    feedback, conf = llm_agent.get_response(positive_message)
    positive_message.append({"role": "assistant", "content": feedback})
    return positive_message

def negative_messages(prompt_templates, llm_agent, message, response):
    negative_message = deepcopy(message)
    negative_message.append({"role": "assistant", "content": response})
    negative_message.append({"role": "user", "content": prompt_templates.negetive_feedback})
    feedback, conf = llm_agent.get_response(negative_message)
    negative_message.append({"role": "assistant", "content": feedback})
    return negative_message

def rethinking_item(args, idx, all_answers, responses, prompt_templates, llm_agent, messages, fake_reward):
    response = responses[idx]
    item = all_answers[idx]
    if args.reward: # use real reward
        if item == fake_reward:
            return positive_messages(prompt_templates, llm_agent, messages, response)
        else:
            return negative_messages(prompt_templates, llm_agent, messages, response)
    else: # use fake reward
        if item == fake_reward:
            return negative_messages(prompt_templates, llm_agent, messages, response)
        else:
            return positive_messages(prompt_templates, llm_agent, messages, response)

def tr_icrl(args, prompt_templates, llm_agent, retrieval_results, ground_truth):
    messages = [[{"role": "system", "content": prompt_templates.zero_shot_system_role}] for _ in range(args.rollout)]
    icrl_num_threads = args.rollout
    for res_idx, retrieval_result in enumerate(retrieval_results):
        for message in messages:
            message.append({"role": "user", "content": prompt_templates.zero_shot_cot_prompt_r1.format(question=retrieval_result.strip())})
        responses = []
        def get_single_response(msg):
            res, conf = llm_agent.get_response(msg)
            return res
        # 1st step rollout
        with ThreadPoolExecutor(max_workers=icrl_num_threads) as executor:
            responses = list(executor.map(get_single_response, messages))
        # 2nd step get fake labels
        predictions_candidate = [] # answers with [real]
        all_answers = [] # answers with [NULL and real]
        for response in responses:
            prediction_candidate = extract_boxed_answer_r1(response)
            if prediction_candidate is not None:
                predictions_candidate.append(prediction_candidate)
            all_answers.append(prediction_candidate)
        fake_reward = ""
        if args.unlabel:
            fake_reward = vote(predictions_candidate)
        else:
            fake_reward = ground_truth[0]
        logging.info(f"voting label is {fake_reward}")
        final_results = []
        # 3rd step get feedback
        def process_feedback(i):
            return rethinking_item(
                args, 
                i, 
                all_answers, 
                responses, 
                prompt_templates, 
                llm_agent, 
                messages[i],
                fake_reward
            )
        with ThreadPoolExecutor(max_workers=icrl_num_threads) as executor:
            final_results = list(executor.map(process_feedback, range(len(messages))))
        messages = final_results
    return messages

def tts_cure(args, prompt_templates, input_sample, llm_agent, output, context_rewards):
    if args.majority_vote: # majority voting
        max_workers = args.rollout
    else: # not majority voting
        max_workers = 1
        context_rewards = [context_rewards[0]]
    futures = []
    def worker(context_reward):
        item_message = deepcopy(context_reward)
        item_message.append({
            "role": "user",
            "content": prompt_templates.zero_shot_cot_prompt_r1.format(
                question=input_sample["question"].strip()
            )
        })
        response, conf = llm_agent.get_response(item_message)
        item_message.append({"role": "assistant", "content": response})
        return response, extract_boxed_answer_r1(response), conf, item_message

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for context_reward in context_rewards:
            futures.append(executor.submit(worker, context_reward))

    responses, predictions, confs, tts_messages, good_preds = [], [], None, [], []
    for f in as_completed(futures):
        response, pred, conf, item_message = f.result()
        responses.append(response)
        predictions.append(pred)
        if pred is not None:
           good_preds.append(pred)
        tts_messages.append(item_message)
    print(good_preds) 
    prediction = vote(good_preds)
    return responses, predictions, prediction, confs, tts_messages

def complete_item(args, task, llm_agent, prompt_templates, input_sample, demonstrations_messages, index):
    try:
        # Inference
        output = input_sample
        predictions = []
        logging.debug("-------------------------------------------")
        logging.info(f"### Start Index: {index}")
        ground_truth = ""
        if args.method != "zero_shot":
            # 1st phase get reward in text
            if args.retrieval:
                if args.method != "zero_shot": # zero_shot
                    query = input_sample["question"]
                    if args.similar == "least": # least similar # query, documents, vector_dbs, steps
                        retrieval_results, best_indexs = retrieve_less_similar(query, args.trains_data, args.vector_dbs, args.step)
                    else: # most similar
                        retrieval_results, best_indexs = retrieve_most_similar(query, args.trains_data, args.vector_dbs, args.step)

                    if not args.unlabel: # using train data
                        ground_truth = [args.lebels[index] for index in best_indexs]
            
            else: # random 
                query = input_sample["question"]
                docs = args.trains_data.copy()
                if query in docs:
                    docs.remove(query)
                retrieval_results = docs[:args.step]
                logging.info(f"query is {query}")

        if args.sequence == "upper":
            retrieval_results = retrieval_results
        elif args.sequence == "descending":
            retrieval_results = retrieval_results[::-1]
        elif args.sequence == "random":
            retrieval_results = random.sample(retrieval_results, len(retrieval_results))
        else:
            raise Exception("Sequence Error!")
        
        if args.method == "zero_shot": # how can we get content message
            logging.info("zero shot!")
        elif args.method == "tr_icrl":
            in_context_rewards = tr_icrl(args, prompt_templates, llm_agent, retrieval_results, ground_truth)
        else:
            raise Exception('Method Error!')

        if args.method == "zero_shot":
            if args.prompting_type == "ao":
                messages = [{"role": "system", "content": prompt_templates.zero_shot_ao_system_role}]
                responses, prediction, conf, messages = zero_shot_ao_r1(prompt_templates, input_sample, llm_agent, output,
                                                                    args.dataset, task, messages)
            elif args.prompting_type == "cot":
                messages = [{"role": "system", "content": prompt_templates.zero_shot_system_role}]
                responses, prediction, conf, messages = zero_shot_cot_r1(args, prompt_templates, input_sample, llm_agent, output,
                                                                     args.dataset, task, messages)
        
        elif args.method == "tr_icrl":
            responses, predictions, prediction, conf, messages = tts_cure(args, prompt_templates, input_sample, llm_agent, output, in_context_rewards)
        else:
            raise Exception('Method Error!')

        # Evaluation
        logging.debug(f"### Label: {input_sample['answer']}")
        logging.debug(f"### Prediction: {prediction}")
        label = input_sample['answer'][0] if isinstance(input_sample['answer'], list) else input_sample['answer']
        is_correct = set_correctness(label, prediction, args.dataset, input_sample)
        logging.info(f"### Completed Index: {index} [{is_correct}]")
        
        # Prepare final output
        output['messages'] = messages
        output['responses'] = responses
        output['prediction'] = predictions
        output['label'] = label
        output['vote_prediction'] = prediction
        output['correct'] = is_correct

    except Exception as e:
        logging.error(f"Error: {e}")
        logging.error(traceback.format_exc())
        exit()

    return output, conf, index

def general_inference(args, task, llm_agent, prompt_templates):
    if args.method == "tr_icrl":
        if args.retrieval:
            retrieval = "retrieval"
        else:
            retrieval = "not_retrieval"
        if args.majority_vote:
            tts = "majority_vote"
        else:
            tts = "not_majority_vote"
        if args.reward:
            reward = "reward"
        else:
            reward = "spurious_reward"
        if args.unlabel:
            unlabel = "unsupervised"
        else:
            unlabel = "supervised"
        
        tmp_dir = os.path.join(f"outputs", args.model, args.dataset, args.method, args.prompting_type, args.similar, retrieval, tts, reward, unlabel, args.sequence, f"step_{args.step}") # output path
    
    elif args.method == "zero_shot":
        tmp_dir = os.path.join(f"outputs", args.model, args.dataset, args.method, args.prompting_type)
    
    else:
        raise Exception('Method Error!')
    
    args.tmp_dir = tmp_dir
    
    # Path Config
    task_suffix = f"_{task}" if task else ""
    input_path = os.path.join(INPUT_PATH.format(dataset=args.dataset), f"{args.dataset}{task_suffix}_input.jsonl")
    tmp_path = os.path.join(tmp_dir, f"{args.dataset}{task_suffix}_rollout{args.rollout}_temp{args.temperature}.jsonl")

    # File Initialization
    os.makedirs(tmp_dir, exist_ok=True)
    init_file_if_needed(tmp_path, args.remove_cache)

    # Load Inputs
    outputs = []
    with open(tmp_path, 'r', encoding='utf-8') as f:
        outputs = [json.loads(line) for line in f if line.strip()]
    if args.dataset in ["aime2024", "aime2025"]:
        inputs = []
        for _ in range(16):
            with open(input_path, 'r', encoding='utf-8') as f:
                inputs.extend(json.loads(line) for line in f if line.strip())
    else:
        with open(input_path, 'r', encoding='utf-8') as f:
            inputs = [json.loads(line) for line in f if line.strip()]

    start = len(outputs)
    end = min(args.max_samples if args.max_samples != -1 else len(inputs), len(inputs))
    logging.info(start)
    logging.info(end)

    assert end > 0

    batch_size = min(max(1, min(args.num_threads * 1, 500)), end - start)
    if start >= end:
        return
    
    # Load Demonstration
    demonstrations_messages = []
    if args.method == "tr_icrl":
        if args.unlabel:
            if args.similar == "another":
                retrieval_dataset = ""
                if args.dataset in ["math500", "amc", "gsm8k", "aime2024", "aime2025"]:
                    retrieval_dataset = "medqa"
                elif args.dataset in ["medqa" ,"medxpertqa"]:
                    retrieval_dataset = "math500"
                else:
                    raise Exception("Dataset Error!")
                logging.info(f"retrieval dataset is {retrieval_dataset}")
                train_path = os.path.join(INPUT_PATH.format(dataset=retrieval_dataset), f"{retrieval_dataset}{task_suffix}_input.jsonl")
            else:
                train_path = input_path
            args.trains_data = load_embedding(args.max_trains, train_path) # test data input
        else:
            demostrations_path = os.path.join(DEMO_PATH.format(dataset=args.dataset), f"{args.dataset}{task_suffix}_input.jsonl")
            args.trains_data = load_embedding(args.max_trains, demostrations_path) # train data input
            args.labels = load_answers(args.max_trains, demostrations_path)
        
        if args.retrieval:
            args.vector_dbs = get_embedding(args.trains_data)
        else:
            args.vector_dbs = []


    for i in range(start, end, batch_size):
        batch_start, batch_end = i, min(i + batch_size, end)
        inputs_process = inputs[batch_start:batch_end]

        # Single Thread
        if args.num_threads == 1:
            for index, input_sample in enumerate(inputs_process):
                result=[]
                output, conf, _ = complete_item(args, task, llm_agent, prompt_templates, input_sample,
                                                demonstrations_messages, batch_start + index + 1)
                result.append(output)
                write_to_file(tmp_path, result)

        # Multi Thread
        else:
            num_threads = min(args.num_threads,
                                len(inputs_process))  # Ensure num_threads is less than or equal to the batch size
            chunk_size = max(1, len(inputs_process) // num_threads)  # Ensure chunk_size is at least 1
            futures = []
            with ThreadPoolExecutor(max_workers=num_threads) as executor:
                for chunk_index in range(0, len(inputs_process), chunk_size):
                    chunk = inputs_process[chunk_index:chunk_index + chunk_size]
                    for index, item in enumerate(chunk):
                        futures.append(
                            executor.submit(
                                complete_item,
                                args,
                                task,
                                llm_agent,
                                prompt_templates,
                                item,
                                demonstrations_messages,
                                batch_start + chunk_index + index + 1
                            )
                        )
                batch_results = []
                for future in as_completed(futures):
                    output, conf, index = future.result()
                    batch_results.append(output)
                write_to_file(tmp_path, batch_results)
                        

if __name__ == '__main__':
    print("start")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='gpt-4o-2024-11-20', type=str)
    parser.add_argument("--dataset", default='math500', type=str)
    parser.add_argument("--task", default="text", type=str)
    parser.add_argument("--method", default='tr_icrl', type=str, choices = ["tr_icrl", "zero_shot"])
    parser.add_argument("--prompting-type", default='cot', type=str, choices = ["ao", "cot"])
    parser.add_argument("--remove-cache", default=False, action='store_true', help="remove cache")
    parser.add_argument("--num-threads", default=4, type=int)
    parser.add_argument("--temperature", default=0.7, type=float)
    parser.add_argument("--top_p", default=0.8, type=float)
    parser.add_argument("--max-samples", default=-1, type=int)
    parser.add_argument("--max_trains", default=-1, type=int)

    parser.add_argument("--retrieval", default=True, type=str2bool)
    parser.add_argument("--rollout", default=8, type=int)
    parser.add_argument("--reward", default=True, type=str2bool)
    parser.add_argument("--majority_vote", default=True, type=str2bool)
    parser.add_argument("--unlabel", default=True, type=str2bool)
    parser.add_argument("--step", default=1, type=int)
    parser.add_argument("--sequence", default="upper", type=str, choices = ["upper", "descending", "random"])
    parser.add_argument("--similar", default="most", type=str)


    args = parser.parse_args()
    print(args.num_threads)

    llm_agent, tasks = setup(args.model, args.dataset, args.method, args.prompting_type)
    llm_agent.temperature = args.temperature
    llm_agent.top_p = args.top_p

    tasks = [args.task] if args.task else tasks

    prompt_templates = PromptTemplates().load_templates(args.dataset, args.model)

    if tasks:
        for task in tasks:
            general_inference(args, task, llm_agent, prompt_templates)
    else:
        general_inference(args, None, llm_agent, prompt_templates)
    
    print("process done")
