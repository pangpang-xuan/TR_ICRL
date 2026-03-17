import itertools
import logging
import re
import numpy as np
from collections import Counter
import requests
import random
import time
from requests.exceptions import ConnectionError
import io
from itertools import islice
from openai import OpenAI
import math
import json

def extract_boxed_answer_r1(text):
    if len(text)==1:
        return text
    match = re.search(r'\\boxed{(.*?)}', text)
    return match.group(1) if match else None


def vote(choices) -> str:
    if not choices:
        return "Z"
    
    frequency = Counter(choices)
    max_count = max(frequency.values())
    candidates = [key for key, value in frequency.items() if value == max_count]
    result = random.choice(candidates)
    logging.info(
        f"Fake Label Voting completed. Total {len(choices)} choices. "
        f"Most frequent: '{result}' (appeared {max_count} times, randomly selected from {candidates})"
    )
    return result


def load_embedding(max_trains, input_path):
    trains = []
    answers = []
    with open(input_path, 'r', encoding='utf-8') as f:
        non_empty_lines = (line for line in f if line.strip())
        if max_trains != -1:
            non_empty_lines = islice(non_empty_lines, max_trains)
        for line in non_empty_lines:
            data = json.loads(line)
            trains.append(data["question"])
    logging.info("train data load over")
    return trains

def load_answers(max_trains, input_path):
    answers = []
    with open(input_path, 'r', encoding='utf-8') as f:
        non_empty_lines = (line for line in f if line.strip())
        if max_trains != -1:
            non_empty_lines = islice(non_empty_lines, max_trains)
        for line in non_empty_lines:
            data = json.loads(line)
            answers.append(data["label"])
    logging.info("train data load over")
    return answers

def str2bool(v):
    if isinstance(v, bool):
        return v
    return v.lower() in ("true", "1", "yes", "y")



