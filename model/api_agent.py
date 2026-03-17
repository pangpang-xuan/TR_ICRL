from openai import OpenAI
from typing import List
from model.base_agent import LLMAgent
import base64
import traceback
import time
import random
from transformers import AutoTokenizer 
import requests
import json
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
logging.getLogger('openai').setLevel(logging.WARNING)


class APIAgent(LLMAgent):
    def __init__(self, model_name, temperature=0, top_p=1.0) -> None:
        super().__init__(model_name, temperature, top_p)

        self.max_tokens = 8192
        
        if model_name in ["Qwen2.5-7B-Instruct", "Llama-3.1-8B-Instruct", "Qwen3-8B", "DeepSeek-R1-Distill-Llama-8B", 
                          "DeepSeek-R1-0528-Qwen3-8B", "GLM4-9B", "Gemma2-9B", "Mistral-7B-Instruct", "Qwen3-32B"]:
            print("VLLM")
            openai_api_key = "EMPTY"
            openai_api_base = "http://localhost:8848/v1"
            self.client = OpenAI(
                api_key=openai_api_key,
                base_url=openai_api_base,
                timeout=1200.0,
            )

        else:
            raise ValueError("Model not supported")

    def get_response(self, messages: List[dict]) -> str:
        if self.model_name in ["Qwen2.5-7B-Instruct", "Llama-3.1-8B-Instruct", "Qwen3-8B", "DeepSeek-R1-Distill-Llama-8B", 
                             "DeepSeek-R1-0528-Qwen3-8B", "GLM4-9B"]:
            local_model_dict = {
                "Qwen2.5-7B-Instruct": "models/huggingface.co/Qwen/Qwen2.5-7B-Instruct",
                "Llama-3.1-8B-Instruct": "models/huggingface.co/meta-llama/Llama-3.1-8B-Instruct",
                "Qwen3-8B": "models/huggingface.co/Qwen/Qwen3-8B",
                "DeepSeek-R1-Distill-Llama-8B":"models/huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                "DeepSeek-R1-0528-Qwen3-8B":"models/huggingface.co/deepseek-ai/DeepSeek-R1-0528-Qwen3-8B",
                "GLM4-9B":"models/huggingface.co/zai-org/glm-4-9b-chat"
            }
            local_model = local_model_dict[self.model_name]
            for _ in range(20):
                try:
                    completion = self.client.chat.completions.create(
                        messages=messages,
                        model=local_model,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        top_p = self.top_p
                    )
                    response = completion.choices[0].message.content
                    break
                except Exception as e:
                    if "bad_response_status_code" in str(e):
                        print("Bad Response")
                        response = "No answer provided: bad_response."
                        break
                    elif "content_filter" in str(e):
                        print("Content Filter")
                        response = "No answer provided: content_filter."
                        break
                    else:
                        print(e)
                        print(traceback.format_exc())
                        time.sleep(5)
                        response = "No answer provided."
        
        elif self.model_name in ["Gemma2-9B"]:
            messages = [m for m in messages if m["role"] != "system"]
            local_model_dict = {
                "Gemma2-9B":"models/huggingface.co/google/gemma-2-9b-it"
            }
            local_model = local_model_dict[self.model_name]
            for _ in range(10):
                try:
                    completion = self.client.chat.completions.create(
                        messages=messages,
                        model=local_model,
                        temperature=self.temperature,
                        top_p = self.top_p,
                        stream=True,
                    )
                    chunks = []
                    for chunk in completion:
                        if chunk.choices:
                            delta = chunk.choices[0].delta
                            if delta and delta.content:
                                chunks.append(delta.content)
                    response = "".join(chunks)
                    break
                except Exception as e:
                    if "bad_response_status_code" in str(e):
                        print("Bad Response")
                        response = "No answer provided: bad_response."
                        break
                    elif "content_filter" in str(e):
                        print("Content Filter")
                        response = "No answer provided: content_filter."
                        break
                    else:
                        print(e)
                        print(traceback.format_exc())
                        time.sleep(5)
                        response = "No answer provided."

        elif self.model_name in ["Qwen3-32B"]:
            local_model_dict = {
                "Qwen3-32B":"models/huggingface.co/Qwen/Qwen3-32B"
            }
            local_model = local_model_dict[self.model_name]
            for _ in range(10):
                try:
                    completion = self.client.chat.completions.create(
                        messages=messages,
                        model=local_model,
                        temperature=self.temperature,
                        top_p = self.top_p,
                        stream=True,
                        extra_body={
                            "chat_template_kwargs": {"enable_thinking": False},
                        }
                    )
                    chunks = []
                    for chunk in completion:
                        if chunk.choices:
                            delta = chunk.choices[0].delta
                            if delta and delta.content:
                                chunks.append(delta.content)
                    response = "".join(chunks)
                    break
                except Exception as e:
                    if "bad_response_status_code" in str(e):
                        print("Bad Response")
                        response = "No answer provided: bad_response."
                        break
                    elif "content_filter" in str(e):
                        print("Content Filter")
                        response = "No answer provided: content_filter."
                        break
                    else:
                        print(e)
                        print(traceback.format_exc())
                        time.sleep(5)
                        response = "No answer provided."
      
        else:
            for _ in range(10):
                try:
                    completion = self.client.chat.completions.create(
                        messages=messages,
                        model=self.model_name,
                        temperature=self.temperature,  
                        max_tokens=self.max_tokens,
                        top_p = self.top_p,
                        logprobs=True,
                        seed=0,
                    )
                    response = completion.choices[0].message.content
                    break
                except Exception as e:
                    if "bad_response_status_code" in str(e):
                        print("Bad Response")
                        response = "No answer provided: bad_response."
                        break
                    elif "content_filter" in str(e):
                        print("Content Filter")
                        response = "No answer provided: content_filter."
                        break
                    else:
                        print(e)
                        print(traceback.format_exc())
                        time.sleep(2)
                        response = "No answer provided."
        
        try:
            log_probs = completion.choices[0].logprobs.content
            log_probs = [token_logprob.logprob for token_logprob in log_probs]
        except Exception as e:
            log_probs = []
        return response, log_probs