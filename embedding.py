import torch
from vllm import LLM
import os
from openai import OpenAI
import torch.nn.functional as F
import numpy as np
import random


task = 'Given an input question, retrieve the most semantically similar question from the dataset—prioritizing strong conceptual and contextual relevance.'


def format_query(task: str, query: str) -> str:
    return f"Instruct: {task}\nQuery: {query}"
    # return query

def get_embedding(input_texts):
    openai_api_key = "EMPTY"
    openai_api_base = "http://localhost:8000/v1"
    client = OpenAI(
        api_key=openai_api_key,
        base_url=openai_api_base,
        timeout=600.0,
    )
    model = "models/huggingface.co/Qwen/Qwen3-Embedding-8B"
    responses = client.embeddings.create(
        input = input_texts,
        model = model
    )
    embeddings = torch.tensor([o.embedding for o in responses.data])
    return embeddings

def retrieve_most_similar(query, documents, vector_dbs, steps):
    docs = documents.copy()
    vectors = vector_dbs.clone()
    if query in docs:
        index_to_remove = docs.index(query)
        docs.pop(index_to_remove)
        mask = torch.ones(len(vectors), dtype=torch.bool)
        mask[index_to_remove] = False
        vectors = vectors[mask]
    if len(docs) == 0:
        return None
    input_texts = [format_query(task, query)]
    embeddings = get_embedding(input_texts)
    query_emb = embeddings[0]
    scores = F.cosine_similarity(query_emb.unsqueeze(0), vectors)
    actual_k = min(steps, len(docs))
    top_scores, top_indices = torch.topk(scores, k=actual_k, largest=True)
    ascending_indices = top_indices.flip(0).tolist()
    result_docs = [docs[i] for i in ascending_indices]
    print(f"Top-{actual_k} indices (ascending similarity): {ascending_indices}")
    return result_docs, ascending_indices


def retrieve_less_similar(query, documents, vector_dbs, steps):
    docs = documents.copy()
    vectors = vector_dbs.clone()
    if query in docs:
        index_to_remove = docs.index(query)
        docs.pop(index_to_remove)
        mask = torch.ones(len(vectors), dtype=torch.bool)
        mask[index_to_remove] = False
        vectors = vectors[mask]
    if len(docs) == 0:
        return None
    input_texts = [format_query(task, query)]
    embeddings = get_embedding(input_texts)
    query_emb = embeddings[0]
    scores = F.cosine_similarity(query_emb.unsqueeze(0), vectors)
    actual_k = min(steps, len(docs))
    top_scores, top_indices = torch.topk(scores, k=actual_k, largest=False)
    ascending_indices = top_indices.flip(0).tolist()
    result_docs = [docs[i] for i in ascending_indices]
    print(f"Top-{actual_k} indices (ascending similarity): {ascending_indices}")
    return result_docs, ascending_indices



