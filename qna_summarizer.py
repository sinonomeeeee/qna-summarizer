import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
import numpy as np

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModel.from_pretrained(MODEL_NAME)

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size())
    return torch.sum(token_embeddings * input_mask_expanded, 1) / \
           torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_sentence_embeddings(sentences):
    with torch.no_grad():
        encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
        model_output = model(**encoded_input)
        embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
        return embeddings.cpu().numpy()

def summarize_questions(questions, n_clusters=13):
    if not questions:
        return []

    if n_clusters is None or n_clusters < 1:
        n_clusters = 5

    n_clusters = min(n_clusters, len(questions))

    embeddings = get_sentence_embeddings(questions)
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    clustered = {}
    for label, question in zip(labels, questions):
        clustered.setdefault(label, []).append(question)

    summary = []
    for cluster_id, cluster_questions in clustered.items():
        representative = max(cluster_questions, key=len)
        summary.append({
            "cluster": cluster_id + 1,
            "representative": representative,
            "questions": cluster_questions
        })

    return summary


