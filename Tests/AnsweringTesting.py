import json
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F

with open("questionBank.json", "r", encoding="utf-8") as f:
    data = json.load(f)


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0] #First element of model_output contains all token embeddings
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)    

def answer(sentence):

    sentences = [sentence]

    # Load model from HuggingFace Hub
    tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

    # Tokenize sentences
    encoded_input = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')

    # Compute token embeddings
    with torch.no_grad():
        model_output = model(**encoded_input)

    # Perform pooling
    sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])

    # Normalize embeddings
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    sentence_vec = np.array(sentence_embeddings)

    maxSim = 0
    currentAnswer = ""
    for item in data["questions"]:
        item_vec = np.array(item["embedding"])
        dotOf = np.dot(sentence_vec.ravel(), item_vec.ravel())
        if dotOf > maxSim:
            maxSim = dotOf
            currentAnswer = item["answer"]

    if maxSim <= 0.3:
        return "I dont know!"
    else:
        return currentAnswer
    

question = input("ask me: ")

print(answer(question))