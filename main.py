from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
import pickle
from nltk.tokenize import word_tokenize
import re
import string
from model import GRUModel
from nltk.corpus import stopwords
import snowballstemmer

app = FastAPI()

with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

embedding_matrix = np.load("embedding_matrix.npy")
with open("word_index.pkl", "rb") as f:
    word_index = pickle.load(f)

# Preprocessing function for text
def preprocess_text(text):
    nepali_punctuation = "।,?!;:\"'()[]{}-…"
    custom_punctuation = string.punctuation + nepali_punctuation
    pattern = r'[^\u0900-\u095F\u0970-\u097F\s]'
    text=re.sub(pattern,'',text)
    text=text.translate(str.maketrans('','',custom_punctuation))
    tokens=word_tokenize(text)
    stop_word=set(stopwords.words('nepali'))
    tokens=[word for word in tokens if word not in stop_word]
    stemmer=snowballstemmer.stemmer('nepali')
    tokens=[stemmer.stemWord(word) for word in tokens]
    clean_text=' '.join(tokens)
    return clean_text

def text_to_embedding(tokens, max_len=100, embedding_dim=100):
    embeddings = []
    for token in tokens:
        if token in word_index:
            embeddings.append(embedding_matrix[word_index[token]])
        else:
            embeddings.append(np.zeros(embedding_dim))

    # Padding
    if len(embeddings) > max_len:
        embeddings = embeddings[:max_len]
    else:
        padding = [np.zeros(embedding_dim) for _ in range(max_len - len(embeddings))]
        embeddings.extend(padding)

    return np.array(embeddings)

# Load the GRU model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GRUModel(embedding_dim=100, hidden_size=128, num_layers=3, output_dim=5)
model.load_state_dict(torch.load("model/news_gru_model.pth"))
model.to(device)
model.eval()

# Define input data structure
class NewsItem(BaseModel):
    text: str

# Prediction endpoint
@app.post("/predict")
def predict_category(item: NewsItem):
    # Preprocess input text
    tokens = preprocess_text(item.text)
    
    # Convert text to embeddings
    input_embeddings = text_to_embedding(tokens)
    
    # Convert to tensor
    input_tensor = torch.tensor([input_embeddings], dtype=torch.float32).to(device)
    
    # Get prediction from model
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1).item()

    # Return the predicted category
    return {"category": prediction}
