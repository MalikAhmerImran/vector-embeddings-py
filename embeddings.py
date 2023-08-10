import pandas as pd
import requests
import faiss
import numpy as np

def get_embedding(text, model_name, api_key):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "input": text,
        "model": model_name
    }
    response = requests.post("https://api.openai.com/v1/embeddings", headers=headers, json=data)
    return np.array(response.json()['data'][0]['embedding'])

# Set your API key and model name here
api_key = "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"
model_name = "text-embedding-ada-002"

# Read the Excel file
file_path = r'C:\Users\User\Desktop\intership\sources\Prompt Task.xlsx'
data = pd.read_excel(file_path)

# Define the columns you want to process
original_prompt_column = "Original Prompt"
decompressed_prompt_column = "Decompressed Prompt"

# Slice the first 15 entries for both columns
original_prompt_texts = data[original_prompt_column].head(15)
decompressed_prompt_texts = data[decompressed_prompt_column].head(15)

# Concatenate embeddings into a single array
embedding_matrix = np.vstack([get_embedding(text, model_name, api_key) for text in original_prompt_texts] +
                             [get_embedding(text, model_name, api_key) for text in decompressed_prompt_texts])

# Get the dimension of the embeddings
d = embedding_matrix.shape[-1]

# Make FAISS available
index = faiss.IndexFlatL2(d)  # Build the index

# Add vectors to the index
index.add(embedding_matrix)

# Print the FAISS index (vector database)
print("FAISS Index (Vector Database):")
print(index)

# Get the number of embeddings stored in the index
num_embeddings = index.ntotal

# Print each embedding
for i in range(num_embeddings):
    vector = index.reconstruct(i)
    print(f"Embedding at index {i}:")
    print(vector)

print("Embeddings processed and stored in FAISS index")
faiss.write_index(index, "hy_index.faiss")
