from transformers import BertTokenizer, BertForSequenceClassification
import requests
import Levenshtein
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import pearsonr,spearmanr

def get_embeddings(input_text, model_name, api_key):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }
    data = {
        "input": input_text,
        "model": model_name
    }
    response = requests.post("https://api.openai.com/v1/embeddings", headers=headers, json=data)
    return response.json()

# Set your API key here
api_key = "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"
model_name_ada = "text-embedding-ada-002"

input_text1 = "The sun sets behind the mountains, painting the sky with warm hues."
input_text2 = "The sun sets."

ada_embeddings1 = get_embeddings(input_text1, model_name_ada, api_key)
ada_embeddings2 = get_embeddings(input_text2, model_name_ada, api_key)

#print(ada_embeddings2['data'][0]['embedding'])

# # Extract embeddings from the API response
embeddings1 = np.array(ada_embeddings1['data'][0]['embedding'])
embeddings2 = np.array(ada_embeddings2['data'][0]['embedding'])

# Calculate cosine similarity between embeddings
cosine_sim = cosine_similarity(embeddings1.reshape(1, -1), embeddings2.reshape(1, -1))[0, 0]
# Calculate Pearson Correlation between embeddings
embeddings1_normalized = embeddings1 / np.sum(embeddings1)
embeddings2_normalized = embeddings2 / np.sum(embeddings2)
pearson_corr, _ = pearsonr(embeddings1, embeddings2)

# Calculate Spearman Correlation between embeddings
spearman_corr, _ = spearmanr(embeddings1, embeddings2)
levenshtein_dist = Levenshtein.distance(input_text1, input_text2)

print("Ada Model Embeddings for Input 1:", ada_embeddings1)
print("Ada Model Embeddings for Input 2:", ada_embeddings2)
print("Cosine Similarity:", cosine_sim)
print("Pearson Correlation:", pearson_corr)
print("Spearman Correlation:", spearman_corr)
print("Levenshtein Distance:", levenshtein_dist)
