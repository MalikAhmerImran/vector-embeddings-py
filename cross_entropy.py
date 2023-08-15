import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from scipy.stats import entropy
import numpy as np

# Load the data
file_path = r'C:\Users\User\Desktop\intership\sources\Prompt Task.xlsx'
data = pd.read_excel(file_path)

# Get original prompts and decompressions as separate lists for the first 15 rows
original_prompts = data['Original Prompt'].head(74).tolist()
decompressions = data['Decompressed Prompt'].head(74).tolist()

def calculate_entropy(texts):
    # Convert to term frequency representation
    vectorizer = CountVectorizer()
    tf = vectorizer.fit_transform(texts)

    # Convert to dense array and to float type
    tf = tf.toarray().astype(float)

    # Add smoothing
    epsilon = 1e-9
    tf += epsilon

    # Normalize to get probability distribution
    prob = tf.sum(axis=0) / tf.sum()

    # Calculate entropy
    return entropy(prob)
from scipy.special import kl_div

def calculate_cross_entropy(texts1, texts2):
    # Convert both sets of texts to term frequency representation
    vectorizer = CountVectorizer()
    tf1 = vectorizer.fit_transform(texts1)
    tf2 = vectorizer.transform(texts2)

    # Convert to dense array and to float type
    tf1 = tf1.toarray().astype(float)
    tf2 = tf2.toarray().astype(float)

    # Add smoothing
    epsilon = 1e-9
    tf1 += epsilon
    tf2 += epsilon

    # Normalize to get probability distributions
    prob1 = tf1.sum(axis=0) / tf1.sum()
    prob2 = tf2.sum(axis=0) / tf2.sum()

    # Calculate cross-entropy
    return entropy(prob1, prob2)

cross_entropy_value = calculate_cross_entropy(original_prompts, decompressions)


original_prompts_entropy = calculate_entropy(original_prompts)
decompressions_entropy = calculate_entropy(decompressions)

print('Entropy of Original Prompts:', original_prompts_entropy)
print('Entropy of Decompressed Prompts:', decompressions_entropy)
print('Cross-Entropy between Original Prompts and Decompressed Prompts:', cross_entropy_value)
