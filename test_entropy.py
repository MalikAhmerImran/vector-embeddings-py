from scipy.stats import entropy
from collections import Counter
from ahmer import score


# Generate two strings
string1 = "My name is Malik Ahmer Imran.I am doing Computer Engineering from national university of technoogy the department of pakistan army.I am currently in the final year"
string2 = "this is black cat"

def calculate_string_entropy(s):
    # Tokenize the string by splitting on spaces (for simplicity)
    tokens = s.split()
    
    # Count the occurrences of each token
    token_counts = Counter(tokens)
    
    # Calculate the probability distribution
    total_tokens = sum(token_counts.values())
    probabilities = [count / total_tokens for count in token_counts.values()]
    
    # Calculate the entropy
    return entropy(probabilities)

entropy1 = calculate_string_entropy(string1)
entropy2 = calculate_string_entropy(string2)
P, R, F1 = score([string1], [string2], lang="en")

print(f'Precision: {P.item()}')
print(f'Recall: {R.item()}')
print(f'F1 Score: {F1.item()}')


print('Entropy of string1:', entropy1)
print('Entropy of string2:', entropy2)
