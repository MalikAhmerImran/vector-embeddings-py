from bert_score import score

string1 = "My name is Malik Ahmer Imran.I am doing Computer Engineering from national university of technoogy the department of pakistan army.I am currently in the final year"
string2 = "this is black cat"

P, R, F1 = score([string1], [string2], lang="en")

print(f'Precision: {P.item()}')
print(f'Recall: {R.item()}')
print(f'F1 Score: {F1.item()}')
