import pandas as pd
import tiktoken

file_path = r'C:\Users\User\Desktop\intership\sources\Prompt Task.xlsx'
df = pd.read_excel(file_path)

def num_tokens(string, encoding_name="cl100k_base"):
    encoding = tiktoken.get_encoding(encoding_name)
    return len(encoding.encode(string))

def calculate_token_reduction(dataframe,num_of_prompts):
    dataframe['Num Tokens Original'] = dataframe['Original Prompt'][:num_of_prompts].apply(num_tokens)
    dataframe['Num Tokens Compressed'] = dataframe['Compressed Prompt'][:num_of_prompts].apply(num_tokens)
    dataframe['Token Reduction Percentage 1'] = ((dataframe['Num Tokens Original'] - dataframe['Num Tokens Compressed']) / dataframe['Num Tokens Original']) * 100

    dataframe['Difference'] = dataframe['Token Reduction Percentage'] - dataframe['Token Reduction Percentage 1']

    dataframe.to_excel('sources/result_2.xlsx', index=False)

calculate_token_reduction(df,15)
