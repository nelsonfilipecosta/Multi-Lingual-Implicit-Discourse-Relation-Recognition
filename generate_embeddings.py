import re
import time
import pandas as pd
from sentence_transformers import SentenceTransformer

path = 'Data/PDTB-3.0/pdtb_3.csv'
columns = ['relation', 'arg1_arg2', 'sense1', 'multi_sense1']

model_name = 'sentence-transformers/all-MiniLM-L6-v2'
# model_name = 'sentence-transformers/all-mpnet-base-v2'
# model_name = 'sentence-transformers/multi-qa-MiniLM-L6-cos-v1'

short_model_name = re.search(r'[^/]+$', model_name).group()

def generate_ptlm_embeddings(df, model_name):
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input of 'generate_ptlm_embeddings' must be a pandas dataframe.")
    if 'arg1_arg2' not in df.columns:
        raise ValueError("Dataframe passed to 'generate_ptlm_embeddings' must contain 'arg1_arg2' column.")
    
    model = SentenceTransformer(model_name)
    df['embeddings'] = model.encode(df['arg1_arg2']).tolist()

start_time = time.time()
print(f'Generating embeddings with ' + short_model_name + '...')

df = pd.read_csv(path, usecols=columns)

df = df[df['multi_sense1'].isna()]  # remove instances with two senses
df.reset_index(drop=True, inplace=True)  # reset index after removing rows

generate_ptlm_embeddings(df, model_name)

df = df[['relation', 'sense1', 'embeddings']]

df.to_csv('Data/PDTB-3.0/pdtb_3_embeddings_' + short_model_name + '.csv', index=False)

print(f'Completed in {(time.time()-start_time)/60:.2f} minutes.')