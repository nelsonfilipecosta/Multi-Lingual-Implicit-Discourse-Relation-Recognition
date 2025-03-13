import re
import time
import pandas as pd
from sentence_transformers import SentenceTransformer

path = 'Data/PDTB-3.0/pdtb_3.csv'
columns = ['arg1_arg2', 'sense1', 'multi_sense1']

model_name = 'sentence-transformers/all-MiniLM-L6-v2'
short_model_name = re.search(r'[^/]+$', model_name).group()

start_time = time.time()
print(f'Generating embeddings with ' + short_model_name + '...')

df = pd.read_csv(path, usecols=columns)

model = SentenceTransformer(model_name)

df['embeddings'] = df['arg1_arg2'].apply(lambda x: model.encode(x).tolist())

df = df[['sense1', 'multi_sense1', 'arg1_arg2', 'embeddings']]

df.to_csv('Data/PDTB-3.0/pdtb_3_embeddings_' + short_model_name + '.csv', index=False)

print(f'Completed in {(time.time()-start_time)/60:.2f} minutes.')
