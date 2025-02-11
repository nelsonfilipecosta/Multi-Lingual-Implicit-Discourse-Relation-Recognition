import os
import time
import pandas as pd
import regex as re
from sklearn.preprocessing import normalize
from sklearn.model_selection import ShuffleSplit

if not os.path.exists('Data'):
   os.makedirs('Data')
if not os.path.exists('Data/DiscoGeM'):
   os.makedirs('Data/DiscoGeM')

start_time = time.time()
print(f'Preparing DiscoGeM corpus...')

discogem_columns = ['itemid',
                    'arg1_singlesentence',
                    'arg2_singlesentence',
                    'majority_softlabel',
                    'majoritylabel_sampled']

pdtb_columns = ['itemid',
                'arg1',
                'arg2',
                'majority_softlabel']

df_discogem = pd.read_csv('Corpora/DiscoGeM/DiscoGeM_corpus/DiscoGeMcorpus_annotations_wide.csv', usecols = discogem_columns)

df_discogem['arg1_arg2'] = df_discogem['arg1_singlesentence'].copy() + ' ' + df_discogem['arg2_singlesentence'].copy()

df_discogem = df_discogem[['itemid', 'arg1_singlesentence', 'arg2_singlesentence', 'arg1_arg2', 'majoritylabel_sampled', 'majority_softlabel']]

df_discogem.rename(columns={'arg1_singlesentence': 'arg1', 'arg2_singlesentence': 'arg2', 'majoritylabel_sampled': 'majority_level_3'}, inplace=True)

label_names = re.findall(r'[\w\-]+(?=\:)', df_discogem['majority_softlabel'].iloc[0])

for i in label_names:
    df_discogem[i] = ''

for row in range(len(df_discogem['majority_softlabel'])):
    label_values = re.findall(r'(?<=:)(\d\.?(\d*)?)', df_discogem['majority_softlabel'].iloc[row])
    for i in range(len(label_names)):
        df_discogem.loc[row, label_names[i]] = label_values[i][0]

df_discogem.drop(columns=['majority_softlabel'], inplace=True)

#####

df_pdtb = pd.read_csv('Corpora/DiscoGeM/DiscoGeM_corpus/DiscoGeMcorpus_300PDTB_annotations_wide_frances.csv', usecols = pdtb_columns, sep='\\t', engine='python')

df_pdtb['arg1_arg2'] = df_pdtb['arg1'].copy() + ' ' + df_pdtb['arg2'].copy()

df_pdtb['majority_level_3'] = ''

df_pdtb = df_pdtb[['itemid', 'arg1', 'arg2', 'arg1_arg2', 'majority_level_3', 'majority_softlabel']]

for i in label_names:
    df_pdtb[i] = ''

for row in range(len(df_pdtb['majority_softlabel'])):
    filled_label_names = re.findall(r'[\w\-]+(?=\:)', df_pdtb['majority_softlabel'].iloc[row])
    label_values = re.findall(r'(?<=:)(\d\.?(\d*)?)', df_pdtb['majority_softlabel'].iloc[row])
    for i in range(len(filled_label_names)):
        df_pdtb.loc[row, filled_label_names[i]] = label_values[i][0]

df_pdtb.drop(columns=['majority_softlabel'], inplace=True)

pd.set_option('future.no_silent_downcasting', True)
df_pdtb.replace('', 0, inplace=True)

#####

labels_to_exclude = ['arg1-as-negcond', 'arg2-as-negcond', 'disjunction', 'arg1-as-excpt', 'arg2-as-excpt', 'differentcon', 'norel']

df_discogem.drop(columns=labels_to_exclude, inplace=True)
df_pdtb.drop(columns=labels_to_exclude, inplace=True)

df_discogem.rename(columns={'arg2-as-subst': 'substitution'}, inplace=True)
df_pdtb.rename(columns={'arg2-as-subst': 'substitution'}, inplace=True)

for label in labels_to_exclude:
    label_names.remove(label)

label_names.remove('arg2-as-subst')
label_names.append('substitution')

df_discogem[label_names] = df_discogem[label_names].astype(float)
df_pdtb[label_names] = df_pdtb[label_names].astype(float)

df_discogem[label_names] = normalize(df_discogem[label_names], norm='l1', axis=1)
df_pdtb[label_names] = normalize(df_pdtb[label_names], norm='l1', axis=1)

df_discogem['majority_level_3'] = df_discogem[label_names].idxmax(axis=1)
df_pdtb['majority_level_3'] = df_pdtb[label_names].idxmax(axis=1)

label_names_2 = ['synchronous', 'asynchronous', 'cause', 'condition', 'purpose',
                 'concession', 'contrast', 'similarity', 'conjunction', 'instantiation',
                 'level-of-detail', 'equivalence', 'manner', 'substitution']

df_discogem['asynchronous'] = df_discogem['precedence'] + df_discogem['succession']
df_discogem['cause'] = df_discogem['reason'] + df_discogem['result']
df_discogem['condition'] = df_discogem['arg1-as-cond'] + df_discogem['arg2-as-cond']
df_discogem['purpose'] = df_discogem['arg1-as-goal'] + df_discogem['arg2-as-goal']
df_discogem['concession'] = df_discogem['arg1-as-denier'] + df_discogem['arg2-as-denier']
df_discogem['instantiation'] = df_discogem['arg1-as-instance'] + df_discogem['arg2-as-instance']
df_discogem['level-of-detail'] = df_discogem['arg1-as-detail'] + df_discogem['arg2-as-detail']
df_discogem['manner'] = df_discogem['arg1-as-manner'] + df_discogem['arg2-as-manner']

df_pdtb['asynchronous'] = df_pdtb['precedence'] + df_pdtb['succession']
df_pdtb['cause'] = df_pdtb['reason'] + df_pdtb['result']
df_pdtb['condition'] = df_pdtb['arg1-as-cond'] + df_pdtb['arg2-as-cond']
df_pdtb['purpose'] = df_pdtb['arg1-as-goal'] + df_pdtb['arg2-as-goal']
df_pdtb['concession'] = df_pdtb['arg1-as-denier'] + df_pdtb['arg2-as-denier']
df_pdtb['instantiation'] = df_pdtb['arg1-as-instance'] + df_pdtb['arg2-as-instance']
df_pdtb['level-of-detail'] = df_pdtb['arg1-as-detail'] + df_pdtb['arg2-as-detail']
df_pdtb['manner'] = df_pdtb['arg1-as-manner'] + df_pdtb['arg2-as-manner']

df_discogem['majority_level_2'] = df_discogem[label_names_2].idxmax(axis=1)
df_pdtb['majority_level_2'] = df_pdtb[label_names_2].idxmax(axis=1)

label_names_1 = ['temporal', 'contingency', 'comparison', 'expansion']

df_discogem['temporal'] = df_discogem['synchronous'] + df_discogem['asynchronous']
df_discogem['contingency'] = df_discogem['cause'] + df_discogem['condition'] + df_discogem['purpose']
df_discogem['comparison'] = df_discogem['concession'] + df_discogem['contrast'] + df_discogem['similarity']
df_discogem['expansion'] = df_discogem['conjunction'] + df_discogem['instantiation'] + df_discogem['level-of-detail'] + df_discogem['equivalence'] + df_discogem['manner'] + df_discogem['substitution']

df_pdtb['temporal'] = df_pdtb['synchronous'] + df_pdtb['asynchronous']
df_pdtb['contingency'] = df_pdtb['cause'] + df_pdtb['condition'] + df_pdtb['purpose']
df_pdtb['comparison'] = df_pdtb['concession'] + df_pdtb['contrast'] + df_pdtb['similarity']
df_pdtb['expansion'] = df_pdtb['conjunction'] + df_pdtb['instantiation'] + df_pdtb['level-of-detail'] + df_pdtb['equivalence'] + df_pdtb['manner'] + df_pdtb['substitution']

df_discogem['majority_level_1'] = df_discogem[label_names_1].idxmax(axis=1)
df_pdtb['majority_level_1'] = df_pdtb[label_names_1].idxmax(axis=1)

df_discogem['synchronous_2'] = df_discogem['synchronous']
df_discogem['contrast_2'] = df_discogem['contrast']
df_discogem['similarity_2'] = df_discogem['similarity']
df_discogem['conjunction_2'] = df_discogem['conjunction']
df_discogem['equivalence_2'] = df_discogem['equivalence']
df_discogem['substitution_2'] = df_discogem['substitution']
df_pdtb['synchronous_2'] = df_pdtb['synchronous']
df_pdtb['contrast_2'] = df_pdtb['contrast']
df_pdtb['similarity_2'] = df_pdtb['similarity']
df_pdtb['conjunction_2'] = df_pdtb['conjunction']
df_pdtb['equivalence_2'] = df_pdtb['equivalence']
df_pdtb['substitution_2'] = df_pdtb['substitution']

df_discogem['norel'] = 0.0
df_discogem['norel_1'] = 0.0
df_discogem['norel_2'] = 0.0
df_pdtb['norel'] = 0.0
df_pdtb['norel_1'] = 0.0
df_pdtb['norel_2'] = 0.0

df_discogem = df_discogem[['itemid', 'arg1', 'arg2', 'arg1_arg2', 'majority_level_3', 'synchronous',
                           'precedence', 'succession', 'reason', 'result', 'arg1-as-cond', 'arg2-as-cond',
                           'arg1-as-goal', 'arg2-as-goal', 'arg1-as-denier', 'arg2-as-denier', 'contrast',
                           'similarity', 'conjunction', 'equivalence', 'arg1-as-instance', 'arg2-as-instance',
                           'arg1-as-detail', 'arg2-as-detail', 'arg1-as-manner', 'arg2-as-manner', 'substitution',
                           'norel', 'majority_level_2', 'synchronous_2', 'asynchronous', 'cause', 'condition',
                           'purpose', 'concession', 'contrast_2', 'similarity_2', 'conjunction_2', 'equivalence_2',
                           'instantiation', 'level-of-detail', 'manner', 'substitution_2', 'norel_2', 'majority_level_1',
                           'temporal', 'contingency', 'comparison', 'expansion', 'norel_1']]

df_pdtb = df_pdtb[['itemid', 'arg1', 'arg2', 'arg1_arg2', 'majority_level_3', 'synchronous',
                   'precedence', 'succession', 'reason', 'result', 'arg1-as-cond', 'arg2-as-cond',
                   'arg1-as-goal', 'arg2-as-goal', 'arg1-as-denier', 'arg2-as-denier', 'contrast',
                   'similarity', 'conjunction', 'equivalence', 'arg1-as-instance', 'arg2-as-instance',
                   'arg1-as-detail', 'arg2-as-detail', 'arg1-as-manner', 'arg2-as-manner', 'substitution',
                   'norel', 'majority_level_2', 'synchronous_2', 'asynchronous', 'cause', 'condition',
                   'purpose', 'concession', 'contrast_2', 'similarity_2', 'conjunction_2', 'equivalence_2',
                   'instantiation', 'level-of-detail', 'manner', 'substitution_2', 'norel_2', 'majority_level_1',
                   'temporal', 'contingency', 'comparison', 'expansion', 'norel_1']]

#####

random_rows = df_discogem.sample(n=600, random_state=17)

df_norel = pd.DataFrame({
    'itemid': random_rows['itemid'].iloc[0:300].values + '+' + random_rows['itemid'].iloc[300:600].values,
    'arg1': random_rows['arg1'].iloc[0:300].values,
    'arg2': random_rows['arg2'].iloc[300:600].values,
    'arg1_arg2': random_rows['arg1'].iloc[0:300].values + ' ' + random_rows['arg2'].iloc[300:600].values,
    'majority_level_3': 'norel',
    'synchronous': 0.0,
    'precedence': 0.0,
    'succession': 0.0,
    'reason': 0.0,
    'result': 0.0,
    'arg1-as-cond': 0.0,
    'arg2-as-cond': 0.0,
    'arg1-as-goal': 0.0,
    'arg2-as-goal': 0.0,
    'arg1-as-denier': 0.0,
    'arg2-as-denier': 0.0,
    'contrast': 0.0,
    'similarity': 0.0,
    'conjunction': 0.0,
    'equivalence': 0.0,
    'arg1-as-instance': 0.0,
    'arg2-as-instance': 0.0,
    'arg1-as-detail': 0.0,
    'arg2-as-detail': 0.0,
    'arg1-as-manner': 0.0,
    'arg2-as-manner': 0.0,
    'substitution': 0.0,
    'norel': 1.0,
    'majority_level_2': 'norel',
    'synchronous_2': 0.0,
    'asynchronous': 0.0,
    'cause': 0.0,
    'condition': 0.0,
    'purpose': 0.0,
    'concession': 0.0,
    'contrast_2': 0.0,
    'similarity_2': 0.0,
    'conjunction_2': 0.0,
    'equivalence_2': 0.0,
    'instantiation': 0.0,
    'level-of-detail': 0.0,
    'manner': 0.0,
    'substitution_2': 0.0,
    'norel_2': 1.0,
    'majority_level_1': 'norel',
    'temporal': 0.0,
    'contingency': 0.0,
    'comparison': 0.0,
    'expansion': 0.0,
    'norel_1': 1.0
})

#####

# uncomment for 'norel' data
# df_discogem = pd.concat([df_discogem, df_norel], ignore_index=True)

gs_test = ShuffleSplit(n_splits=1, test_size=0.21, random_state=17)
train_idx, test_idx = next(gs_test.split(df_discogem, df_discogem['majority_level_3']))
df_temp = df_discogem.iloc[train_idx]
df_test = df_discogem.iloc[test_idx]

gs_validation = ShuffleSplit(n_splits=1, test_size=0.125, random_state=16)
train_idx_discogem, validation_idx_discogem = next(gs_validation.split(df_temp, df_temp['majority_level_3']))
train_idx_pdtb, validation_idx_pdtb = next(gs_validation.split(df_pdtb, df_pdtb['majority_level_3']))
df_train = pd.concat([df_temp.iloc[train_idx_discogem], df_pdtb.iloc[train_idx_pdtb]], ignore_index=True)
df_validation = pd.concat([df_temp.iloc[validation_idx_discogem], df_pdtb.iloc[validation_idx_pdtb]], ignore_index=True)

df_discogem.to_csv('Data/DiscoGeM/discogem.csv', index=False)
df_pdtb.to_csv('Data/DiscoGeM/discogem_pdtb.csv', index=False)
df_train.to_csv('Data/DiscoGeM/discogem_train.csv', index=False)
df_validation.to_csv('Data/DiscoGeM/discogem_validation.csv', index=False)
df_test.to_csv('Data/DiscoGeM/discogem_test.csv', index=False)

print(f'Completed in {(time.time()-start_time)/60:.2f} minutes.')