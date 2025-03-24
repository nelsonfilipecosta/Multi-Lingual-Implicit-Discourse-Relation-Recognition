import os
import time
import numpy as np
import pandas as pd
import regex as re

if not os.path.exists('Data'):
   os.makedirs('Data')
if not os.path.exists('Data/DiscoGeM-2.0'):
   os.makedirs('Data/DiscoGeM-2.0')

start_time = time.time()
print(f'Preparing DiscoGeM corpus...')

discogem_columns = ['split',
                    'itemid',
                    'orig_lang',
                    'available_langs',
                    'arg1_context_en',
                    'arg2_context_en',
                    'MV_en',
                    'MV_dist_en',
                    'arg1_context_de',
                    'arg2_context_de',
                    'MV_de',
                    'MV_dist_de',
                    'arg1_context_fr',
                    'arg2_context_fr',
                    'MV_fr',
                    'MV_dist_fr',
                    'arg1_context_cs',
                    'arg2_context_cs',
                    'MV_cs',
                    'MV_dist_cs']

df_discogem = pd.read_csv('Corpora/DiscoGeM-2.0/DiscoGeM-2.0_corpus/DiscoGeM-2.0_items.csv', usecols=discogem_columns, delimiter='\t', dtype=str)

df_discogem['arg1_arg2_en'] = df_discogem['arg1_context_en'].copy() + ' ' + df_discogem['arg2_context_en'].copy()
df_discogem['arg1_arg2_de'] = df_discogem['arg1_context_de'].copy() + ' ' + df_discogem['arg2_context_de'].copy()
df_discogem['arg1_arg2_fr'] = df_discogem['arg1_context_fr'].copy() + ' ' + df_discogem['arg2_context_fr'].copy()
df_discogem['arg1_arg2_cs'] = df_discogem['arg1_context_cs'].copy() + ' ' + df_discogem['arg2_context_cs'].copy()

df_discogem = df_discogem[['split',
                           'itemid',
                           'orig_lang',
                           'available_langs',
                           'arg1_context_en',
                           'arg2_context_en',
                           'arg1_arg2_en',
                           'MV_en',
                           'MV_dist_en',
                           'arg1_context_de',
                           'arg2_context_de',
                           'arg1_arg2_de',
                           'MV_de',
                           'MV_dist_de',
                           'arg1_context_fr',
                           'arg2_context_fr',
                           'arg1_arg2_fr',
                           'MV_fr',
                           'MV_dist_fr',
                           'arg1_context_cs',
                           'arg2_context_cs',
                           'arg1_arg2_cs',
                           'MV_cs',
                           'MV_dist_cs']]

df_discogem.rename(columns={'arg1_context_en': 'arg1_en',
                            'arg2_context_en': 'arg2_en',
                            'MV_en': 'majority_level_3_en',
                            'arg1_context_de': 'arg1_de',
                            'arg2_context_de': 'arg2_de',
                            'MV_de': 'majority_level_3_de',
                            'arg1_context_fr': 'arg1_fr',
                            'arg2_context_fr': 'arg2_fr',
                            'MV_fr': 'majority_level_3_fr',
                            'arg1_context_cs': 'arg1_cs',
                            'arg2_context_cs': 'arg2_cs',
                            'MV_cs': 'majority_level_3_cs'},
                   inplace=True)

df_discogem['available_langs'] = df_discogem['available_langs'].str.replace("[", "").str.replace("]", "").str.replace("'", "")
df_discogem['MV_dist_en'] = df_discogem['MV_dist_en'].str.replace("{", "").str.replace("}", "").str.replace(",", ";").str.replace("'", "").str.replace(": ", ":")
df_discogem['MV_dist_de'] = df_discogem['MV_dist_de'].str.replace("{", "").str.replace("}", "").str.replace(",", ";").str.replace("'", "").str.replace(": ", ":")
df_discogem['MV_dist_fr'] = df_discogem['MV_dist_fr'].str.replace("{", "").str.replace("}", "").str.replace(",", ";").str.replace("'", "").str.replace(": ", ":")
df_discogem['MV_dist_cs'] = df_discogem['MV_dist_cs'].str.replace("{", "").str.replace("}", "").str.replace(",", ";").str.replace("'", "").str.replace(": ", ":")

label_names = re.findall(r'[\w\-]+(?=\:)', df_discogem['MV_dist_en'].iloc[0])

lang_iter = 0
for lang in ['_en', '_de', '_fr', '_cs']:
    for i in label_names:
        df_discogem.insert(9+lang_iter, i+lang, '')
        lang_iter += 1
    lang_iter += 5

df_discogem = df_discogem.copy()  # defragment dataframe to reorganize memory

for lang in ['_en', '_de', '_fr', '_cs']:
    for row in range(len(df_discogem['MV_dist'+lang])):
        
        cell_value = str(df_discogem['MV_dist'+lang].iloc[row]) # convert to string to avoid NaN values
        
        if cell_value:
            label_values = re.findall(r'(?<=:)(\d\.?(\d*)?)', cell_value)
        else:
            label_values = [] # skip regex processing if there are no values
        
        for i in range(len(label_names)):
            if i < len(label_values): # if there are values to assign
                df_discogem.loc[row, label_names[i]+lang] = label_values[i][0]

df_discogem.drop(columns=['MV_dist_en'], inplace=True)
df_discogem.drop(columns=['MV_dist_de'], inplace=True)
df_discogem.drop(columns=['MV_dist_fr'], inplace=True)
df_discogem.drop(columns=['MV_dist_cs'], inplace=True)

for label in label_names:
    for lang in ['_en', '_de', '_fr', '_cs']:
        df_discogem[label+lang] = df_discogem[label+lang].replace('', np.nan).astype(float)

label_names_2 = ['synchronous', 'asynchronous', 'cause', 'condition', 'neg-condition', 'purpose',
                 'concession', 'contrast', 'similarity', 'conjunction', 'disjunction', 'equivalence',
                 'exception', 'instantiation', 'level-of-detail', 'manner', 'substitution']

label_names_1 = ['temporal', 'contingency', 'comparison', 'expansion']

for lang in ['_en', '_de', '_fr', '_cs']:
    # complete level-2 senses
    df_discogem['asynchronous'+lang]    = df_discogem['precedence'+lang]       + df_discogem['succession'+lang]
    df_discogem['cause'+lang]           = df_discogem['reason'+lang]           + df_discogem['result'+lang]
    df_discogem['condition'+lang]       = df_discogem['arg1-as-cond'+lang]     + df_discogem['arg2-as-cond'+lang]
    df_discogem['neg-condition'+lang]   = df_discogem['arg1-as-negcond'+lang]  + df_discogem['arg2-as-negcond'+lang]
    df_discogem['purpose'+lang]         = df_discogem['arg1-as-goal'+lang]     + df_discogem['arg2-as-goal'+lang]
    df_discogem['concession'+lang]      = df_discogem['arg1-as-denier'+lang]   + df_discogem['arg2-as-denier'+lang]
    df_discogem['exception'+lang]       = df_discogem['arg1-as-excpt'+lang]    + df_discogem['arg2-as-excpt'+lang]
    df_discogem['instantiation'+lang]   = df_discogem['arg1-as-instance'+lang] + df_discogem['arg2-as-instance'+lang]
    df_discogem['level-of-detail'+lang] = df_discogem['arg1-as-detail'+lang]   + df_discogem['arg2-as-detail'+lang]
    df_discogem['manner'+lang]          = df_discogem['arg1-as-manner'+lang]   + df_discogem['arg2-as-manner'+lang]
    df_discogem['substitution'+lang]    = df_discogem['arg1-as-subst'+lang]    + df_discogem['arg2-as-subst'+lang]
    # get majority level-2 sense
    df_discogem['majority_level_2'+lang] = df_discogem[[label+lang for label in label_names_2]].idxmax(axis=1)
    # complete level-1 senses
    df_discogem['temporal'+lang]    =  df_discogem['synchronous'+lang]   + df_discogem['asynchronous'+lang]
    df_discogem['contingency'+lang] = (df_discogem['cause'+lang]         + df_discogem['condition'+lang]     + df_discogem['neg-condition'+lang]   +
                                       df_discogem['purpose'+lang])
    df_discogem['comparison'+lang]  =  df_discogem['concession'+lang]    + df_discogem['contrast'+lang]      + df_discogem['similarity'+lang]
    df_discogem['expansion'+lang]   = (df_discogem['conjunction'+lang]   + df_discogem['disjunction'+lang]   + df_discogem['equivalence'+lang]     +
                                       df_discogem['exception'+lang]     + df_discogem['instantiation'+lang] + df_discogem['level-of-detail'+lang] +
                                       df_discogem['manner'+lang]        + df_discogem['substitution'+lang])
    # get majority level-1 sense
    df_discogem['majority_level_1'+lang] = df_discogem[[label+lang for label in label_names_1]].idxmax(axis=1)
    # duplicate level-2 senses without level-3 senses
    df_discogem['synchronous_3'+lang] = df_discogem['synchronous'+lang]
    df_discogem['contrast_3'+lang]    = df_discogem['contrast'+lang]
    df_discogem['similarity_3'+lang]  = df_discogem['similarity'+lang]
    df_discogem['conjunction_3'+lang] = df_discogem['conjunction'+lang]
    df_discogem['disjunction_3'+lang] = df_discogem['disjunction'+lang]
    df_discogem['equivalence_3'+lang] = df_discogem['equivalence'+lang]

# reorganize columns
df_discogem = df_discogem[['split', 'itemid', 'orig_lang', 'available_langs', 'arg1_en', 'arg2_en', 'arg1_arg2_en', 'majority_level_1_en',
                           'temporal_en', 'contingency_en', 'comparison_en', 'expansion_en', 'majority_level_2_en', 'synchronous_en', 'asynchronous_en',
                           'cause_en', 'condition_en', 'neg-condition_en', 'purpose_en', 'concession_en', 'contrast_en', 'similarity_en', 'conjunction_en',
                           'disjunction_en', 'equivalence_en', 'exception_en', 'instantiation_en', 'level-of-detail_en', 'manner_en', 'substitution_en',
                           'majority_level_3_en', 'synchronous_3_en', 'precedence_en', 'succession_en', 'reason_en', 'result_en', 'arg1-as-cond_en', 'arg2-as-cond_en',
                           'arg1-as-negcond_en', 'arg2-as-negcond_en', 'arg1-as-goal_en', 'arg2-as-goal_en', 'arg1-as-denier_en', 'arg2-as-denier_en', 'contrast_3_en',
                           'similarity_3_en', 'conjunction_3_en', 'disjunction_3_en', 'equivalence_3_en', 'arg1-as-excpt_en', 'arg2-as-excpt_en', 'arg1-as-instance_en',
                           'arg2-as-instance_en', 'arg1-as-detail_en', 'arg2-as-detail_en', 'arg1-as-manner_en', 'arg2-as-manner_en', 'arg1-as-subst_en',
                           'arg2-as-subst_en', 'norel_en', 'arg1_de', 'arg2_de', 'arg1_arg2_de', 'majority_level_1_de', 'temporal_de', 'contingency_de', 'comparison_de',
                           'expansion_de', 'majority_level_2_de', 'synchronous_de', 'asynchronous_de', 'cause_de', 'condition_de', 'neg-condition_de', 'purpose_de',
                           'concession_de', 'contrast_de', 'similarity_de', 'conjunction_de', 'disjunction_de', 'equivalence_de', 'exception_de', 'instantiation_de',
                           'level-of-detail_de', 'manner_de', 'substitution_de', 'majority_level_3_de', 'synchronous_3_de', 'precedence_de', 'succession_de', 'reason_de',
                           'result_de', 'arg1-as-cond_de', 'arg2-as-cond_de', 'arg1-as-negcond_de', 'arg2-as-negcond_de', 'arg1-as-goal_de', 'arg2-as-goal_de',
                           'arg1-as-denier_de', 'arg2-as-denier_de', 'contrast_3_de', 'similarity_3_de', 'conjunction_3_de', 'disjunction_3_de', 'equivalence_3_de',
                           'arg1-as-excpt_de', 'arg2-as-excpt_de', 'arg1-as-instance_de', 'arg2-as-instance_de', 'arg1-as-detail_de', 'arg2-as-detail_de',
                           'arg1-as-manner_de', 'arg2-as-manner_de', 'arg1-as-subst_de', 'arg2-as-subst_de', 'norel_de', 'arg1_fr', 'arg2_fr', 'arg1_arg2_fr',
                           'majority_level_1_fr', 'temporal_fr', 'contingency_fr', 'comparison_fr', 'expansion_fr', 'majority_level_2_fr', 'synchronous_fr',
                           'asynchronous_fr', 'cause_fr', 'condition_fr', 'neg-condition_fr', 'purpose_fr', 'concession_fr', 'contrast_fr', 'similarity_fr',
                           'conjunction_fr', 'disjunction_fr', 'equivalence_fr', 'exception_fr', 'instantiation_fr', 'level-of-detail_fr', 'manner_fr', 'substitution_fr',
                           'majority_level_3_fr', 'synchronous_3_fr', 'precedence_fr', 'succession_fr', 'reason_fr', 'result_fr', 'arg1-as-cond_fr', 'arg2-as-cond_fr',
                           'arg1-as-negcond_fr', 'arg2-as-negcond_fr', 'arg1-as-goal_fr', 'arg2-as-goal_fr', 'arg1-as-denier_fr', 'arg2-as-denier_fr', 'contrast_3_fr',
                           'similarity_3_fr', 'conjunction_3_fr', 'disjunction_3_fr', 'equivalence_3_fr', 'arg1-as-excpt_fr', 'arg2-as-excpt_fr', 'arg1-as-instance_fr',
                           'arg2-as-instance_fr', 'arg1-as-detail_fr', 'arg2-as-detail_fr', 'arg1-as-manner_fr', 'arg2-as-manner_fr', 'arg1-as-subst_fr', 'arg2-as-subst_fr',
                           'norel_fr', 'arg1_cs', 'arg2_cs', 'arg1_arg2_cs', 'majority_level_1_cs', 'temporal_cs', 'contingency_cs', 'comparison_cs', 'expansion_cs',
                           'majority_level_2_cs', 'synchronous_cs', 'asynchronous_cs', 'cause_cs', 'condition_cs', 'neg-condition_cs', 'purpose_cs', 'concession_cs',
                           'contrast_cs', 'similarity_cs', 'conjunction_cs', 'disjunction_cs', 'equivalence_cs', 'exception_cs', 'instantiation_cs', 'level-of-detail_cs',
                           'manner_cs', 'substitution_cs', 'majority_level_3_cs', 'synchronous_3_cs', 'precedence_cs', 'succession_cs', 'reason_cs', 'result_cs',
                           'arg1-as-cond_cs', 'arg2-as-cond_cs', 'arg1-as-negcond_cs', 'arg2-as-negcond_cs', 'arg1-as-goal_cs', 'arg2-as-goal_cs', 'arg1-as-denier_cs',
                           'arg2-as-denier_cs', 'contrast_3_cs', 'similarity_3_cs', 'conjunction_3_cs', 'disjunction_3_cs', 'equivalence_3_cs', 'arg1-as-excpt_cs',
                           'arg2-as-excpt_cs', 'arg1-as-instance_cs', 'arg2-as-instance_cs', 'arg1-as-detail_cs', 'arg2-as-detail_cs', 'arg1-as-manner_cs',
                           'arg2-as-manner_cs', 'arg1-as-subst_cs', 'arg2-as-subst_cs', 'norel_cs']]

df_discogem_train = df_discogem[df_discogem['split'] == 'train']
df_discogem_validation = df_discogem[df_discogem['split'] == 'dev']
df_discogem_test = df_discogem[df_discogem['split'] == 'test']

df_discogem.to_csv('Data/DiscoGeM-2.0/discogem_2.csv', index=False)
df_discogem_train.to_csv('Data/DiscoGeM-2.0/discogem_2_train.csv', index=False)
df_discogem_validation.to_csv('Data/DiscoGeM-2.0/discogem_2_validation.csv', index=False)
df_discogem_test.to_csv('Data/DiscoGeM-2.0/discogem_2_test.csv', index=False)

en_columns = ['split', 'itemid', 'orig_lang', 'available_langs', 'arg1_arg2_en', 'majority_level_1_en', 'temporal_en', 'contingency_en',
              'comparison_en', 'expansion_en', 'majority_level_2_en', 'synchronous_en', 'asynchronous_en', 'cause_en', 'condition_en',
              'neg-condition_en', 'purpose_en', 'concession_en', 'contrast_en', 'similarity_en', 'conjunction_en', 'disjunction_en',
              'equivalence_en', 'exception_en', 'instantiation_en', 'level-of-detail_en', 'manner_en', 'substitution_en', 'majority_level_3_en',
              'synchronous_3_en', 'precedence_en', 'succession_en', 'reason_en', 'result_en', 'arg1-as-cond_en', 'arg2-as-cond_en',
              'arg1-as-negcond_en', 'arg2-as-negcond_en', 'arg1-as-goal_en', 'arg2-as-goal_en', 'arg1-as-denier_en', 'arg2-as-denier_en',
              'contrast_3_en', 'similarity_3_en', 'conjunction_3_en', 'disjunction_3_en', 'equivalence_3_en', 'arg1-as-excpt_en', 'arg2-as-excpt_en',
              'arg1-as-instance_en', 'arg2-as-instance_en', 'arg1-as-detail_en', 'arg2-as-detail_en', 'arg1-as-manner_en', 'arg2-as-manner_en',
              'arg1-as-subst_en', 'arg2-as-subst_en', 'norel_en']

de_columns = ['split', 'itemid', 'orig_lang', 'available_langs', 'arg1_arg2_de', 'majority_level_1_de', 'temporal_de', 'contingency_de',
              'comparison_de', 'expansion_de', 'majority_level_2_de', 'synchronous_de', 'asynchronous_de', 'cause_de', 'condition_de',
              'neg-condition_de', 'purpose_de', 'concession_de', 'contrast_de', 'similarity_de', 'conjunction_de', 'disjunction_de',
              'equivalence_de', 'exception_de', 'instantiation_de', 'level-of-detail_de', 'manner_de', 'substitution_de', 'majority_level_3_de',
              'synchronous_3_de', 'precedence_de', 'succession_de', 'reason_de', 'result_de', 'arg1-as-cond_de', 'arg2-as-cond_de',
              'arg1-as-negcond_de', 'arg2-as-negcond_de', 'arg1-as-goal_de', 'arg2-as-goal_de', 'arg1-as-denier_de', 'arg2-as-denier_de',
              'contrast_3_de', 'similarity_3_de', 'conjunction_3_de', 'disjunction_3_de', 'equivalence_3_de', 'arg1-as-excpt_de', 'arg2-as-excpt_de',
              'arg1-as-instance_de', 'arg2-as-instance_de', 'arg1-as-detail_de', 'arg2-as-detail_de', 'arg1-as-manner_de', 'arg2-as-manner_de',
              'arg1-as-subst_de', 'arg2-as-subst_de', 'norel_de']

fr_columns = ['split', 'itemid', 'orig_lang', 'available_langs', 'arg1_arg2_fr', 'majority_level_1_fr', 'temporal_fr', 'contingency_fr',
              'comparison_fr', 'expansion_fr', 'majority_level_2_fr', 'synchronous_fr', 'asynchronous_fr', 'cause_fr', 'condition_fr',
              'neg-condition_fr', 'purpose_fr', 'concession_fr', 'contrast_fr', 'similarity_fr', 'conjunction_fr', 'disjunction_fr',
              'equivalence_fr', 'exception_fr', 'instantiation_fr', 'level-of-detail_fr', 'manner_fr', 'substitution_fr', 'majority_level_3_fr',
              'synchronous_3_fr', 'precedence_fr', 'succession_fr', 'reason_fr', 'result_fr', 'arg1-as-cond_fr', 'arg2-as-cond_fr',
              'arg1-as-negcond_fr', 'arg2-as-negcond_fr', 'arg1-as-goal_fr', 'arg2-as-goal_fr', 'arg1-as-denier_fr', 'arg2-as-denier_fr',
              'contrast_3_fr', 'similarity_3_fr', 'conjunction_3_fr', 'disjunction_3_fr', 'equivalence_3_fr', 'arg1-as-excpt_fr', 'arg2-as-excpt_fr',
              'arg1-as-instance_fr', 'arg2-as-instance_fr', 'arg1-as-detail_fr', 'arg2-as-detail_fr', 'arg1-as-manner_fr', 'arg2-as-manner_fr',
              'arg1-as-subst_fr', 'arg2-as-subst_fr', 'norel_fr']

cs_columns = ['split', 'itemid', 'orig_lang', 'available_langs', 'arg1_arg2_cs', 'majority_level_1_cs', 'temporal_cs', 'contingency_cs',
              'comparison_cs', 'expansion_cs', 'majority_level_2_cs', 'synchronous_cs', 'asynchronous_cs', 'cause_cs', 'condition_cs',
              'neg-condition_cs', 'purpose_cs', 'concession_cs', 'contrast_cs', 'similarity_cs', 'conjunction_cs', 'disjunction_cs',
              'equivalence_cs', 'exception_cs', 'instantiation_cs', 'level-of-detail_cs', 'manner_cs', 'substitution_cs', 'majority_level_3_cs',
              'synchronous_3_cs', 'precedence_cs', 'succession_cs', 'reason_cs', 'result_cs', 'arg1-as-cond_cs', 'arg2-as-cond_cs',
              'arg1-as-negcond_cs', 'arg2-as-negcond_cs', 'arg1-as-goal_cs', 'arg2-as-goal_cs', 'arg1-as-denier_cs', 'arg2-as-denier_cs',
              'contrast_3_cs', 'similarity_3_cs', 'conjunction_3_cs', 'disjunction_3_cs', 'equivalence_3_cs', 'arg1-as-excpt_cs', 'arg2-as-excpt_cs',
              'arg1-as-instance_cs', 'arg2-as-instance_cs', 'arg1-as-detail_cs', 'arg2-as-detail_cs', 'arg1-as-manner_cs', 'arg2-as-manner_cs',
              'arg1-as-subst_cs', 'arg2-as-subst_cs', 'norel_cs']

df_discogem_single_lang_en = df_discogem[en_columns].copy()
df_discogem_single_lang_de = df_discogem[de_columns].copy()
df_discogem_single_lang_fr = df_discogem[fr_columns].copy()
df_discogem_single_lang_cs = df_discogem[cs_columns].copy()

df_discogem_single_lang_de.dropna(subset=['majority_level_1_de'], inplace=True)
df_discogem_single_lang_fr.dropna(subset=['majority_level_1_fr'], inplace=True)
df_discogem_single_lang_cs.dropna(subset=['majority_level_1_cs'], inplace=True)

df_discogem_single_lang_en.insert(4, 'inst_lang', 'en')
df_discogem_single_lang_de.insert(4, 'inst_lang', 'de')
df_discogem_single_lang_fr.insert(4, 'inst_lang', 'fr')
df_discogem_single_lang_cs.insert(4, 'inst_lang', 'cs')

single_lang_columns = ['split', 'itemid', 'orig_lang', 'available_langs', 'inst_lang', 'arg1_arg2', 'majority_level_1',
                       'temporal', 'contingency', 'comparison', 'expansion', 'majority_level_2', 'synchronous', 'asynchronous',
                       'cause', 'condition', 'neg-condition', 'purpose', 'concession', 'contrast', 'similarity', 'conjunction',
                       'disjunction', 'equivalence', 'exception', 'instantiation', 'level-of-detail', 'manner', 'substitution',
                       'majority_level_3', 'synchronous_3', 'precedence', 'succession', 'reason', 'result', 'arg1-as-cond', 'arg2-as-cond',
                       'arg1-as-negcond', 'arg2-as-negcond', 'arg1-as-goal', 'arg2-as-goal', 'arg1-as-denier', 'arg2-as-denier', 'contrast_3',
                       'similarity_3', 'conjunction_3', 'disjunction_3', 'equivalence_3', 'arg1-as-excpt', 'arg2-as-excpt', 'arg1-as-instance',
                       'arg2-as-instance', 'arg1-as-detail', 'arg2-as-detail', 'arg1-as-manner', 'arg2-as-manner', 'arg1-as-subst',
                       'arg2-as-subst', 'norel']

df_discogem_single_lang_en.columns = single_lang_columns
df_discogem_single_lang_de.columns = single_lang_columns
df_discogem_single_lang_fr.columns = single_lang_columns
df_discogem_single_lang_cs.columns = single_lang_columns

# english

df_discogem_single_lang_en['majority_level_1'] = df_discogem_single_lang_en['majority_level_1'].str.replace(r'_en$', '', regex=True)
df_discogem_single_lang_en['majority_level_2'] = df_discogem_single_lang_en['majority_level_2'].str.replace(r'_en$', '', regex=True)
df_discogem_single_lang_en['majority_level_3'] = df_discogem_single_lang_en['majority_level_3'].str.replace(r'_en$', '', regex=True)

df_discogem_single_lang_en_train = df_discogem_single_lang_en[df_discogem_single_lang_en['split'] == 'train']
df_discogem_single_lang_en_validation = df_discogem_single_lang_en[df_discogem_single_lang_en['split'] == 'dev']
df_discogem_single_lang_en_test = df_discogem_single_lang_en[df_discogem_single_lang_en['split'] == 'test']

df_discogem_single_lang_en.to_csv('Data/DiscoGeM-2.0/df_discogem_single_lang_en.csv', index=False)
df_discogem_single_lang_en_train.to_csv('Data/DiscoGeM-2.0/df_discogem_single_lang_en_train.csv', index=False)
df_discogem_single_lang_en_validation.to_csv('Data/DiscoGeM-2.0/df_discogem_single_lang_en_validation.csv', index=False)
df_discogem_single_lang_en_test.to_csv('Data/DiscoGeM-2.0/df_discogem_single_lang_en_test.csv', index=False)

# german

df_discogem_single_lang_de['majority_level_1'] = df_discogem_single_lang_de['majority_level_1'].str.replace(r'_de$', '', regex=True)
df_discogem_single_lang_de['majority_level_2'] = df_discogem_single_lang_de['majority_level_2'].str.replace(r'_de$', '', regex=True)
df_discogem_single_lang_de['majority_level_3'] = df_discogem_single_lang_de['majority_level_3'].str.replace(r'_de$', '', regex=True)

df_discogem_single_lang_de_train = df_discogem_single_lang_de[df_discogem_single_lang_de['split'] == 'train']
df_discogem_single_lang_de_validation = df_discogem_single_lang_de[df_discogem_single_lang_de['split'] == 'dev']
df_discogem_single_lang_de_test = df_discogem_single_lang_de[df_discogem_single_lang_de['split'] == 'test']

df_discogem_single_lang_de.to_csv('Data/DiscoGeM-2.0/df_discogem_single_lang_de.csv', index=False)
df_discogem_single_lang_de_train.to_csv('Data/DiscoGeM-2.0/df_discogem_single_lang_de_train.csv', index=False)
df_discogem_single_lang_de_validation.to_csv('Data/DiscoGeM-2.0/df_discogem_single_lang_de_validation.csv', index=False)
df_discogem_single_lang_de_test.to_csv('Data/DiscoGeM-2.0/df_discogem_single_lang_de_test.csv', index=False)

# french

df_discogem_single_lang_fr['majority_level_1'] = df_discogem_single_lang_fr['majority_level_1'].str.replace(r'_fr$', '', regex=True)
df_discogem_single_lang_fr['majority_level_2'] = df_discogem_single_lang_fr['majority_level_2'].str.replace(r'_fr$', '', regex=True)
df_discogem_single_lang_fr['majority_level_3'] = df_discogem_single_lang_fr['majority_level_3'].str.replace(r'_fr$', '', regex=True)

df_discogem_single_lang_fr_train = df_discogem_single_lang_fr[df_discogem_single_lang_fr['split'] == 'train']
df_discogem_single_lang_fr_validation = df_discogem_single_lang_fr[df_discogem_single_lang_fr['split'] == 'frv']
df_discogem_single_lang_fr_test = df_discogem_single_lang_fr[df_discogem_single_lang_fr['split'] == 'test']

df_discogem_single_lang_fr.to_csv('Data/DiscoGeM-2.0/df_discogem_single_lang_fr.csv', index=False)
df_discogem_single_lang_fr_train.to_csv('Data/DiscoGeM-2.0/df_discogem_single_lang_fr_train.csv', index=False)
df_discogem_single_lang_fr_validation.to_csv('Data/DiscoGeM-2.0/df_discogem_single_lang_fr_validation.csv', index=False)
df_discogem_single_lang_fr_test.to_csv('Data/DiscoGeM-2.0/df_discogem_single_lang_fr_test.csv', index=False)

# czech

df_discogem_single_lang_cs['majority_level_1'] = df_discogem_single_lang_cs['majority_level_1'].str.replace(r'_cs$', '', regex=True)
df_discogem_single_lang_cs['majority_level_2'] = df_discogem_single_lang_cs['majority_level_2'].str.replace(r'_cs$', '', regex=True)
df_discogem_single_lang_cs['majority_level_3'] = df_discogem_single_lang_cs['majority_level_3'].str.replace(r'_cs$', '', regex=True)

df_discogem_single_lang_cs_train = df_discogem_single_lang_cs[df_discogem_single_lang_cs['split'] == 'train']
df_discogem_single_lang_cs_validation = df_discogem_single_lang_cs[df_discogem_single_lang_cs['split'] == 'dev']
df_discogem_single_lang_cs_test = df_discogem_single_lang_cs[df_discogem_single_lang_cs['split'] == 'test']

df_discogem_single_lang_cs.to_csv('Data/DiscoGeM-2.0/df_discogem_single_lang_cs.csv', index=False)
df_discogem_single_lang_cs_train.to_csv('Data/DiscoGeM-2.0/df_discogem_single_lang_cs_train.csv', index=False)
df_discogem_single_lang_cs_validation.to_csv('Data/DiscoGeM-2.0/df_discogem_single_lang_cs_validation.csv', index=False)
df_discogem_single_lang_cs_test.to_csv('Data/DiscoGeM-2.0/df_discogem_single_lang_cs_test.csv', index=False)

# all languages

df_discogem_single_lang_all = pd.concat([df_discogem_single_lang_en, df_discogem_single_lang_de,
                                         df_discogem_single_lang_fr, df_discogem_single_lang_cs])

df_discogem_single_lang_all_train = df_discogem_single_lang_all[df_discogem_single_lang_all['split'] == 'train']
df_discogem_single_lang_all_validation = df_discogem_single_lang_all[df_discogem_single_lang_all['split'] == 'dev']
df_discogem_single_lang_all_test = df_discogem_single_lang_all[df_discogem_single_lang_all['split'] == 'test']

df_discogem_single_lang_all.to_csv('Data/DiscoGeM-2.0/df_discogem_single_lang_all.csv', index=False)
df_discogem_single_lang_all_train.to_csv('Data/DiscoGeM-2.0/df_discogem_single_lang_all_train.csv', index=False)
df_discogem_single_lang_all_validation.to_csv('Data/DiscoGeM-2.0/df_discogem_single_lang_all_validation.csv', index=False)
df_discogem_single_lang_all_test.to_csv('Data/DiscoGeM-2.0/df_discogem_single_lang_all_test.csv', index=False)

print(f'Completed in {(time.time()-start_time)/60:.2f} minutes.')