import os
import re
import sys
import time
import wandb
import pandas as pd
import numpy as np
import torch
import transformers
from transformers import AutoTokenizer
from sklearn.preprocessing import normalize
from scipy.spatial.distance import jensenshannon
from models import *


BATCH_SIZE = 16
NUMBER_OF_SENSES = {'level_1': 4,
                    'level_2': 17,
                    'level_3': 28}

if len(sys.argv) != 10:
    print('The expected comand should be: python classification_model.py train_language test_language architecture model loss optimizer learning_rate scheduler wandb')
    sys.exit()

TRAIN_LANG = sys.argv[1]
if TRAIN_LANG not in ['all', 'en', 'de', 'fr', 'cs']:
    print('Type a valid language: all, en, de, fr or cs.')
    exit()

TEST_LANG = sys.argv[2]
if TEST_LANG not in ['all', 'en', 'de', 'fr', 'cs']:
    print('Type a valid language: all, en, de, fr or cs.')
    exit()

ARCH = sys.argv[3]
if ARCH not in ['original', 'concat', 'wsum']:
    print('Type a valid architecture: original, concat or wsum.')
    exit()

MODEL_NAME = sys.argv[4]
if MODEL_NAME not in ['bert-base-uncased', 'distilbert-base-uncased', 'roberta-base', 'distilroberta-base', 'xlm-roberta-base', 'google/flan-t5-base', 'EuroBERT/EuroBERT-210m', 'answerdotai/ModernBERT-base']:
    print('Type a valid model name: bert-base-uncased, distilbert-base-uncased, roberta-base, distilroberta-base, xlm-roberta-base, google/flan-t5-base, EuroBERT/EuroBERT-210m or answerdotai/ModernBERT-base.')
    exit()

LOSS = sys.argv[5]
if LOSS not in ['cross-entropy', 'l1', 'l2', 'smooth-l1']:
    print('Type a valid loss: cross-entropy, l1, l2 or smooth-l1.')
    exit()

OPTIMIZER = sys.argv[6]
if OPTIMIZER not in ['adam', 'adamw', 'sgd', 'rms']:
    print('Type a valid optimizer: adam, adamw, sgd or rms.')
    exit()

LEARNING_RATE = float(sys.argv[7])
if LEARNING_RATE not in [1e-4, 5e-5, 1e-5, 5e-6, 1e-6]:
    print('Type a valid learning rate: 1e-4, 5e-5, 1e-5, 5e-6 or 1e-6.')
    exit()

SCHEDULER = sys.argv[8]
if SCHEDULER not in ['linear', 'cosine', 'none']:
    print('Type a valid scheduler: linear, cosine or none.')
    exit()

WANDB = sys.argv[9]
if WANDB not in ['true', 'false']:
    print('Type a valid wandb bool: true or false.')
    exit()


def log_wandb(mode, js_1, js_2, js_3, loss=None):
    'Log metrics on Weights & Biases.'

    if mode == 'Training':
        wandb.log({mode + ' JS Distance (Level-1)': js_1,
                   mode + ' JS Distance (Level-2)': js_2,
                   mode + ' JS Distance (Level-3)': js_3,
                   mode + ' Loss'                 : loss})
    else: 
        wandb.log({mode + ' JS Distance (Level-1)': js_1,
                   mode + ' JS Distance (Level-2)': js_2,
                   mode + ' JS Distance (Level-3)': js_3})


def create_dataloader(path):
    'Create dataloader class for multi-label implicit discourse relation recognition data splits.'
    
    # read pre-processed data
    df = pd.read_csv(path)

    # initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    transformers.logging.set_verbosity_error() # remove model checkpoint warning

    # prepare text encodings and labels
    encodings = tokenizer(list(df['arg1_arg2']), truncation=True, padding=True)
    labels = np.hstack((np.array(df.iloc[:,7:11]),   # level-1 columns
                        np.array(df.iloc[:,12:29]),  # level-2 columns
                        np.array(df.iloc[:,30:58]))) # level-3 columns

    # generate datasets
    dataset = Multi_IDDR_Dataset(encodings, labels)

    # generate dataloaders
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    return dataloader


def normalize_prob_distribution(distribution):
    'Shift values in a probability distribution so that they are all positive and apply l1 normalization so that thay add to 1.'

    shifted_distribution = distribution - np.min(distribution)
    normalized_distribution = normalize(shifted_distribution.reshape(1, -1), norm='l1')

    return normalized_distribution.reshape(-1)


def shift_prob_distribution(distribution, margin):
    'Marginally shift and renormalize probability distribution to avoid division by 0 in KL and JS distance calculation.'

    shifted_distribution = distribution + margin
    normalized_distribution = normalize(shifted_distribution.reshape(1, -1), norm='l1')

    return normalized_distribution.reshape(-1)


def get_js_distance(level, labels, predictions):
    'Calculate the average of the Jensen–Shannon distance between two probability distributions over all instances in a batch.'

    js_distance = 0

    for i in range(labels.size(dim=0)):
        true_distribution = labels[i].detach().numpy()
        true_distribution = shift_prob_distribution(true_distribution, 0.001) # shift distribution to avoid division by 0
        predicted_distribution = normalize_prob_distribution(predictions[i].detach().numpy())
        predicted_distribution = shift_prob_distribution(predicted_distribution, 0.001) # shift distribution to avoid division by 0

        js_distance += jensenshannon(true_distribution, predicted_distribution, base=2)

    print(level + f' || JS Distance: {js_distance / labels.size(dim=0):.4f}')

    return js_distance / labels.size(dim=0)


def test_loop(mode, dataloader, scheduler_bool=False, iteration=None):
    'Validation and test loop of the classification model.'

    # group metric across all batches
    js_1 = 0
    js_2 = 0
    js_3 = 0
    labels_l1 = []
    labels_l2 = []
    labels_l3 = []
    predictions_l1 = []
    predictions_l2 = []
    predictions_l3 = []

    model.eval()

    with torch.no_grad():

        for batch_idx, batch in enumerate(dataloader):

            # forward pass
            model_output = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])

            js_1 += get_js_distance('Level-1', batch['labels_level_1'], model_output['classifier_level_1'])
            js_2 += get_js_distance('Level-2', batch['labels_level_2'], model_output['classifier_level_2'])
            js_3 += get_js_distance('Level-3', batch['labels_level_3'], model_output['classifier_level_3'])

            labels_l1.extend(torch.argmax(batch['labels_level_1'], dim=1).tolist())
            labels_l2.extend(torch.argmax(batch['labels_level_2'], dim=1).tolist())
            labels_l3.extend(torch.argmax(batch['labels_level_3'], dim=1).tolist())
            predictions_l1.extend(torch.argmax(model_output['classifier_level_1'], dim=1).tolist())
            predictions_l2.extend(torch.argmax(model_output['classifier_level_2'], dim=1).tolist())
            predictions_l3.extend(torch.argmax(model_output['classifier_level_3'], dim=1).tolist())

    if i != None:
        if not os.path.exists('Results'):
            os.makedirs('Results')
        if not os.path.exists('Results/DiscoGeM-2.0_'+TRAIN_LANG):
            os.makedirs('Results/DiscoGeM-2.0_'+TRAIN_LANG)

    after_slash_model_name = re.search(r'[^/]+$', MODEL_NAME).group()

    if mode == 'Testing':
        if scheduler_bool == False:
            results_path = ARCH+'_'+after_slash_model_name+'_'+str(LEARNING_RATE)+'_'+str(i+1)+'.txt'
        else:
            results_path = ARCH+'_'+after_slash_model_name+'_'+str(LEARNING_RATE)+'_'+SCHEDULER+'_'+str(i+1)+'.txt'
        np.savetxt('Results/DiscoGeM-2.0_'+TRAIN_LANG+'/'+TEST_LANG+'_labels_l1_' + results_path, np.array(labels_l1), delimiter = ',')
        np.savetxt('Results/DiscoGeM-2.0_'+TRAIN_LANG+'/'+TEST_LANG+'/labels_l2_' + results_path, np.array(labels_l2), delimiter = ',')
        np.savetxt('Results/DiscoGeM-2.0_'+TRAIN_LANG+'/'+TEST_LANG+'/labels_l3_' + results_path, np.array(labels_l3), delimiter = ',')
        np.savetxt('Results/DiscoGeM-2.0_'+TRAIN_LANG+'/'+TEST_LANG+'/predictions_l1_' + results_path, np.array(predictions_l1), delimiter = ',')
        np.savetxt('Results/DiscoGeM-2.0_'+TRAIN_LANG+'/'+TEST_LANG+'/predictions_l2_' + results_path, np.array(predictions_l2), delimiter = ',')
        np.savetxt('Results/DiscoGeM-2.0_'+TRAIN_LANG+'/'+TEST_LANG+'/predictions_l3_' + results_path, np.array(predictions_l3), delimiter = ',')

    js_1 = js_1 / len(dataloader)
    js_2 = js_2 / len(dataloader)
    js_3 = js_3 / len(dataloader)
    
    if WANDB == 'true':
        log_wandb(mode, js_1, js_2, js_3)


for i in range(3):

    if WANDB == 'true':
        wandb.login()
        wandb.init(project = 'Multi-IDRR',
                   name = 'test-'+TRAIN_LANG+'-'+TEST_LANG+'-'+ARCH+'-'+MODEL_NAME+'-'+str(i+1),
                   config = {'Language': TRAIN_LANG+'-'+TEST_LANG,
                             'Architecture': ARCH,
                             'Model': MODEL_NAME,
                             'Loss': LOSS,
                             'Optimizer': OPTIMIZER,
                             'Learning Rate': LEARNING_RATE,
                             'Scheduler': SCHEDULER})

    test_loader = create_dataloader('Data/DiscoGeM-2.0/discogem_2_single_lang_' + TEST_LANG + '_test.csv')

    if ARCH == 'original':
        model = Multi_IDDR_Classifier(MODEL_NAME, NUMBER_OF_SENSES)
    elif ARCH == 'concat':
        model = Multi_IDDR_Classifier_Concat(MODEL_NAME, NUMBER_OF_SENSES)
    elif ARCH == 'wsum' and MODEL_NAME != 'google/flan-t5-base':
        model = Multi_IDDR_Classifier_WSum(MODEL_NAME, NUMBER_OF_SENSES)
    else:
        model = Multi_IDDR_Classifier_WSum_T5(MODEL_NAME, NUMBER_OF_SENSES)

    after_slash_model_name = re.search(r'[^/]+$', MODEL_NAME).group()

    if SCHEDULER == 'none':
        model_path = 'Models/DiscoGeM-2.0_'+TRAIN_LANG+'/'+ARCH+'_'+after_slash_model_name+'_'+str(LEARNING_RATE)+'_'+str(i+1)+'.pth'
    else:
        model_path = 'Models/DiscoGeM-2.0_'+TRAIN_LANG+'/'+ARCH+'_'+after_slash_model_name+'_'+str(LEARNING_RATE)+'_'+SCHEDULER+'_'+str(i+1)+'.pth'

    model.load_state_dict(torch.load(model_path))

    print('Starting testing...')
    start_time = time.time()

    test_loop('Testing', test_loader, iteration=i)

    print(f'Total testing time: {(time.time()-start_time)/60:.2f} minutes')

    wandb.finish()