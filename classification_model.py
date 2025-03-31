import os
import sys
import time
import wandb
import pandas as pd
import numpy as np
import torch
import transformers
from transformers import AutoTokenizer
from transformers import AutoModel
from sklearn import metrics
from sklearn.preprocessing import normalize
from scipy.stats import entropy
from scipy.spatial.distance import jensenshannon
from models import *


EPOCHS = 10
BATCH_SIZE = 16
NUMBER_OF_SENSES = {'level_1': 4,
                    'level_2': 17,
                    'level_3': 28}

if len(sys.argv) != 8:
    print('The expected comand should be: python classification_model.py language architecture model loss optimizer learning_rate scheduler')
    sys.exit()

LANG = sys.argv[1]
if LANG not in ['all', 'en', 'de', 'fr', 'cs']:
    print('Type a valid language: all, en, de, fr or cs.')
    exit()

ARCH = sys.argv[2]
if ARCH not in ['concat', 'wsum', 'wsum_red', 'wsum_ind']:
    print('Type a valid architecture: concat, wsum, wsum_red or wsum_ind.')
    exit()

MODEL_NAME = sys.argv[3]
if MODEL_NAME not in ['bert-base-uncased', 'distilbert-base-uncased', 'roberta-base', 'distilroberta-base']:
    print('Type a valid model name: bert-base-uncased, distilbert-base-uncased, roberta-base or distilroberta-base.')
    exit()

LOSS = sys.argv[4]
if LOSS not in ['cross-entropy', 'l1', 'l2', 'smooth-l1']:
    print('Type a valid loss: cross-entropy, l1, l2 or smooth-l1.')
    exit()

OPTIMIZER = sys.argv[5]
if OPTIMIZER not in ['adam', 'adamw', 'sgd', 'rms']:
    print('Type a valid optimizer: adam, adamw, sgd or rms.')
    exit()

LEARNING_RATE = float(sys.argv[6])
if LEARNING_RATE not in [1e-4, 5e-5, 1e-5, 5e-6, 1e-6]:
    print('Type a valid learning rate: 1e-4, 5e-5, 1e-5, 5e-6 or 1e-6.')
    exit()

SCHEDULER = sys.argv[7]
if SCHEDULER not in ['linear', 'cosine', 'none']:
    print('Type a valid scheduler: linear, cosine or none.')
    exit()

WANDB = 0 # set 1 for logging and 0 for local runs


def log_wandb(mode, js_1, f1_score_1, precision_1, recall_1, js_2, f1_score_2, precision_2, recall_2, js_3, f1_score_3, precision_3, recall_3, loss=None):
    'Log metrics on Weights & Biases.'

    if mode == 'Training':
        wandb.log({mode + ' JS Distance (Level-1)': js_1,
                   mode + ' F1 Score (Level-1)'   : f1_score_1,
                   mode + ' Precision (Level-1)'  : precision_1,
                   mode + ' Recall (Level-1)'     : recall_1,
                   mode + ' JS Distance (Level-2)': js_2,
                   mode + ' F1 Score (Level-2)'   : f1_score_2,
                   mode + ' Precision (Level-2)'  : precision_2,
                   mode + ' Recall (Level-2)'     : recall_2,
                   mode + ' JS Distance (Level-3)': js_3,
                   mode + ' F1 Score (Level-3)'   : f1_score_3,
                   mode + ' Precision (Level-3)'  : precision_3,
                   mode + ' Recall (Level-3)'     : recall_3,
                   mode + ' Loss'                  : loss})
    else: 
        wandb.log({mode + ' JS Distance (Level-1)': js_1,
                   mode + ' F1 Score (Level-1)'   : f1_score_1,
                   mode + ' Precision (Level-1)'  : precision_1,
                   mode + ' Recall (Level-1)'     : recall_1,
                   mode + ' JS Distance (Level-2)': js_2,
                   mode + ' F1 Score (Level-2)'   : f1_score_2,
                   mode + ' Precision (Level-2)'  : precision_2,
                   mode + ' Recall (Level-2)'     : recall_2,
                   mode + ' JS Distance (Level-3)': js_3,
                   mode + ' F1 Score (Level-3)'   : f1_score_3,
                   mode + ' Precision (Level-3)'  : precision_3,
                   mode + ' Recall (Level-3)'     : recall_3})


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


def get_loss(predictions, labels):
    'Calculate overall loss of the model as the sum of the cross entropy losses of each classification head.'

    loss_level_1 = loss_function(predictions['classifier_level_1'], labels['labels_level_1'])
    loss_level_2 = loss_function(predictions['classifier_level_2'], labels['labels_level_2'])
    loss_level_3 = loss_function(predictions['classifier_level_3'], labels['labels_level_3'])
    
    return loss_level_1 + loss_level_2 + loss_level_3


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


def get_kl_distance(level, labels, predictions):
    'Calculate the average of the Kullback–Leibler distance between two probability distributions over all instances in a batch.'

    kl_distance = 0

    for i in range(labels.size(dim=0)):
        true_distribution = labels[i].detach().numpy()
        true_distribution = shift_prob_distribution(true_distribution, 0.001) # shift distribution to avoid division by 0
        predicted_distribution = normalize_prob_distribution(predictions[i].detach().numpy())
        predicted_distribution = shift_prob_distribution(predicted_distribution, 0.001) # shift distribution to avoid division by 0

        kl_distance += entropy(true_distribution, predicted_distribution, base=2)

    print(level + f' || KL Distance: {kl_distance / labels.size(dim=0):.4f}')

    return kl_distance / labels.size(dim=0)


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


def get_hell_distance(level, labels, predictions):
    'Calculate the average of the Hellinger distance between two probability distributions over all instances in a batch.'

    hell_distance = 0

    for i in range(labels.size(dim=0)):
        true_distribution = labels[i].detach().numpy()
        predicted_distribution = normalize_prob_distribution(predictions[i].detach().numpy())

        hell_distance += np.sqrt(np.sum((np.sqrt(true_distribution)-np.sqrt(predicted_distribution))**2)) / np.sqrt(2)
    
    print(level + f' || Hellinger Distance: {hell_distance / labels.size(dim=0):.4f}')

    return hell_distance / labels.size(dim=0)


def get_single_metrics(level, labels, predictions):
    'Get f1-score, precision and recall metrics for single-label classification.'

    f1_score    = metrics.f1_score(labels, predictions, average='weighted', zero_division=0)
    precision   = metrics.precision_score(labels, predictions, average='weighted', zero_division=0)
    recall      = metrics.recall_score(labels, predictions, average='weighted', zero_division=0)
    
    print(level + f' || F1 Score: {f1_score:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}')
    
    return f1_score, precision, recall


def train_loop(dataloader, scheduler_bool=False):
    'Train loop of the classification model.'

    model.train()

    for batch_idx, batch in enumerate(dataloader):

        # forward pass
        model_output = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
        loss = get_loss(model_output, batch)

        # backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if scheduler_bool == True:
            scheduler.step()

        # log results every 10 batches
        if not batch_idx % 10:

            print(f'Epoch: {epoch+1:02d}/{EPOCHS:02d} | Batch: {batch_idx:04d}/{len(dataloader):04d} | Loss: {loss:.4f}')

            js_1 = get_js_distance('Level-1', batch['labels_level_1'], model_output['classifier_level_1'])
            js_2 = get_js_distance('Level-2', batch['labels_level_2'], model_output['classifier_level_2'])
            js_3 = get_js_distance('Level-3', batch['labels_level_3'], model_output['classifier_level_3'])
            
            if WANDB == 1:
                log_wandb('Training', js_1, js_2, js_3, loss)
    
    return model.state_dict()


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
        if not os.path.exists('Results/DiscoGeM-2.0_' + LANG):
            os.makedirs('Results/DiscoGeM-2.0_' + LANG)

    if mode == 'Testing':
        if scheduler_bool == False:
            results_path = LOSS+'_'+OPTIMIZER+'_'+str(LEARNING_RATE)+'_'+str(iteration)+'.txt'
        else:
            results_path = LOSS+'_'+OPTIMIZER+'_'+str(LEARNING_RATE)+'_'+SCHEDULER+ '_'+str(iteration)+'.txt'
        np.savetxt('Results/DiscoGeM-2.0_' + LANG + '/labels_l1_' + results_path, np.array(labels_l1), delimiter = ',')
        np.savetxt('Results/DiscoGeM-2.0_' + LANG + '/labels_l2_' + results_path, np.array(labels_l2), delimiter = ',')
        np.savetxt('Results/DiscoGeM-2.0_' + LANG + '/labels_l3_' + results_path, np.array(labels_l3), delimiter = ',')
        np.savetxt('Results/DiscoGeM-2.0_' + LANG + '/predictions_l1_' + results_path, np.array(predictions_l1), delimiter = ',')
        np.savetxt('Results/DiscoGeM-2.0_' + LANG + '/predictions_l2_' + results_path, np.array(predictions_l2), delimiter = ',')
        np.savetxt('Results/DiscoGeM-2.0_' + LANG + '/predictions_l3_' + results_path, np.array(predictions_l3), delimiter = ',')

    js_1 = js_1 / len(dataloader)
    js_2 = js_2 / len(dataloader)
    js_3 = js_3 / len(dataloader)
    
    if WANDB == 1:
        log_wandb(mode, js_1, js_2, js_3)


for i in range(3):

    if WANDB == 1:
        wandb.login()
        run = wandb.init(project = 'IDRR', config = {'Model': MODEL_NAME,
                                                     'Epochs': EPOCHS,
                                                     'Batch Size': BATCH_SIZE,
                                                     'Loss': LOSS,
                                                     'Optimizer': OPTIMIZER,
                                                     'Learning Rate': LEARNING_RATE})
    
    train_loader      = create_dataloader('Data/DiscoGeM-2.0/discogem_2_single_lang_' + LANG + '_train.csv')
    validation_loader = create_dataloader('Data/DiscoGeM-2.0/discogem_2_single_lang_' + LANG + '_validation.csv')
    test_loader       = create_dataloader('Data/DiscoGeM-2.0/discogem_2_single_lang_' + LANG + '_test.csv')

    if ARCH == 'concat':
        model = Multi_IDDR_Classifier_Concat(MODEL_NAME, NUMBER_OF_SENSES)
    elif ARCH == 'wsum':
        model = Multi_IDDR_Classifier_WSum(MODEL_NAME, NUMBER_OF_SENSES)
    elif ARCH == 'wsum_red':
        model = Multi_IDDR_Classifier_WSum_Reduced_Dim(MODEL_NAME, NUMBER_OF_SENSES)
    else:
        model = Multi_IDDR_Classifier_WSum_Independent(MODEL_NAME, NUMBER_OF_SENSES)

    # choose loss
    if LOSS == 'cross-entropy':
        loss_function = torch.nn.CrossEntropyLoss(reduction='mean')
    elif LOSS == 'l1':
        loss_function = torch.nn.L1Loss(reduction='mean')
    elif LOSS == 'l2':
        loss_function = torch.nn.MSELoss(reduction='mean')
    else:
        loss_function = torch.nn.SmoothL1Loss(reduction='mean')

    # choose optimizer
    if OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, amsgrad=False)
    elif OPTIMIZER == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, amsgrad=False)
    elif OPTIMIZER == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, nesterov=False)
    else:
        optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARNING_RATE)
    
    # choose scheduler
    if SCHEDULER == 'linear':
        scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1, end_factor=0.5, total_iters=int(EPOCHS/2))
    elif SCHEDULER == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(EPOCHS/3), eta_min=0.5*LEARNING_RATE)

    print('Starting training...')
    start_time = time.time()

    for epoch in range(EPOCHS):
        if SCHEDULER == 'none':
            model_dict = train_loop(train_loader)
        else:
            model_dict = train_loop(train_loader, scheduler_bool=True)
        test_loop('Validation', validation_loader)
    
    # save model configuration
    if not os.path.exists('Models'):
        os.makedirs('Models')
    if not os.path.exists('Models/DiscoGeM-2.0_' + LANG):
        os.makedirs('Models/DiscoGeM-2.0_' + LANG)

    if SCHEDULER == 'none':
        model_path = 'Models/DiscoGeM-2.0_'+LANG+'/'+MODEL_NAME+'_'+LOSS+'_'+OPTIMIZER+'_'+str(LEARNING_RATE)+'_'+str(i+1)+'.pth'
    else:
        model_path = 'Models/DiscoGeM-2.0_'+LANG+'/'+MODEL_NAME+'_'+LOSS+'_'+OPTIMIZER+'_'+str(LEARNING_RATE)+'_'+SCHEDULER+'_'+str(i+1)+'.pth'
    torch.save(model_dict, model_path)

    if SCHEDULER == 'none':
        test_loop('Testing', test_loader, iteration=i)
    else:
        test_loop('Testing', test_loader, scheduler_bool=True, iteration=i+1)

    print(f'Total training time: {(time.time()-start_time)/60:.2f} minutes')

    wandb.finish()