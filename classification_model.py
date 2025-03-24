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


EPOCHS = 10
BATCH_SIZE = 16
NUMBER_OF_SENSES = {'level_1': 4,
                    'level_2': 14,
                    'level_3': 22}

MODEL_NAME = sys.argv[1]
if MODEL_NAME not in ['bert-base-uncased', 'distilbert-base-uncased', 'roberta-base', 'distilroberta-base']:
    print('Type a valid model name: bert-base-uncased, distilbert-base-uncased, roberta-base or distilroberta-base.')
    exit()

LOSS = sys.argv[2]
if LOSS not in ['cross-entropy', 'l1', 'l2', 'smooth-l1']:
    print('Type a valid loss: cross-entropy, l1, l2 or smooth-l1.')
    exit()

OPTIMIZER = sys.argv[3]
if OPTIMIZER not in ['adam', 'adamw', 'sgd', 'rms']:
    print('Type a valid optimizer: adam, adamw, sgd or rms.')
    exit()

LEARNING_RATE = float(sys.argv[4])
if LEARNING_RATE not in [1e-4, 5e-5, 1e-5, 5e-6, 1e-6]:
    print('Type a valid learning rate: 1e-4, 5e-5, 1e-5, 5e-6 or 1e-6.')
    exit()

SCHEDULER = sys.argv[5]
if SCHEDULER not in ['linear', 'cosine']:
    print('Type a valid scheduler: linear or cosine.')
    exit()

WANDB = 1 # set 1 for logging and 0 for local runs


class Multi_IDDR_Dataset(torch.utils.data.Dataset):
    'Dataset class for multi-label implicit discourse relation regognition classification tasks.'

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    
    def __getitem__(self, idx):
        item = {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}
        item['labels_level_3'] = torch.tensor(self.labels[idx,0:22], dtype=torch.float32)  # level-3 columns
        item['labels_level_2'] = torch.tensor(self.labels[idx,22:36], dtype=torch.float32) # level-2 columns
        item['labels_level_1'] = torch.tensor(self.labels[idx,36:40], dtype=torch.float32) # level-1 columns
        return item

    def __len__(self):
        return self.labels.shape[0]


class Multi_IDDR_Classifier(torch.nn.Module):
    'Multi-head classification model for multi-label implicit discourse relation recognition.'
    
    def __init__(self, model_name, number_of_senses):
        super().__init__()
        self.pretrained_model   = AutoModel.from_pretrained(model_name)
        self.hidden             = torch.nn.Linear(self.pretrained_model.config.hidden_size, self.pretrained_model.config.hidden_size)
        self.dropout            = torch.nn.Dropout(p=0.5)
        self.classifier_level_1 = torch.nn.Linear(self.pretrained_model.config.hidden_size, number_of_senses['level_1'])
        self.classifier_level_2 = torch.nn.Linear(self.pretrained_model.config.hidden_size, number_of_senses['level_2'])
        self.classifier_level_3 = torch.nn.Linear(self.pretrained_model.config.hidden_size, number_of_senses['level_3'])
    
    def forward(self, input_ids, attention_mask):
        llm_states = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = llm_states.last_hidden_state
        output = last_hidden_state[:, 0]
        output = self.hidden(output)
        output = self.dropout(output)
        logits = {'classifier_level_1': self.classifier_level_1(output),
                  'classifier_level_2': self.classifier_level_2(output),
                  'classifier_level_3': self.classifier_level_3(output)}
        return logits


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
    labels = np.hstack((np.array(df.iloc[:,5:27]),   # level-3 columns
                        np.array(df.iloc[:,29:43]),  # level-2 columns
                        np.array(df.iloc[:,45:49]))) # level-1 columns

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


def train_loop(dataloader):
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
        scheduler.step()

        # log results every 10 batches
        if not batch_idx % 10:

            print(f'Epoch: {epoch+1:02d}/{EPOCHS:02d} | Batch: {batch_idx:04d}/{len(dataloader):04d} | Loss: {loss:.4f}')

            js_1 = get_js_distance('Level-1', batch['labels_level_1'], model_output['classifier_level_1'])
            js_2 = get_js_distance('Level-2', batch['labels_level_2'], model_output['classifier_level_2'])
            js_3 = get_js_distance('Level-3', batch['labels_level_3'], model_output['classifier_level_3'])

            f1_score_1, precision_1, recall_1 = get_single_metrics('Level-1',
                                                                   torch.argmax(batch['labels_level_1'], dim=1).numpy(),
                                                                   torch.argmax(model_output['classifier_level_1'], dim=1).numpy())
            f1_score_2, precision_2, recall_2 = get_single_metrics('Level-2',
                                                                   torch.argmax(batch['labels_level_2'], dim=1).numpy(),
                                                                   torch.argmax(model_output['classifier_level_2'], dim=1).numpy())
            f1_score_3, precision_3, recall_3 = get_single_metrics('Level-3',
                                                                   torch.argmax(batch['labels_level_3'], dim=1).numpy(),
                                                                   torch.argmax(model_output['classifier_level_3'], dim=1).numpy())
            
            if WANDB == 1:
                log_wandb('Training', js_1, f1_score_1, precision_1, recall_1, js_2, f1_score_2, precision_2, recall_2, js_3, f1_score_3, precision_3, recall_3, loss)
    
    return model.state_dict()


def test_loop(mode, dataloader, iteration=None):
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
        if not os.path.exists('Results/DiscoGeM_' + str(iteration)):
            os.makedirs('Results/DiscoGeM_' + str(iteration))

    if mode == 'Testing':
        np.savetxt('Results/DiscoGeM_' + str(iteration) + '/labels_l1.txt', np.array(labels_l1), delimiter = ',')
        np.savetxt('Results/DiscoGeM_' + str(iteration) + '/labels_l2.txt', np.array(labels_l2), delimiter = ',')
        np.savetxt('Results/DiscoGeM_' + str(iteration) + '/labels_l3.txt', np.array(labels_l3), delimiter = ',')
        np.savetxt('Results/DiscoGeM_' + str(iteration) + '/predictions_l1.txt', np.array(predictions_l1), delimiter = ',')
        np.savetxt('Results/DiscoGeM_' + str(iteration) + '/predictions_l2.txt', np.array(predictions_l2), delimiter = ',')
        np.savetxt('Results/DiscoGeM_' + str(iteration) + '/predictions_l3.txt', np.array(predictions_l3), delimiter = ',')

    js_1 = js_1 / len(dataloader)
    js_2 = js_2 / len(dataloader)
    js_3 = js_3 / len(dataloader)

    f1_score_1, precision_1, recall_1 = get_single_metrics('Level-1', np.array(labels_l1), np.array(predictions_l1))
    f1_score_2, precision_2, recall_2 = get_single_metrics('Level-2', np.array(labels_l2), np.array(predictions_l2))
    f1_score_3, precision_3, recall_3 = get_single_metrics('Level-3', np.array(labels_l3), np.array(predictions_l3))
    
    if WANDB == 1:
        log_wandb(mode, js_1, f1_score_1, precision_1, recall_1, js_2, f1_score_2, precision_2, recall_2, js_3, f1_score_3, precision_3, recall_3)


for i in range(3):

    if WANDB == 1:
        wandb.login()
        run = wandb.init(project = 'IDRR', config = {'Model': MODEL_NAME,
                                                     'Epochs': EPOCHS,
                                                     'Batch Size': BATCH_SIZE,
                                                     'Loss': LOSS,
                                                     'Optimizer': OPTIMIZER,
                                                     'Learning Rate': LEARNING_RATE})
    
    train_loader      = create_dataloader('Data/DiscoGeM/discogem_train.csv')
    validation_loader = create_dataloader('Data/DiscoGeM/discogem_validation.csv')
    test_loader       = create_dataloader('Data/DiscoGeM/discogem_test.csv')

    model = Multi_IDDR_Classifier(MODEL_NAME, NUMBER_OF_SENSES)

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
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, int(EPOCHS/3), eta_min=0.5*LEARNING_RATE)

    print('Starting training...')
    start_time = time.time()

    for epoch in range(EPOCHS):
        model_dict = train_loop(train_loader)
        test_loop('Validation', validation_loader)
    
    # save model configuration
    folder = 'Model_'+MODEL_NAME+'_'+LOSS+'_'+OPTIMIZER+'_'+str(LEARNING_RATE)+'_'+str(i+1)
    if not os.path.exists(folder):
        os.makedirs(folder)
    torch.save(model_dict, folder+'/model.pth')

    test_loop('Testing', test_loader, i)

    print(f'Total training time: {(time.time()-start_time)/60:.2f} minutes')

    wandb.finish()