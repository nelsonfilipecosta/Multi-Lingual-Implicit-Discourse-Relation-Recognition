import torch
from transformers import AutoModel


class Multi_IDDR_Dataset(torch.utils.data.Dataset):
    'Dataset class for multi-label implicit discourse relation regognition classification tasks.'

    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(value[idx]) for key, value in self.encodings.items()}
        item['labels_level_1'] = torch.tensor(self.labels[idx,0:4], dtype=torch.float32)   # level-1 columns (4 senses)
        item['labels_level_2'] = torch.tensor(self.labels[idx,4:21], dtype=torch.float32)  # level-2 columns (17 senses)
        item['labels_level_3'] = torch.tensor(self.labels[idx,21:49], dtype=torch.float32) # level-3 columns (28 senses)
        return item

    def __len__(self):
        return self.labels.shape[0]


class Multi_IDDR_Classifier_Concat(torch.nn.Module):
    'Multi-head classification model for multi-label implicit discourse relation recognition.'
    
    def __init__(self, model_name, number_of_senses):
        super().__init__()
        self.pretrained_model   = AutoModel.from_pretrained(model_name)
        # common layers
        self.hidden             = torch.nn.Linear(self.pretrained_model.config.hidden_size, self.pretrained_model.config.hidden_size)
        self.dropout            = torch.nn.Dropout(p=0.5)
        # classification head for each level
        self.classifier_level_1 = torch.nn.Linear(self.pretrained_model.config.hidden_size, number_of_senses['level_1'])
        self.classifier_level_2 = torch.nn.Linear(self.pretrained_model.config.hidden_size + number_of_senses['level_1'], number_of_senses['level_2'])
        self.classifier_level_3 = torch.nn.Linear(self.pretrained_model.config.hidden_size + number_of_senses['level_1'] + number_of_senses['level_2'], number_of_senses['level_3'])
    
    def forward(self, input_ids, attention_mask):
        llm_states = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = llm_states.last_hidden_state
        # intermediate states
        intermediate_output = last_hidden_state[:, 0]
        intermediate_output = self.hidden(intermediate_output)
        intermediate_output = self.dropout(intermediate_output)
        # classifier for each level
        output_1 = self.classifier_level_1(intermediate_output)
        output_2 = self.classifier_level_2(torch.cat([intermediate_output, output_1], dim=-1))
        output_3 = self.classifier_level_3(torch.cat([intermediate_output, output_1, output_2], dim=-1))
        # final output
        logits = {'classifier_level_1': output_1,
                  'classifier_level_2': output_2,
                  'classifier_level_3': output_3}
        return logits


class Multi_IDDR_Classifier_WSum(torch.nn.Module):
    'Some text here'

    def __init__(self, model_name, number_of_senses):
        super().__init__()
        self.pretrained_model   = AutoModel.from_pretrained(model_name)
        # common layers
        self.hidden             = torch.nn.Linear(self.pretrained_model.config.hidden_size, self.pretrained_model.config.hidden_size)
        self.dropout            = torch.nn.Dropout(p=0.5)
        # linear layers to increase the dimensions of output_1 and output_2
        self.increase_dimensions_1  = torch.nn.Linear(number_of_senses['level_1'], self.pretrained_model.config.hidden_size)
        self.increase_dimensions_2  = torch.nn.Linear(number_of_senses['level_2'], self.pretrained_model.config.hidden_size)
        # classification head for each level
        self.classifier_level_1 = torch.nn.Linear(self.pretrained_model.config.hidden_size, number_of_senses['level_1'])
        self.classifier_level_2 = torch.nn.Linear(self.pretrained_model.config.hidden_size, number_of_senses['level_2'])
        self.classifier_level_3 = torch.nn.Linear(self.pretrained_model.config.hidden_size, number_of_senses['level_3'])
        # learnable weight parameters for the weighted sum
        self.alpha_param = torch.nn.Parameter(torch.tensor(0.5))  # initialize at 0.5
        self.beta_1_param = torch.nn.Parameter(torch.tensor(0.25)) # initialize at 0.25
        self.beta_2_param = torch.nn.Parameter(torch.tensor(0.25)) # initialize at 0.25
        self.beta_3_param = torch.nn.Parameter(torch.tensor(0.5)) # initialize at 0.5

    def forward(self, input_ids, attention_mask):
        llm_states = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = llm_states.last_hidden_state
        # intermediate states
        intermediate_output = last_hidden_state[:, 0]
        intermediate_output = self.hidden(intermediate_output)
        intermediate_output = self.dropout(intermediate_output)
        # level-1 classifier
        output_1 = self.classifier_level_1(intermediate_output)
        # increase dimensions to match the intermediate output
        increased_output_1 = torch.relu(self.increase_dimensions_1(output_1)) # ReLU helps stabilizing learning when projecting from small to high dimensions
        # weighted sum of inputs for level-2 classifier
        alpha = torch.sigmoid(self.alpha_param) # contribution of output_1
        combined_input_2 = alpha * increased_output_1 + (1 - alpha) * intermediate_output
        # level-2 classifier
        output_2 = self.classifier_level_2(combined_input_2)
        # increase dimensions to match the intermediate output
        increased_output_2 = torch.relu(self.increase_dimensions_2(output_2)) # ReLU helps stabilizing learning when projecting from small to high dimensions
        # weighted sum of inputs for level-3 classifier
        beta_1, beta_2, beta_3 = torch.softmax(torch.stack([self.beta_1_param, self.beta_2_param, self.beta_3_param]), dim=0) # the softmax ensures the sum of the betas is in [0,1]
        combined_input_3 = beta_1 * increased_output_1 + beta_2 * increased_output_2 + beta_3 * intermediate_output
        # level-3 classifier
        output_3 = self.classifier_level_3(combined_input_3)
        # final output
        logits = {'classifier_level_1': output_1,
                  'classifier_level_2': output_2,
                  'classifier_level_3': output_3}
        return logits


class Multi_IDDR_Classifier_WSum_Reduced_Dim(torch.nn.Module):
    'Some text here'

    def __init__(self, model_name, number_of_senses):
        super().__init__()
        self.pretrained_model   = AutoModel.from_pretrained(model_name)
        # common layers
        self.hidden             = torch.nn.Linear(self.pretrained_model.config.hidden_size, self.pretrained_model.config.hidden_size)
        self.dropout            = torch.nn.Dropout(p=0.5)
        # linear layer to reduce the dimension of the hidden state
        self.reduce_dimensions  = torch.nn.Linear(self.pretrained_model.config.hidden_size, number_of_senses['level_3'])
        # linear layers to increase the dimensions of output_1 and output_2
        self.increase_dimensions_1  = torch.nn.Linear(number_of_senses['level_1'], number_of_senses['level_3'])
        self.increase_dimensions_2  = torch.nn.Linear(number_of_senses['level_2'], number_of_senses['level_3'])
        # classification head for each level
        self.classifier_level_1 = torch.nn.Linear(number_of_senses['level_3'], number_of_senses['level_1'])
        self.classifier_level_2 = torch.nn.Linear(number_of_senses['level_3'], number_of_senses['level_2'])
        self.classifier_level_3 = torch.nn.Linear(number_of_senses['level_3'], number_of_senses['level_3'])
        # learnable weight parameters for the weighted sum
        self.alpha_param = torch.nn.Parameter(torch.tensor(0.5))  # initialize at 0.5
        self.beta_1_param = torch.nn.Parameter(torch.tensor(0.25)) # initialize at 0.25
        self.beta_2_param = torch.nn.Parameter(torch.tensor(0.25)) # initialize at 0.25

    def forward(self, input_ids, attention_mask):
        llm_states = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = llm_states.last_hidden_state
        # intermediate states
        intermediate_output = last_hidden_state[:, 0]
        intermediate_output = self.hidden(intermediate_output)
        intermediate_output = self.dropout(intermediate_output)
        reduced_intermediate_output = self.reduce_dimensions(intermediate_output)
        # level-1 classifier
        output_1 = self.classifier_level_1(reduced_intermediate_output)
        # increase dimensions to match the intermediate output
        increased_output_1 = torch.relu(self.increase_dimensions_1(output_1)) # ReLU helps stabilizing learning when projecting from small to high dimensions
        # weighted sum of inputs for level-2 classifier
        alpha = torch.sigmoid(self.alpha_param) # contribution of output_1
        combined_input_2 = alpha * increased_output_1 + (1 - alpha) * reduced_intermediate_output
        # level-2 classifier
        output_2 = self.classifier_level_2(combined_input_2)
        # increase dimensions to match the intermediate output
        increased_output_2 = torch.relu(self.increase_dimensions_2(output_2)) # ReLU helps stabilizing learning when projecting from small to high dimensions
        # weighted sum of inputs for level-3 classifier
        beta_1, beta_2, beta_3 = torch.softmax(torch.stack([self.beta_1_param, self.beta_2_param, torch.tensor(0.5)]), dim=0) # the softmax ensures the sum of the betas is in [0,1]
        combined_input_3 = beta_1 * increased_output_1 + beta_2 * increased_output_2 + beta_3 * reduced_intermediate_output
        # level-3 classifier
        output_3 = self.classifier_level_3(combined_input_3)
        # final output
        logits = {'classifier_level_1': output_1,
                  'classifier_level_2': output_2,
                  'classifier_level_3': output_3}
        return logits


class Multi_IDDR_Classifier_WSum_Independent(torch.nn.Module):
    'Some text here'

    def __init__(self, model_name, number_of_senses):
        super().__init__()
        self.pretrained_model   = AutoModel.from_pretrained(model_name)
        # common layers
        self.hidden             = torch.nn.Linear(self.pretrained_model.config.hidden_size, self.pretrained_model.config.hidden_size)
        self.dropout            = torch.nn.Dropout(p=0.5)
        # linear layer to reduce the dimension of the hidden state
        self.reduce_dimensions  = torch.nn.Linear(self.pretrained_model.config.hidden_size, number_of_senses['level_3'])
        # linear layers to increase the dimensions of output_1 and output_2
        self.increase_dimensions_1  = torch.nn.Linear(number_of_senses['level_1'], number_of_senses['level_3'])
        self.increase_dimensions_2  = torch.nn.Linear(number_of_senses['level_2'], number_of_senses['level_3'])
        # classification head for each level
        self.classifier_level_1 = torch.nn.Linear(number_of_senses['level_3'], number_of_senses['level_1'])
        self.classifier_level_2 = torch.nn.Linear(number_of_senses['level_3'], number_of_senses['level_2'])
        self.classifier_level_3 = torch.nn.Linear(number_of_senses['level_3'], number_of_senses['level_3'])
        # learnable weight parameters for the weighted sum
        self.alpha_param = torch.nn.Parameter(torch.tensor(0.5))  # initialize at 0.5
        self.beta_1_param = torch.nn.Parameter(torch.tensor(0.5)) # initialize at 0.5
        self.beta_2_param = torch.nn.Parameter(torch.tensor(0.5)) # initialize at 0.5
        self.theta_param = torch.nn.Parameter(torch.tensor(0.5))  # initialize at 0.5

    def forward(self, input_ids, attention_mask):
        llm_states = self.pretrained_model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = llm_states.last_hidden_state
        # intermediate states
        intermediate_output = last_hidden_state[:, 0]
        intermediate_output = self.hidden(intermediate_output)
        intermediate_output = self.dropout(intermediate_output)
        reduced_intermediate_output = self.reduce_dimensions(intermediate_output)
        # level-1 classifier
        output_1 = self.classifier_level_1(reduced_intermediate_output)
        # increase dimensions to match the intermediate output
        increased_output_1 = torch.relu(self.increase_dimensions_1(output_1)) # ReLU helps stabilizing learning when projecting from small to high dimensions
        # weighted sum of inputs for level-2 classifier
        alpha = torch.sigmoid(self.alpha_param) # contribution of output_1
        combined_input_2 = alpha * increased_output_1 + (1 - alpha) * reduced_intermediate_output
        # level-2 classifier
        output_2 = self.classifier_level_2(combined_input_2)
        # increase dimensions to match the intermediate output
        increased_output_2 = torch.relu(self.increase_dimensions_2(output_2)) # ReLU helps stabilizing learning when projecting from small to high dimensions
        # weighted sum of inputs for level-3 classifier
        beta_1 = torch.sigmoid(self.beta_1_param)   # contribution of output_1 is in [0,1]
        beta_2 = torch.sigmoid(self.beta_2_param)   # contribution of output_2 is in [0,1]
        theta = torch.sigmoid(self.theta_param)     # balance between output_1 and output_2 is in [0,1]
        combined_input_3 = theta * (beta_1 * increased_output_1 + (1 - beta_1) * reduced_intermediate_output) + (1 - theta) * (beta_2 * increased_output_2 + (1 - beta_2) * reduced_intermediate_output)
        # level-3 classifier
        output_3 = self.classifier_level_3(combined_input_3)
        # final output
        logits = {'classifier_level_1': output_1,
                  'classifier_level_2': output_2,
                  'classifier_level_3': output_3}
        return logits