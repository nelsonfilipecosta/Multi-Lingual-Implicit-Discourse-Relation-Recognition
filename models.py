import torch
from transformers import AutoModel, AutoModelForSeq2SeqLM


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


class Multi_IDDR_Classifier(torch.nn.Module):
    'Multi-head classification model for multi-label implicit discourse relation recognition.'
    
    def __init__(self, model_name, number_of_senses):
        super().__init__()
        self.pretrained_model   = AutoModel.from_pretrained(model_name)
        hidden_dimension        = self.pretrained_model.config.hidden_size
        # common layers
        self.hidden             = torch.nn.Linear(hidden_dimension, hidden_dimension)
        self.dropout            = torch.nn.Dropout(p=0.5)
        # classification layers for each level
        self.classifier_level_1 = torch.nn.Linear(hidden_dimension, number_of_senses['level_1'])
        self.classifier_level_2 = torch.nn.Linear(hidden_dimension, number_of_senses['level_2'])
        self.classifier_level_3 = torch.nn.Linear(hidden_dimension, number_of_senses['level_3'])
    
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
        hidden_dimension        = self.pretrained_model.config.hidden_size
        # common layers
        self.hidden             = torch.nn.Linear(hidden_dimension, hidden_dimension)
        self.dropout            = torch.nn.Dropout(p=0.5)
        # linear layers to increase the dimensions of output_1 and output_2
        self.increase_dimensions_1 = torch.nn.Sequential(torch.nn.Linear(number_of_senses['level_1'], hidden_dimension//2), # middle step to increase dimensions
                                                         torch.nn.GELU(),                                                   # GELU helps stabilizing learning
                                                         torch.nn.Linear(hidden_dimension//2, hidden_dimension),            # when projecting from small to high dimensions
                                                         torch.nn.GELU())
        self.increase_dimensions_2 = torch.nn.Sequential(torch.nn.Linear(number_of_senses['level_2'], hidden_dimension//2), # middle step to increase dimensions
                                                         torch.nn.GELU(),                                                   # GELU helps stabilizing learning
                                                         torch.nn.Linear(hidden_dimension//2, hidden_dimension),            # when projecting from small to high dimensions
                                                         torch.nn.GELU())
        # classification layers for each level
        self.classifier_level_1 = torch.nn.Linear(hidden_dimension, number_of_senses['level_1'])
        self.classifier_level_2 = torch.nn.Linear(hidden_dimension, number_of_senses['level_2'])
        self.classifier_level_3 = torch.nn.Linear(hidden_dimension, number_of_senses['level_3'])
        # learnable weight parameters for the weighted sums
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
        increased_output_1 = self.increase_dimensions_1(output_1)
        # weighted sum of inputs for level-2 classifier
        alpha = torch.sigmoid(self.alpha_param)
        combined_input_2 = alpha * increased_output_1 + (1 - alpha) * intermediate_output
        # level-2 classifier
        output_2 = self.classifier_level_2(combined_input_2)
        # increase dimensions to match the intermediate output
        increased_output_2 = self.increase_dimensions_2(output_2)
        # weighted sum of inputs for level-3 classifier
        beta_1, beta_2, beta_3 = torch.softmax(torch.stack([self.beta_1_param, self.beta_2_param, self.beta_3_param]), dim=0)
        combined_input_3 = beta_1 * increased_output_1 + beta_2 * increased_output_2 + beta_3 * intermediate_output
        # level-3 classifier
        output_3 = self.classifier_level_3(combined_input_3)
        # final output
        logits = {'classifier_level_1': output_1,
                  'classifier_level_2': output_2,
                  'classifier_level_3': output_3}
        return logits


class Multi_IDDR_Classifier_WSum_T5(torch.nn.Module):
    'Some text here'

    def __init__(self, model_name, number_of_senses):
        super().__init__()
        self.pretrained_model   = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        hidden_dimension        = self.pretrained_model.config.d_model
        # common layers
        self.hidden             = torch.nn.Linear(hidden_dimension, hidden_dimension)
        self.dropout            = torch.nn.Dropout(p=0.5)
        # linear layers to increase the dimensions of output_1 and output_2
        self.increase_dimensions_1 = torch.nn.Sequential(torch.nn.Linear(number_of_senses['level_1'], hidden_dimension//2), # middle step to increase dimensions
                                                         torch.nn.GELU(),                                                   # GELU helps stabilizing learning
                                                         torch.nn.Linear(hidden_dimension//2, hidden_dimension),            # when projecting from small to high dimensions
                                                         torch.nn.GELU())
        self.increase_dimensions_2 = torch.nn.Sequential(torch.nn.Linear(number_of_senses['level_2'], hidden_dimension//2), # middle step to increase dimensions
                                                         torch.nn.GELU(),                                                   # GELU helps stabilizing learning
                                                         torch.nn.Linear(hidden_dimension//2, hidden_dimension),            # when projecting from small to high dimensions
                                                         torch.nn.GELU())
        # classification layers for each level
        self.classifier_level_1 = torch.nn.Linear(hidden_dimension, number_of_senses['level_1'])
        self.classifier_level_2 = torch.nn.Linear(hidden_dimension, number_of_senses['level_2'])
        self.classifier_level_3 = torch.nn.Linear(hidden_dimension, number_of_senses['level_3'])
        # learnable weight parameters for the weighted sums
        self.alpha_param = torch.nn.Parameter(torch.tensor(0.5))  # initialize at 0.5
        self.beta_1_param = torch.nn.Parameter(torch.tensor(0.25)) # initialize at 0.25
        self.beta_2_param = torch.nn.Parameter(torch.tensor(0.25)) # initialize at 0.25
        self.beta_3_param = torch.nn.Parameter(torch.tensor(0.5)) # initialize at 0.5

    def forward(self, input_ids, attention_mask):
        llm_econder_states  = self.pretrained_model.encoder(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state   = llm_econder_states.last_hidden_state
        # intermediate states
        intermediate_output = last_hidden_state[:, 0]
        intermediate_output = self.hidden(intermediate_output)
        intermediate_output = self.dropout(intermediate_output)
        # level-1 classifier
        output_1 = self.classifier_level_1(intermediate_output)
        # increase dimensions to match the intermediate output
        increased_output_1 = self.increase_dimensions_1(output_1)
        # weighted sum of inputs for level-2 classifier
        alpha = torch.sigmoid(self.alpha_param)
        combined_input_2 = alpha * increased_output_1 + (1 - alpha) * intermediate_output
        # level-2 classifier
        output_2 = self.classifier_level_2(combined_input_2)
        # increase dimensions to match the intermediate output
        increased_output_2 = self.increase_dimensions_2(output_2)
        # weighted sum of inputs for level-3 classifier
        beta_1, beta_2, beta_3 = torch.softmax(torch.stack([self.beta_1_param, self.beta_2_param, self.beta_3_param]), dim=0)
        combined_input_3 = beta_1 * increased_output_1 + beta_2 * increased_output_2 + beta_3 * intermediate_output
        # level-3 classifier
        output_3 = self.classifier_level_3(combined_input_3)
        # final output
        logits = {'classifier_level_1': output_1,
                  'classifier_level_2': output_2,
                  'classifier_level_3': output_3}
        return logits