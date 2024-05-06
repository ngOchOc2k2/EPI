import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import json
import torch.utils.checkpoint
from transformers.models.bert.configuration_bert import BertConfig
from transformers import BertModel, BertForMaskedLM, BertConfig
from peft import LoraConfig, get_peft_model, PeftModel
from copy import deepcopy
import os
import numpy as np
import adapters
from adapters import PrefixTuningConfig



class base_model(nn.Module):
    def __init__(self):
        super(base_model, self).__init__()
        self.zero_const = nn.Parameter(torch.Tensor([0]))
        self.zero_const.requires_grad = False
        self.pi_const = nn.Parameter(torch.Tensor([3.14159265358979323846]))
        self.pi_const.requires_grad = False

    def save_checkpoint(self, path):
        torch.save(self.state_dict(), path)

    def load_checkpoint(self, path):
        self.load_state_dict(torch.load(os.path.join(path)))
        self.eval()

    def save_parameters(self, path):
        f = open(path, "w")
        f.write(json.dumps(self.get_parameters("list")))
        f.close()

    def load_parameters(self, path):
        f = open(path, "r")
        parameters = json.loads(f.read())
        f.close()
        for i in parameters:
            parameters[i] = torch.Tensor(parameters[i])
        self.load_state_dict(parameters, strict=False)
        self.eval()

    def get_parameters(self, mode="numpy", param_dict=None):
        all_param_dict = self.state_dict()
        if param_dict == None:
            param_dict = all_param_dict.keys()
        res = {}
        for param in param_dict:
            if mode == "numpy":
                res[param] = all_param_dict[param].cpu().numpy()
            elif mode == "list":
                res[param] = all_param_dict[param].cpu().numpy().tolist()
            else:
                res[param] = all_param_dict[param]
        return res

    def set_parameters(self, parameters):
        for i in parameters:
            parameters[i] = torch.Tensor(parameters[i])
        self.load_state_dict(parameters, strict = False)
        self.eval()
        
    

class Bert_Prompt(nn.Module):
    def __init__(self, config, path_adapter=True):
        super(Bert_Prompt, self).__init__()

        # get config
        self.bert_config = BertConfig.from_pretrained(config.bert_path)

        # get output size
        self.output_size = config.encoder_output_size

        # get model
        self.encoder = BertModel.from_pretrained(config.bert_path).cuda()
        

        config.hidden_size = self.bert_config.hidden_size
        config.output_size = config.encoder_output_size
        self.encoder.resize_token_embeddings(config.vocab_size + config.marker_size)
        
        if path_adapter == True:
            adapters.init(self.encoder)
            config = PrefixTuningConfig(flat=False, prefix_length=30)
            self.encoder.add_adapter("prefix_tuning", config=config)
            self.encoder.train_adapter("prefix_tuning")


    def save_adapter(self, save_path):
        self.encoder.save_adapter(save_path, "prefix_tuning")
    

    def forward(self, inputs):
        e11 = []
        e21 = []

        for i in range(inputs.size()[0]):
            try:
                tokens = inputs[i].cpu().numpy()
                e11.append(np.argwhere(tokens == 30522)[0][0])
                e21.append(np.argwhere(tokens == 30524)[0][0])
            except:
                print(inputs[i])

        tokens_output = self.encoder(inputs)[0] # [B,N] --> [B,N,H]
        output = []

        for i in range(len(e11)):
            instance_output = torch.index_select(tokens_output, 0, torch.tensor(i).cuda())
            instance_output = torch.index_select(instance_output, 1, torch.tensor([e11[i], e21[i]]).cuda())
            output.append(instance_output)  # [B,N] --> [B,2,H]

        output = torch.cat(output, dim=0)
        output = output.view(output.size()[0], -1)  # [B,N] --> [B,H*2]

        return output


    
class LayerNorm(nn.Module):
    def __init__(self, input_dim, cond_dim=0, center=True, scale=True, epsilon=None, conditional=False,
                 hidden_units=None, hidden_activation='linear', hidden_initializer='xaiver', **kwargs):
        super(LayerNorm, self).__init__()
        self.center = center
        self.scale = scale
        self.conditional = conditional
        self.hidden_units = hidden_units
        self.hidden_initializer = hidden_initializer
        self.epsilon = epsilon or 1e-12
        self.input_dim = input_dim
        self.cond_dim = cond_dim

        if self.center:
            self.beta = Parameter(torch.zeros(input_dim))
        if self.scale:
            self.gamma = Parameter(torch.ones(input_dim))

        if self.conditional:
            if self.hidden_units is not None:
                self.hidden_dense = nn.Linear(in_features=self.cond_dim, out_features=self.hidden_units, bias=False)
            if self.center:
                self.beta_dense = nn.Linear(in_features=self.cond_dim, out_features=input_dim, bias=False)
            if self.scale:
                self.gamma_dense = nn.Linear(in_features=self.cond_dim, out_features=input_dim, bias=False)

        self.initialize_weights()

    def initialize_weights(self):
        if self.conditional:
            if self.hidden_units is not None:
                if self.hidden_initializer == 'normal':
                    torch.nn.init.normal(self.hidden_dense.weight)
                elif self.hidden_initializer == 'xavier':  # glorot_uniform
                    torch.nn.init.xavier_uniform_(self.hidden_dense.weight)
            if self.center:
                torch.nn.init.constant_(self.beta_dense.weight, 0)
            if self.scale:
                torch.nn.init.constant_(self.gamma_dense.weight, 0)

    def forward(self, inputs, cond=None):
        if self.conditional:
            if self.hidden_units is not None:
                cond = self.hidden_dense(cond)
            for _ in range(len(inputs.shape) - len(cond.shape)):
                cond = cond.unsqueeze(1) 
            if self.center:
                beta = self.beta_dense(cond) + self.beta
            if self.scale:
                gamma = self.gamma_dense(cond) + self.gamma
        else:
            if self.center:
                beta = self.beta
            if self.scale:
                gamma = self.gamma

        outputs = inputs
        if self.center:
            mean = torch.mean(outputs, dim=-1).unsqueeze(-1)
            outputs = outputs - mean
        if self.scale:
            variance = torch.mean(outputs ** 2, dim=-1).unsqueeze(-1)
            std = (variance + self.epsilon) ** 0.5
            outputs = outputs / std
            outputs = outputs * gamma
        if self.center:
            outputs = outputs + beta

        return outputs



class Classifier_Layer(base_model):
    def __init__(self, config, num_class):
        super(Classifier_Layer, self).__init__()
        
        self.num_class = num_class
        self.head = nn.Sequential(
            nn.Linear(config.hidden_size * 2, self.num_class, bias=True),
        )
        
    def forward(self, inputs):
        return self.head(inputs)


