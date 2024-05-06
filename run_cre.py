import random
import torch
from dataloaders.data_loader import get_data_loader
from sklearn.cluster import KMeans
from dataloaders.sampler import data_sampler
import torch.nn as nn
import torch.optim as optim
import numpy as np
import collections
import json
from tqdm import tqdm
import logging
import logging
from collections import Counter
import subprocess
import os
import re
from copy import deepcopy
from config import Param
from model.epi import EPI
from utils.assets import get_args, fix_seed, tunable_params_stastic
from model.bert import Bert_Prompt, Classifier_Layer


logger = logging.getLogger(__name__)


color_epoch = '\033[92m' 
color_loss = '\033[92m'  
color_number = '\033[93m'
color_reset = '\033[0m'




def set_seed_classifier(config, seed):
    config.n_gpu = torch.cuda.device_count()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if config.n_gpu > 0 and torch.cuda.is_available() and config.use_gpu:
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        

class CELoss(nn.Module):
    def __init__(self, temperature=2.0):
        super(CELoss, self).__init__()
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, pros, label):
        bce_loss = self.ce_loss(pros, label)
        return bce_loss




def save_jsonl(filename, data, name=None):
    if not os.path.exists(filename):
        os.makedirs(filename)
        
    with open(filename + name, 'w') as f:
        for item in data:
            json.dump(item, f)
            f.write('\n')
            
            
def save_model(config, prefix_model, classifier_model, path_adapter, task):
    if not os.path.exists(config.save_checkpoint + path_adapter):
        os.makedirs(config.save_checkpoint + path_adapter)
    
    prefix_model.save_adapter(config.save_checkpoint + path_adapter)
    torch.save(classifier_model, './' + config.save_checkpoint + path_adapter + f'/clasification.pt')
    


def train_simple_model(config, encoder, classifier, training_data, epochs, map_relid2tempid, test_data, seen_relations, steps):
    data_loader = get_data_loader(config, training_data, shuffle=True)
    encoder.train()
    classifier.train()

    optim_acc = 0.0
    criterion = CELoss(temperature=config.kl_temp)
    optimizer = optim.Adam([
        {'params': encoder.parameters(), 'lr': config.lr_encoder},
        {'params': classifier.parameters(), 'lr': config.lr_classifier}
    ])
    
    for epoch_i in range(epochs):
        losses = []
        for step, batch_data in enumerate(tqdm(data_loader)):
            optimizer.zero_grad()
            tokens, attention_mask, labels, task = batch_data
            labels = labels.to(config.device)
            labels = [map_relid2tempid[x.item()] for x in labels]
            
            labels = torch.tensor(labels).to(config.device)
            tokens = torch.stack([x.to(config.device) for x in tokens], dim=0)
            
            reps = encoder(tokens)
            logits = classifier(reps)
            loss = criterion(pros=logits, label=labels)    

            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            
        acc = evaluate_strict_model(config, encoder, classifier, test_data, seen_relations, map_relid2tempid)[0]
        
        print(f"{color_epoch}Epoch:{color_reset} {color_number}{epoch_i}{color_reset}," 
            + f"{color_loss}Loss:{color_reset} {color_number}{np.array(losses).mean()}{color_reset}," 
            + f"{color_epoch}Accuracy:{color_reset} {color_number}{acc}{color_reset}")


        # Get best model 
        if acc >= optim_acc:
            optim_acc = acc
            state_classifier = {
                'state_dict': classifier.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            path_save = f"Task_{steps}"            
            save_model(config, encoder, state_classifier, path_save, steps)

    return optim_acc



# def mahalanobis_distance(x, mu, Sigma):
#     delta = x - mu
#     inv_Sigma = np.linalg.inv(Sigma)
#     return np.sqrt(delta.T @ inv_Sigma @ delta)

# def nearest_mahalanobis_distance(x, mus, Sigmas):
#     distances = [mahalanobis_distance(x, mu, Sigma) for mu, Sigma in zip(mus, Sigmas)]
#     min_distance = min(distances)
#     min_index = distances.index(min_distance)
#     return min_distance, min_index



def compute_mean_and_covariance(data, all_mean, all_covariance, map_relid2tempid, seen_relations):
    encoder = Bert_Prompt(config=config, path_adapter=False).to(config.device)
    data_loader = get_data_loader(config, data, shuffle=True, batch_size=128)
    encoder.eval()
    mean, covariance, all_embeddings = [], [], []
    
    for step, batch_data in enumerate(tqdm(data_loader)):
        tokens, attention_mask, labels, task = batch_data
        labels = labels.to(config.device)
        labels = [map_relid2tempid[x.item()] for x in labels]
        labels = torch.tensor(labels).to(config.device)
        tokens = torch.stack([x.to(config.device) for x in tokens], dim=0)
        
        with torch.no_grad():
            outputs = encoder(tokens)

        all_embeddings.append(outputs.cpu().numpy())
    
    all_embeddings = np.concatenate(all_embeddings, axis=0)
    mean_embedding = np.mean(all_embeddings, axis=0)
    covariance_matrix = np.cov(all_embeddings, rowvar=False)
    
    all_mean.append(mean_embedding)
    all_covariance.append(covariance_matrix)
    return all_mean, all_covariance


def evaluate_strict_model(config, encoder, classifier, test_data, seen_relations, map_relid2tempid):
    if len(test_data) == 0:
        return 0, 0
    
    data_loader = get_data_loader(config, test_data, batch_size=1)
    encoder.eval()
    classifier.eval()
    n = len(test_data)

    correct = 0
    for step, batch_data in enumerate(data_loader):
        tokens, attention_mask, labels, task = batch_data
        labels = labels.to(config.device)
        labels = [map_relid2tempid[x.item()] for x in labels]
        labels = torch.tensor(labels).to(config.device)

        tokens = torch.stack([x.to(config.device) for x in tokens],dim=0)
        reps = encoder(tokens)
        logits = classifier(reps)

        seen_relation_ids = [rel2id[relation] for relation in seen_relations]
        seen_relation_ids = [map_relid2tempid[relation] for relation in seen_relation_ids]
        seen_sim = logits[:, seen_relation_ids].cpu().data.numpy()
        max_smi = np.max(seen_sim, axis=1)

        label_smi = logits[:,labels].cpu().data.numpy()

        if label_smi >= max_smi:
            correct += 1

    return correct/n, correct


# def evaluate_all_data(config, data_test, map_relid2tempid, seen_relations, mean, covariance):
#     encoder = Bert_Prompt(config=config, path_adapter=False).to(config.device)
#     for key, value in enumerate(data_test):
#         data_test_cur = data_test[value]
#         data_phase_two = []
#         total = len(data_test_cur)
#         correct = 0
        
#         data_loader = get_data_loader(config, data_test_cur, batch_size=1)
#         for step, batch_data in enumerate(data_loader):
#             tokens, attention_mask, labels, task = batch_data
#             labels = labels.to(config.device)
#             labels = [map_relid2tempid[x.item()] for x in labels]
#             labels = torch.tensor(labels).to(config.device)

#             tokens = torch.stack([x.to(config.device) for x in tokens],dim=0)
#             reps = np.array(encoder(tokens)[0].tolist())

#             for item in covariance:
#                 item = np.array(item)
                
#             for item in mean:
#                 item = np.array(item)

#             min_value, index = nearest_mahalanobis_distance(reps, mean, covariance)
#             # if index == labels:
#             #     data_phase_two.append(batch_data)
#     return data_phase_two



def evaluate_all_data(config, data_test, map_relid2tempid, seen_relations, mean, covariance):
    encoder = Bert_Prompt(config=config, path_adapter=False).to(config.device)
    mean = [torch.tensor(item, dtype=torch.float32, device=config.device) for item in mean]
    inv_covariance = [torch.inverse(torch.tensor(item, dtype=torch.float32, device=config.device)) for item in covariance]

    total = 0

    for key, data_test_cur in data_test.items():
        data_phase_two = []
        data_loader = get_data_loader(config, data_test_cur, batch_size=32) 
        total = len(data_test_cur)

        for step, batch_data in enumerate(data_loader):
            tokens, attention_mask, labels, task_ids = batch_data
            task_ids_tensor = torch.tensor(task_ids, dtype=torch.long, device=config.device)
            tokens = torch.cat([x.unsqueeze(0) for x in tokens], dim=0).to(config.device)
            reps = encoder(tokens)[0]

            for rep, task_id in zip(reps, task_ids_tensor):
                min_value, index = nearest_mahalanobis_distance(rep, mean, inv_covariance)
                # print(task_id, index)
                if index == task_id.item():  
                    data_phase_two.append(batch_data)  

        print(f"{len(data_phase_two)}/{total}")
    return data_phase_two


def nearest_mahalanobis_distance(rep, mus, inv_Sigmas):
    distances = [mahalanobis_distance(rep, mu, inv_Sigma) for mu, inv_Sigma in zip(mus, inv_Sigmas)]
    min_distance = min(distances)
    min_index = distances.index(min_distance)
    return min_distance, min_index


def mahalanobis_distance(x, mu, inv_Sigma):
    delta = x - mu
    return torch.sqrt(torch.matmul(torch.matmul(delta.T, inv_Sigma), delta))



param = Param()
args = param.args


# Device
torch.cuda.set_device(args.gpu)
args.device = torch.device(args.device)
args.n_gpu = torch.cuda.device_count()
args.task_name = args.dataname


if args.dataname == 'FewRel':
    args.rel_per_task = 8 
    args.num_class = 80
    args.max_length_passage = 1024
    args.batch_size = 32
    args.description_path = "./datasets/standard/description_fewrel.json" 
    
else:
    args.rel_per_task = 4
    args.num_class = 40
    args.batch_size = 16
    args.max_length_passage = 768
    args.description_path = "./datasets/standard/description_tacred.json" 

    
if __name__ == '__main__':
    config = args

    config.device = torch.device(config.device)
    config.n_gpu = torch.cuda.device_count()
    
    args_epi = get_args()
    fix_seed(args_epi.seed)
    epi = Bert_Prompt(config=config).to(config.device)
    
    
    for rou in range(config.total_round):
        random.seed(config.seed + rou*100)
        sampler = data_sampler(config, seed=config.seed + rou*100)
        id2rel = sampler.id2rel
        rel2id = sampler.rel2id
            
        
        num_class = len(sampler.id2rel)
        memorized_samples = []
        memory = collections.defaultdict(list)
        history_relations, list_map_relid2tempid = [], []
        history_data, prev_relations = [], []
        test_cur, test_total = [], []
        classifier = None
        relation_standard, description_class = {}, {}
        total_acc, all_test_data = [], []
        data_for_retrieval, list_retrieval = [], []
        description_original = json.load(open(config.description_path, 'r'))
        all_mean, all_covariance = [], []
        
        for sample in description_original:
            description_class[sample['relation']] = sample['text']
        test_per_task = {}
        
        for steps, (training_data, valid_data, test_data, current_relations, historic_test_data, seen_relations) in enumerate(sampler):
            prev_relations = history_relations[:]
            train_data_all_relation, test_data_all_relation = [], []
            test_per_task[steps] = []
            
            for relation in current_relations:
                history_relations.append(relation)

                # Remove data without entity tokens
                for item in training_data[relation]:
                    item['task'] = steps
                    if 30522 in item['tokens'] and 30523 in item['tokens'] and 30524 in item['tokens'] and 30525 in item['tokens']: 
                        train_data_all_relation.append(item)


                for item in test_data[relation]:
                    item['task'] = steps
                    if 30522 in item['tokens'] and 30523 in item['tokens'] and 30524 in item['tokens'] and 30525 in item['tokens']: 
                        test_data_all_relation.append(item)
                        test_per_task[steps].append(item)


            print(f'Current relation: {current_relations}')
            print(f"Task {steps}, Num class: {len(history_relations)}")
            print("----"*50)
            
            temp_rel2id = [rel2id[x] for x in history_relations]
            map_relid2tempid = {k: v for v, k in enumerate(temp_rel2id)}
            list_map_relid2tempid.append(map_relid2tempid) 
            
            classifier = Classifier_Layer(config=config, num_class=len(history_relations)).to(config.device)
            
            cur_acc = train_simple_model(
                config, 
                epi, 
                classifier, 
                train_data_all_relation, 
                config.classifier_epochs, 
                map_relid2tempid,
                test_data_all_relation,
                seen_relations, 
                steps,
            )
            
            all_mean, all_covariance = compute_mean_and_covariance(
                train_data_all_relation, 
                all_mean, 
                all_covariance,
                map_relid2tempid, 
                seen_relations
            )

            print(len(all_mean))
            print(len(all_covariance))
            torch.cuda.empty_cache()

            
            
            
            evaluate_all_data(config, test_per_task, map_relid2tempid, seen_relations, all_mean, all_covariance)
            # cur_acc = evaluate_all_data(config, test_per_task, history_relations, map_relid2tempid)
            # test_cur.append(cur_acc)
            # total_acc.append(cur_acc)


            torch.cuda.empty_cache()
            
            print('---'*23 + 'Evaluating!' + '---'*23 + '\n')
            print(f'Task--{steps}:')
            # print(f"Length train init: {len(train_data_for_initial)}")
            # print(f"Length test current task: {len(test_data_task)}")
            print(f'Current test acc: {cur_acc}')
            # print(f'Accuracy test all task: {test_cur}')
            # list_retrieval.append(evaluate_strict_all(config, steps, all_test_data, memorized_samples, list_map_relid2tempid, description_class, data_for_retrieval, id2rel, retrieval_path=retrieval_model))
            # print('---'*23 + f'Finish task {steps}!' + '---'*23 + '\n')
            
            # memorized_samples.append({
            #     'relations_task': current_relations,
            #     'data': this_task_memory,
            #     'task': len(memorized_samples)
            # })
            
            # json.dump(list_retrieval, open(config.output_kaggle + f'./task_{steps}.json', 'w'), ensure_ascii=False)
        
        
        if not os.path.exists(f'./results/task_{rou}'):
            os.makedirs(f'./results/task_{rou}')
        
        # mean_cor = {
        #     'mean': all_mean,
        #     'covariance': all_covariance,
        # }
        json.dump(list_retrieval, open(f'./results/task_{rou}/task_{steps}.json', 'w'), ensure_ascii=False)    
        # json.dump(mean_cor, open(f'./mean_covariance.json', 'w'), ensure_ascii=False)    e