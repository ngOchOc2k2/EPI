import os
import argparse
from typing import List

class Param:
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser = self.all_param(parser)
        all_args, unknown = parser.parse_known_args()
        self.args = all_args

    def all_param(self, parser):
        
        # common parameters
        parser.add_argument("--gpu", default=0, type=int)
        parser.add_argument("--dataname", default="TACRED", type=str, help="Use TACRED or FewRel datasets.")
        parser.add_argument("--task_name", default="TACRED", type=str)
        parser.add_argument("--device", default="cuda", type=str)

        # training parameters
        parser.add_argument("--batch_size", default=2, type=int)
        parser.add_argument("--rel_per_task", default=8)
        parser.add_argument("--pattern", default="entity_marker")
        parser.add_argument("--max_length", default=128, type=int)
        parser.add_argument("--encoder_output_size", default=768, type=int)
        parser.add_argument("--vocab_size", default=30522, type=int)
        parser.add_argument("--marker_size", default=4, type=int)
        parser.add_argument("--num_workers", default=0, type=int)
        parser.add_argument("--save_checkpoint", default="./checkpoint/", type=str)

        # epochs
        parser.add_argument("--classifier_epochs", default=1, type=int)

        # seed
        parser.add_argument("--seed", default=2021, type=int)
        

        # dataset path
        parser.add_argument("--data_path", default="./datasets/standard/", type=str)
        parser.add_argument("--no_relation", default="/negative_fewrel.json", type=str)
        parser.add_argument("--description_fewrel", default="/description_fewrel.json", type=str)


        # bert-base-uncased weights path
        parser.add_argument("--bert_path", default="bert-base-uncased", type=str)


        # classifier model config
        parser.add_argument("--drop_p", type=float, default=0.05)
        parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
        parser.add_argument("--total_round", type=int, default=5)
        parser.add_argument("--use_gpu", default=True, type=bool)
        parser.add_argument("--hidden_size", default=768, type=int)
        parser.add_argument("--lr_encoder", default=0.0001, type=float)
        parser.add_argument("--lr_classifier", default=0.001, type=float)
        parser.add_argument("--kl_temp", default=2, type=int)

        
        # lora params
        parser.add_argument("--rank_lora", default=8, type=int)
        
        
        # retrieval model configs
        parser.add_argument("--path_data_retrieval", default=None, type=str)
        parser.add_argument("--bge_model", default="BAAI/bge-m3", type=str)
        parser.add_argument("--description_path", default="./datasets/standard.json", type=str)
        parser.add_argument("--type_similar", default="dense", type=str)
        parser.add_argument("--num_protos", default=10, type=int)
        parser.add_argument("--top_k_negative", default=5, type=int)
        parser.add_argument("--top_k_retrieval", default=10, type=int)
        parser.add_argument("--output_kaggle", default='', type=str)
        parser.add_argument("--max_length_passage", default=512, type=int)
        parser.add_argument("--max_length_query", default=128, type=int)
        parser.add_argument("--colbert_vecs", default=False, type=bool)
        parser.add_argument("--dense_vecs", default=True, type=bool)
        parser.add_argument("--trainable_retrieval", default=True, type=bool)
        parser.add_argument("--description_type", default='single', type=str)
        parser.add_argument("--batch_size_retrieval", default=4, type=int)
        
        # description configs
        parser.add_argument("--use_example", default=True, type=bool)        

        
        # output kaggle 
        parser.add_argument("--checkpoint_kaggle", default='./checkpoint', type=str)
        

        return parser