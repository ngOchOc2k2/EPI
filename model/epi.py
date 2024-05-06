import torch
from torch import nn
from torch.nn import functional as F
from dataclasses import dataclass
from typing import Optional, Tuple
from transformers import AutoModel, AutoConfig
from transformers.utils import ModelOutput
from utils.assets import mahalanobis, get_prompts_data, mean_pooling


class EPI(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.config = AutoConfig.from_pretrained(args.model_name_or_path)
        self.model = AutoModel.from_pretrained(
            args.model_name_or_path, config=self.config).to(self.device)

        self.dropout = nn.Dropout(self.config.hidden_dropout_prob)


        # for prefix
        self.n_layer = self.config.num_hidden_layers
        self.n_head = self.config.num_attention_heads
        self.n_embd = self.config.hidden_size // self.config.num_attention_heads
        self.output_size = args.encoder_output_size
        self.query_size = self.config.hidden_size
        self.model.resize_token_embeddings(args.vocab_size + args.marker_size)
        

        # frozen bert model
        if self.args.frozen:
            for param in self.model.parameters():
                param.requires_grad = False


        self.num_labels = 0
        self.num_tasks = 0
        self.task_range = []


        self.task_means_over_classes = nn.ParameterList()
        self.accumulate_shared_covs = nn.Parameter(torch.zeros(
            self.query_size, self.query_size), requires_grad=False)
        self.cov_inv = nn.Parameter(torch.ones(
            self.query_size, self.query_size), requires_grad=False)
        self.prompts = nn.ParameterList()


    def get_prompts_data(self):
        # inherit previous prompts
        if self.args.prompt_fusion_mode == "mean" and self.num_tasks > -1:
            new_prompt_data = torch.mean(torch.stack(
                [prompt for prompt in self.prompts]), dim=0)
        elif self.args.prompt_fusion_mode == "last" and self.num_tasks > -1:
            # inherit last task prompt
            new_prompt_data = self.prompts[-1].data.clone()
        else:
            new_prompt_data = get_prompts_data(
                self.args, self.config, self.device)
        return new_prompt_data


    def new_task(self, num_labels):
        # frozen previous prompts
        for param in self.prompts.parameters():
            param.requires_grad = False

        prompts_data = self.get_prompts_data()
        self.prompts.append(nn.Parameter(prompts_data, requires_grad=True))
        self.num_tasks += 1
        self.task_range.append((self.num_labels, self.num_labels + num_labels))


    def new_statistic(self, mean, cov):
        self.task_means_over_classes.append(
            nn.Parameter(mean.cuda(), requires_grad=False))

        self.accumulate_shared_covs.data = self.accumulate_shared_covs.data.cpu()
        self.accumulate_shared_covs += cov

        self.cov_inv = nn.Parameter(torch.linalg.pinv(
            self.accumulate_shared_covs / self.num_tasks, hermitian=True), requires_grad=False)


    def get_prompt_indices(self, prelogits):
        """
        arguments:
            prelogits: [bs, hidden_size]
        return:
            indices: [bs], indices of selected prompts
        """

        scores_over_tasks = []
        for idx, mean_over_classes in enumerate(self.task_means_over_classes):
            num_labels, _ = mean_over_classes.shape
            score_over_classes = []
            for c in range(num_labels):
                if self.args.query_mode == "maha_ft":
                    score = mahalanobis(
                        prelogits[idx], mean_over_classes[c], self.cov_inv, norm=2)
                else:
                    raise NotImplementedError
                score_over_classes.append(score)
                
            # [num_labels, n]
            score_over_classes = torch.stack(score_over_classes)
            score, _ = score_over_classes.min(dim=0)

            scores_over_tasks.append(score)
        # [task_num, n]
        scores_over_tasks = torch.stack(scores_over_tasks, dim=0)
        _, indices = torch.min(scores_over_tasks, dim=0)

        return indices, scores_over_tasks


    def get_prelogits(
        self,
        indices,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        past_key_values=None
    ):
        
        print(attention_mask)
        bs = input_ids.shape[0]
        prompt_attention_mask = torch.ones(
            bs, self.args.pre_seq_len, dtype=torch.long, device=attention_mask.device)
        
        print(f"Indices: {indices}")
        prompt = torch.stack([self.prompts[idx] for idx in indices])

        attention_mask = torch.cat(
            [prompt_attention_mask, attention_mask], dim=1)

        bs, psl, hs = prompt.size()
        if self.args.prompt_mode == "prompt":
            prompt = prompt.view(bs, psl, hs)
            raw_embedding = self.model.embeddings(
                input_ids, position_ids, token_type_ids)
            inputs_embeds = torch.cat([prompt, raw_embedding], dim=1)
        elif self.args.prompt_mode == "prefix":
            past_key_values = prompt.view(
                bs, psl, self.n_layer * 2, self.n_head, self.n_embd)
            past_key_values = self.dropout(past_key_values)
            past_key_values = past_key_values.permute([2, 0, 3, 1, 4]).split(2)
            inputs_embeds = self.model.embeddings(
                input_ids, position_ids, token_type_ids)

        outputs = self.model(
            attention_mask=attention_mask,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            past_key_values=past_key_values,
        )

        if self.args.prompt_mode == "prefix":
            attention_mask = attention_mask[:, psl:]

        if self.args.rep_mode == "cls":
            prelogits = outputs[1]
        elif self.args.rep_mode == "avg":
            sequence_output = outputs[0]
            prelogits = mean_pooling(sequence_output, attention_mask)
        else:
            raise NotImplementedError
        prelogits = self.dropout(prelogits)
        return outputs, prelogits


    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        past_key_values=None,
        get_prelogits=False,
        get_scores=False,
        by_prefix=False,
        task_id=None,
        oracle=False,
    ):

        # for code readability
        args = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "token_type_ids": token_type_ids,
            "position_ids": position_ids,
            "head_mask": head_mask,
            "inputs_embeds": inputs_embeds,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "return_dict": return_dict,
            "past_key_values": past_key_values,
        }

        bs, _ = args["input_ids"].shape

        if self.training:
            indices = [self.num_tasks - 1] * bs
        else:
            if by_prefix:
                outputs, prelogits = self.get_prelogits(
                    [self.num_tasks - 1] * bs, **args)
            elif self.args.query_mode != "maha_ft":
                outputs = self.model(**args)
                last_hidden_state = outputs.last_hidden_state
                prelogits = mean_pooling(last_hidden_state, attention_mask)

            # return when get_centre is True
            if get_prelogits:
                return prelogits

            if self.args.query_mode == "maha_ft":
                prelogits = []
                for idx in range(self.num_tasks):
                    outputs, prelogit = self.get_prelogits(
                        [idx] * bs, **args)
                    prelogits.append(prelogit)
            indices, scores_over_tasks = self.get_prompt_indices(prelogits)
            print(f"Indices: {indices}")
            
            if get_scores:
                return scores_over_tasks

        outputs, pooled_output = self.get_prelogits(indices, **args)


        # Trích xuất token e11 và e21
        e11 = []
        e21 = []

        for i in range(input_ids.size()[0]):
            try:
                tokens = input_ids[i].cpu().numpy()
                e11.append(np.argwhere(tokens == 30522)[0][0])
                e21.append(np.argwhere(tokens == 30524)[0][0])
            except:
                print(input_ids[i])

        tokens_output = outputs.last_hidden_state # Token embeddings từ mạng encoder

        logits = []

        for i in range(len(e11)):
            instance_output = torch.index_select(tokens_output, 0, torch.tensor(i).cuda())
            instance_output = torch.index_select(instance_output, 1, torch.tensor([e11[i], e21[i]]).cuda())
            logits.append(instance_output) 

        logits = torch.cat(logits, dim=0)
        logits = output.view(logits.size()[0], -1) 


        return SequenceClassifierOutput(
            logits=output,
            prelogits=pooled_output,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

@dataclass
class SequenceClassifierOutput(ModelOutput):
    logits: torch.FloatTensor = None
    prelogits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


