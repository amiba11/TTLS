# -*- coding: utf-8 -*-
# @Time   : 2020/8/17 19:38
# @Author : Yujie Lu
# @Email  : yujielu1998@gmail.com

# UPDATE:
# @Time   : 2020/8/19, 2020/10/2
# @Author : Yupeng Hou, Yujie Lu
# @Email  : houyupeng@ruc.edu.cn, yujielu1998@gmail.com

r"""
GRU4Rec
################################################

Reference:
    Yong Kiam Tan et al. "Improved Recurrent Neural Networks for Session-based Recommendations." in DLRS 2016.

"""

import torch
from torch import nn
from torch.nn.init import xavier_uniform_, xavier_normal_

from recbole.model.abstract_recommender import SequentialRecommender
from recbole.model.loss import BPRLoss
import torch.nn.functional as F

class GRU4Rec(SequentialRecommender):
    r"""GRU4Rec is a model that incorporate RNN for recommendation.

    Note:

        Regarding the innovation of this article,we can only achieve the data augmentation mentioned
        in the paper and directly output the embedding of the item,
        in order that the generation method we used is common to other sequential models.
    """

    def __init__(self, config, dataset):
        super(GRU4Rec, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config["emb_size"]
        self.hidden_size = config["hidden_size"]
        self.loss_type = config["loss_type"]
        self.num_layers = config["num_layers"]
        self.dropout_prob = config["dropout_prob"]

        # define layers and loss
        self.item_embedding = nn.Embedding(
            self.n_items, self.embedding_size, padding_idx=0
        )
        self.emb_dropout = nn.Dropout(self.dropout_prob)
        self.gru_layers = nn.GRU(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=False,
            batch_first=True,
        )
        self.dense = nn.Linear(self.hidden_size, self.embedding_size)
        if self.loss_type == "BPR":
            self.loss_fct = BPRLoss()
        elif self.loss_type == "CE":
            self.loss_fct = nn.CrossEntropyLoss()
        else:
            raise NotImplementedError("Make sure 'loss_type' in ['BPR', 'CE']!")

        # parameters initialization
        self.apply(self._init_weights)
        self.train_user_seq = {}
        self.eval_user_seq = {}
        self.train_user_emb = {}
        self.eval_user_emb = {}
        self.train_idx = 0
        self.eval_idx = 0

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            xavier_normal_(module.weight)
        elif isinstance(module, nn.GRU):
            xavier_uniform_(module.weight_hh_l0)
            xavier_uniform_(module.weight_ih_l0)
    
    def get_train_emb(self,item_seq, user_idx):
        item_seq_emb = self.item_embedding(item_seq)
        user_emb = torch.mean(item_seq_emb, dim=1)
        user_emb = item_seq_emb.detach().cpu().numpy()
        user_seq = item_seq.detach().cpu().numpy()
        for i in range(len(user_idx)):
            idx = user_idx[i].detach().cpu().item()
            tmp_user_emb = user_emb[i]
            self.train_user_emb[idx] = tmp_user_emb
            self.train_user_seq[idx] = user_seq[i]
        

    def get_eval_emb(self,item_seq, user_idx):
        item_seq_emb = self.item_embedding(item_seq)
        user_emb = torch.mean(item_seq_emb, dim=1)
        user_emb = item_seq_emb.detach().cpu().numpy()
        user_seq = item_seq.detach().cpu().numpy()
        for i in range(len(user_idx)):
            idx = user_idx[i].detach().cpu().item()
            tmp_user_emb = user_emb[i]
            self.eval_user_emb[idx] = tmp_user_emb
            self.eval_user_seq[idx] = user_seq[i]


    def forward(self, item_seq, item_seq_len):
        item_seq_emb = self.item_embedding(item_seq)
        item_seq_emb_dropout = self.emb_dropout(item_seq_emb)
        gru_output, _ = self.gru_layers(item_seq_emb_dropout)
        gru_output = self.dense(gru_output)
        # the embedding of the predicted item, shape of (batch_size, embedding_size)
        seq_output = self.gather_indexes(gru_output, item_seq_len - 1)
        return seq_output

    def calculate_loss(self, interaction):  #!!!!
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        pos_items = interaction[self.POS_ITEM_ID]
        user_idx = interaction[self.USER_ID]

        self.get_train_emb(item_seq, user_idx)

        # 计算训练集的用户表征
        
        if self.loss_type == "BPR":
            neg_items = interaction[self.NEG_ITEM_ID]
            pos_items_emb = self.item_embedding(pos_items)
            neg_items_emb = self.item_embedding(neg_items)
            pos_score = torch.sum(seq_output * pos_items_emb, dim=-1)  # [B]
            neg_score = torch.sum(seq_output * neg_items_emb, dim=-1)  # [B]
            loss = self.loss_fct(pos_score, neg_score)
            return loss
        else:  # self.loss_type = 'CE'
            test_item_emb = self.item_embedding.weight
            logits = torch.matmul(seq_output, test_item_emb.transpose(0, 1))

            entropy_loss = -torch.mean(torch.sum(F.log_softmax(logits, 1) * F.softmax(logits,1), -1))
            loss = self.loss_fct(logits, pos_items)
            return loss * 0.75 + entropy_loss * 0.25
        
            # loss = self.loss_fct(logits, pos_items)
            # return loss 

    ###################################################################################################
    def entropy_loss(self, rating):
        entropy = -torch.mean(torch.sum(F.log_softmax(rating, 1) * F.softmax(rating,1), -1))
        return entropy

    ###################################################################################################

    def predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        test_item = interaction[self.ITEM_ID]
        seq_output = self.forward(item_seq, item_seq_len)
        test_item_emb = self.item_embedding(test_item)
        scores = torch.mul(seq_output, test_item_emb).sum(dim=1)  # [B]
        return scores

    # def full_sort_predict(self, interaction):
    #     item_seq = interaction[self.ITEM_SEQ]
    #     item_seq_len = interaction[self.ITEM_SEQ_LEN]
    #     user_idx = interaction[self.USER_ID]

    #     self.get_eval_emb(item_seq, user_idx)

    #     seq_output = self.forward(item_seq, item_seq_len)
    #     real_items = torch.arange(0, self.n_items).to(self.device)
    #     test_items_emb = self.item_embedding(real_items)
    #     scores = torch.matmul(
    #         seq_output, test_items_emb.transpose(0, 1)
    #     )  # [B, n_items]
    #     return scores

    def full_sort_predict(self, interaction):
        item_seq = interaction[self.ITEM_SEQ]
        item_seq_len = interaction[self.ITEM_SEQ_LEN]
        seq_output = self.forward(item_seq, item_seq_len)
        test_items_emb = self.item_embedding.weight
        scores = torch.matmul(
            seq_output, test_items_emb.transpose(0, 1)
        )  # [B, n_items]
        return scores
