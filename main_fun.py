# coding: UTF-8
import torch
import numpy as np
from pytorch_pretrain import BertModel, BertTokenizer


class TokenEncode(object):
    def __init__(self, bert_path, pad_size=32):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.bert_path = bert_path
        np.random.seed(1)
        torch.manual_seed(1)
        torch.cuda.manual_seed_all(1)
        torch.backends.cudnn.deterministic = True  # 保证每次结果一样
        self.bert = BertModel.from_pretrained(self.bert_path).to(self.device)
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.pad_size = pad_size
        for param in self.bert.parameters():
            param.requires_grad = True

    def get_token_li(self, content):
        content = content.strip()
        token = self.tokenizer.tokenize(content)
        token = ['[CLS]'] + token
        seq_len = len(token)
        mask = []
        # 根据vocab.txt将token列表转换成index列表
        print(token)
        token_ids = self.tokenizer.convert_tokens_to_ids(token)

        if len(token) < self.pad_size:
            mask = [1] * len(token_ids) + [0] * (self.pad_size - len(token))
            token_ids += ([0] * (self.pad_size - len(token)))
        else:
            mask = [1] * self.pad_size
            token_ids = token_ids[:self.pad_size]
            seq_len = self.pad_size
        # token_ids：padding后字符列表的index列表
        # label：标签的编号
        # seq_len：字符有多长，其长度是小于等于seq_len的
        # mask：和token_ids一样的长度，seq_len的前面为1，后面是零
        return token_ids, seq_len, mask

    def get_encode(self, text):
        token_ids, seq_len, mask = self.get_token_li(text)
        print(token_ids)
        print(mask)
        token_ids = torch.tensor(token_ids).reshape(-1, self.pad_size)
        mask = torch.tensor(mask).reshape(-1, self.pad_size)
        print(token_ids.shape)
        print(mask)
        two_dim, pooled = self.bert(token_ids, attention_mask=mask, output_all_encoded_layers=False)
        return two_dim, pooled


if __name__ == '__main__':
    te = TokenEncode('../Bert-Chinese-Text-Classification-Pytorch/bert_pretrain', 32)
    a, b = te.get_encode('我是一只小可爱a')
    print(a.shape, b.shape)
