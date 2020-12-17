# coding: UTF-8
import torch
import numpy as np
from pytorch_pretrain import BertModel, BertTokenizer


class TokenEncode(object):
    def __init__(self, bert_path, pad_size=64):
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
        # 根据vocab.txt将token列表转换成index列表
        # print(token)
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
        """
        这个函数的输出是编好码的bert输出，包含了两个tensor返回结果，用于最终确定的不需要训练参数的文字编码
        :param text:
        :return:
        """
        token_ids, seq_len, mask = self.get_token_li(text)
        # print(token_ids)
        # print(mask)
        token_ids = torch.tensor(token_ids).reshape(-1, self.pad_size).to(self.device)
        mask = torch.tensor(mask).reshape(-1, self.pad_size).to(self.device)
        # print(token_ids.shape)
        # print(mask)
        two_dim, pooled = self.bert(token_ids, attention_mask=mask, output_all_encoded_layers=False)
        return two_dim, pooled

    def get_token_mask(self, text):
        """
        这个寒素的输出分别是文字的列表和mask列表输出，这个用于bert-fine-tune
        :param text:
        :return:
        """
        token_ids, seq_len, mask = self.get_token_li(text)
        # print(token_ids)
        # print(mask)
        token_ids = torch.tensor(token_ids).reshape(-1, self.pad_size).to(self.device)
        mask = torch.tensor(mask).reshape(-1, self.pad_size).to(self.device)
        # print(token_ids.shape)
        # print(mask)
        return token_ids, mask

    def get_token_segment_mask(self, content1, content2):
        content1 = content1.strip()
        content2 = content2.strip()
        token1 = self.tokenizer.tokenize(content1)
        token1 = ['[CLS]'] + token1 + ['[SEP]']

        token2 = self.tokenizer.tokenize(content2)
        token2 = token2 + ['[SEP]']
        token = token1 + token2

        segment = [0] * len(token1) + [1] * len(token2)
        # 根据vocab.txt将token列表转换成index列表
        # print(token)
        token_ids = self.tokenizer.convert_tokens_to_ids(token)

        if len(token) < self.pad_size:
            mask = [1] * len(token_ids) + [0] * (self.pad_size - len(token))
            token_ids += ([0] * (self.pad_size - len(token)))
            segment += ([0] * (self.pad_size - len(token)))

        else:
            mask = [1] * self.pad_size
            token_ids = token_ids[:self.pad_size]
            segment = segment[:self.pad_size]
        # token_ids：padding后字符列表的index列表
        # label：标签的编号
        # seq_len：字符有多长，其长度是小于等于seq_len的
        # mask：和token_ids一样的长度，seq_len的前面为1，后面是零
        token_ids = torch.tensor(token_ids).reshape(-1, self.pad_size).to(self.device)
        mask = torch.tensor(mask).reshape(-1, self.pad_size).to(self.device)
        segment = torch.tensor(segment).reshape(-1, self.pad_size).to(self.device)
        return token_ids, segment, mask


if __name__ == '__main__':
    bert_path = 'D:/Work/Update_Everyday/Bert-Chinese-Text-Classification-Pytorch/bert_pretrain'
    pad_size = 32
    te = TokenEncode(bert_path, pad_size)
    a, b = te.get_encode('我是一只小可爱')
    print(a.shape, b.shape)
    q = torch.nn.AvgPool1d(3)
    q1 = q(a)
    print(q1.shape)
    print(a[:, 0])
    print(b)

    # a = tf.keras.layers.GlobalMaxPooling1D()(q_embedding)
    # t = q_embedding[:, -1]
    # e = q_embedding[:, 0]

    # c, d = te.get_token_mask('今天太阳很好呀')
    # print(c.shape, d.shape)
    # print(c, d)
    # e, f, g = te.get_token_segment_mask('我是一只小可爱', '今天太阳很好呀')
    # print(e.shape, f.shape, g.shape)
    # print(e, f, g)
