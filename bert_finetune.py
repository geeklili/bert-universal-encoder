import pandas as pd
import datetime
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from bert_encoder import TokenEncode
import torch.nn.functional as F
import numpy as np
from pytorch_pretrain import BertModel, BertTokenizer
from collections import defaultdict


EPOCH = 200
BATCH_SIZE = 10
PAD_SIZE = 256
DATA_PATH = './data/口碑数据.xlsx'
BERT_PATH = '../Bert-Chinese-Text-Classification-Pytorch/bert_pretrain'
LR = 0.00001
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

data = pd.read_excel(DATA_PATH, sheet_name=0)


def find_review_time_wheather_useful():
    def return_hour(line):
        line = line.replace(" ", '-')
        new_line = datetime.datetime.strptime(line, '%Y-%m-%d-%H:%M:%S')
        return new_line.hour

    di = defaultdict(list)
    score_li = data[['EVALUATION_SCORES']].values
    for ind, i in enumerate([return_hour(i[0].strip()[:-7]) for i in data[['REVIEW_TIME']].values]):
        di[i].append(int(score_li[ind][0]))
    di_score = dict()
    for a, b in di.items():
        di_score[a] = sum(b) / len(b)
    di_score = sorted(di_score.items(), key=lambda x: x[1])
    print(di_score)

# find_review_time_wheather_useful()
# new_data = data[['CONTENT','EVALUATION_SCORES']]
# new_data[data['EVALUATION_SCORES']==5]

# data[['TITLE', 'CONTENT', 'SHOP_NAME', 'MATRIX_IDS', 'BRAND', 'LINE', 'EVALUATOR_RANK', 'EVALUATION_SCORES']]

comment_li = [i[0] for i in data[['CONTENT','EVALUATION_SCORES']].values]
score_li = [int(i[1])-1 for i in data[['CONTENT','EVALUATION_SCORES']].values]
score_one_hot_li = torch.tensor(pd.get_dummies(data['EVALUATION_SCORES']).values).long()
score_two_li = [1 if j > 3 else 0 for j in score_li]
# 1. 创建bert语言模型的实例，用以获取token的向量
te = TokenEncode(BERT_PATH, PAD_SIZE)
# 2. 创建评论list
comment_li_encode = list()
for i in comment_li:
    a, b = te.get_token_mask(i)
    comment_li_encode.append([a.cpu().detach().numpy(), b.cpu().detach().numpy()])

# 3. 创建dataset于dataloader
class MyDateset(Dataset):
    def __init__(self, x, y):
        super(MyDateset, self).__init__()
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


# dataset = MyDateset(comment_li_encode, score_one_hot_li)
# dataset = MyDateset(comment_li_encode, score_li)
dataset = MyDateset(comment_li_encode, score_two_li)
data_iter = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False, num_workers=5)

# 4. 创建模型
class SentimentAnalysisModel(torch.nn.Module):
    def __init__(self, feature_size, hidden_size, output_size):
        super(SentimentAnalysisModel, self).__init__()
        # self.linear = torch.nn.Linear(feature_size, hidden_size)
        # self.linear2 = torch.nn.Linear(hidden_size, output_size)
        self.soft = torch.nn.Softmax()
        self.linear3 = torch.nn.Linear(feature_size, output_size)
        self.bert = BertModel.from_pretrained(BERT_PATH).to(device)
        for param in self.bert.parameters():
            param.requires_grad = True

    def forward(self, x):
        token_li = x[0]
        mask_li = x[1]
        token_li = torch.tensor(token_li).to(device).squeeze(dim=1)
        mask_li = torch.tensor(mask_li).to(device).squeeze(dim=1)
        _, pooled = self.bert(token_li, attention_mask=mask_li, output_all_encoded_layers=False)
        return self.soft(self.linear3(pooled))
        # return self.soft(self.linear2(self.linear(x)))


# 5. 模型，损失，优化器
model = SentimentAnalysisModel(768, 192, 2).to(device)
criterion = torch.nn.CrossEntropyLoss()
# opti = torch.optim.SGD(model.parameters(), lr=LR)
opti = torch.optim.Adam(model.parameters(), lr=LR)

# 6. 预测，计损，零反替
def evaluate_model(model):
    model.eval()
    pre_data = next(iter(data_iter))
    total = 0
    tp, tn, fp, fn = 0, 0, 0, 0
    for pre_data in data_iter:
        input_data = pre_data[0]
        predict = model(input_data)
        predict_li = [np.argmax(i.cpu().detach().numpy()) for i in predict.squeeze(dim=1)]
        for ind, pre in enumerate(predict_li):
            if pre == pre_data[1][ind].numpy():
                total += 1
                if pre_data[1][ind].numpy() == 0:
                    tn += 1
                else:
                    tp += 1
            else:
                if pre_data[1][ind].numpy() == 0:
                    fp += 1
                else:
                    fn += 1
    model.train()
    pacc = tp / (tp + fp)
    racc = tp / (tp + fn)
    f1 = 2 * pacc * racc / (pacc + racc)

    return pacc, racc, f1


def train_model():
    for ep in range(EPOCH):
        model.train()
        losses_li = []
        for ind, (x, y) in enumerate(data_iter):
            y = torch.tensor(y).to(device)
            pred = model(x)
            pred = pred.squeeze(dim=1)
            los = criterion(pred, y)
            model.zero_grad()
            los.backward()
            opti.step()
            losses_li.append(los.cpu().detach().numpy())
        pacc, racc, f1 = evaluate_model(model)
        line_str = '第{}个epoch loss: {}  准确率: {} 召回率: {} F1值：{}'.format(ep, np.mean(losses_li), pacc, racc, f1)
        print(line_str)


train_model()