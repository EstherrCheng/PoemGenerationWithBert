# -*- coding = utf-8 -*-
# @Time : 2022/11/26 13:37
# @Author : ChengYuYang
# @File : Bert.py
# @Software: PyCharm

# AIStudio中内置paddle稳定版环境，可以先不更新库
import warnings #忽略报错
warnings.filterwarnings('ignore')

import paddle
print(paddle.__version__)

import paddlenlp
test_dataset, dev_dataset, train_dataset = paddlenlp.datasets.load_dataset('poetry', splits=('test','dev','train'), lazy=False)
print('test_dataset 的样本数量：%d'%len(test_dataset))
print('dev_dataset 的样本数量：%d'%len(dev_dataset))
print('train_dataset 的样本数量：%d'%len(train_dataset))

print('单样本示例：%s'%test_dataset[0])

import re
def data_preprocess(dataset):
    for i, data in enumerate(dataset):
        dataset.data[i] = ''.join(list(dataset[i].values()))
        dataset.data[i] = re.sub('\x02', '', dataset[i])
    return dataset

# 开始处理
test_dataset = data_preprocess(test_dataset)
dev_dataset = data_preprocess(dev_dataset)
train_dataset = data_preprocess(train_dataset)
print('处理后的单样本示例：%s'%test_dataset[0])

from paddlenlp.transformers import BertTokenizer

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

# 处理效果展示
for poem in test_dataset[0:2]:
    token_poem, _ = bert_tokenizer.encode(poem).values()
    print(poem)
    print(token_poem)
    print(''.join(bert_tokenizer.convert_ids_to_tokens(token_poem)))

import paddle
from paddle.io import Dataset
import numpy as np


class PoemData(Dataset):
    """
    构造诗歌数据集，继承paddle.io.Dataset
    Parameters:
        poems (list): 诗歌数据列表，每一个元素为一首诗歌，诗歌未经编码
        max_len: 接收诗歌的最大长度
    """

    def __init__(self, poems, tokenizer, max_len=128):
        super(PoemData, self).__init__()
        self.poems = poems
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __getitem__(self, idx):
        line = self.poems[idx]
        token_line = self.tokenizer.encode(line)
        token, token_type = token_line['input_ids'], token_line['token_type_ids']
        if len(token) > self.max_len + 1:
            token = token[:self.max_len] + token[-1:]
            token_type = token_type[:self.max_len] + token_type[-1:]
        input_token, input_token_type = token[:-1], token_type[:-1]
        label_token = np.array((token[1:] + [0] * self.max_len)[:self.max_len], dtype='int64')
        # 输入填充
        input_token = np.array((input_token + [0] * self.max_len)[:self.max_len], dtype='int64')
        input_token_type = np.array((input_token_type + [0] * self.max_len)[:self.max_len], dtype='int64')
        input_pad_mask = (input_token != 0).astype('float32')
        return input_token, input_token_type, input_pad_mask, label_token, input_pad_mask

    def __len__(self):
        return len(self.poems)


from paddlenlp.transformers import BertModel, BertForTokenClassification
from paddle.nn import Layer, Linear, Softmax


class PoetryBertModel(Layer):
    """
    基于BERT预训练模型的诗歌生成模型
    """

    def __init__(self, pretrained_bert_model: str, input_length: int):
        super(PoetryBertModel, self).__init__()
        bert_model = BertModel.from_pretrained(pretrained_bert_model)
        self.vocab_size, self.hidden_size = bert_model.embeddings.word_embeddings.parameters()[0].shape
        self.bert_for_class = BertForTokenClassification(bert_model, self.vocab_size)
        # 生成下三角矩阵，用来mask句子后边的信息
        self.sequence_length = input_length
        # lower_triangle_mask为input_length * input_length的下三角矩阵（包含主对角线），该掩码作为注意力掩码的一部分（在forward的
        # 处理中为0的部分会被处理成无穷小量，以方便在计算注意力权重的时候保证被掩盖的部分权重约等于0）。而之所以写为下三角矩阵的形式，与
        # transformer的多头注意力计算的机制有关，细节可以了解相关论文获悉。
        self.lower_triangle_mask = paddle.tril(paddle.tensor.full((input_length, input_length), 1, 'float32'))

    def forward(self, token, token_type, input_mask, input_length=None):
        # 计算attention mask
        mask_left = paddle.reshape(input_mask, input_mask.shape + [1])
        mask_right = paddle.reshape(input_mask, [input_mask.shape[0], 1, input_mask.shape[1]])
        # 输入句子中有效的位置
        mask_left = paddle.cast(mask_left, 'float32')
        mask_right = paddle.cast(mask_right, 'float32')
        attention_mask = paddle.matmul(mask_left, mask_right)
        # 注意力机制计算中有效的位置
        if input_length is not None:
            # 之所以要再计算一次，是因为用于推理预测时，可能输入的长度不为实例化时设置的长度。这里的模型在训练时假设输入的
            # 长度是被填充成一致的——这一步不是必须的，但是处理成一致长度比较方便处理（对应地，增加了显存的用度）。
            lower_triangle_mask = paddle.tril(paddle.tensor.full((input_length, input_length), 1, 'float32'))
        else:
            lower_triangle_mask = self.lower_triangle_mask
        attention_mask = attention_mask * lower_triangle_mask
        # 无效的位置设为极小值
        attention_mask = (1 - paddle.unsqueeze(attention_mask, axis=[1])) * -1e10
        attention_mask = paddle.cast(attention_mask, self.bert_for_class.parameters()[0].dtype)

        output_logits = self.bert_for_class(token, token_type_ids=token_type, attention_mask=attention_mask)

        return output_logits


class PoetryBertModelLossCriterion(Layer):
    def forward(self, pred_logits, label, input_mask):
        loss = paddle.nn.functional.cross_entropy(pred_logits, label, ignore_index=0, reduction='none')
        masked_loss = paddle.mean(loss * input_mask, axis=0)
        return paddle.sum(masked_loss)

from paddle.static import InputSpec
from paddlenlp.metrics import Perplexity
from paddle.optimizer import AdamW

net = PoetryBertModel('bert-base-chinese', 128)

token_ids = InputSpec((-1, 128), 'int64', 'token')
token_type_ids = InputSpec((-1, 128), 'int64', 'token_type')
input_mask = InputSpec((-1, 128), 'float32', 'input_mask')
label = InputSpec((-1, 128), 'int64', 'label')

inputs = [token_ids, token_type_ids, input_mask]
labels = [label, input_mask]

model = paddle.Model(net, inputs, labels)
model.prepare(optimizer=AdamW(learning_rate=0.0001, parameters=model.parameters()), loss=PoetryBertModelLossCriterion(), metrics=[Perplexity()])

model.summary(inputs, [input.dtype for input in inputs])

from paddle.io import DataLoader

train_loader = DataLoader(PoemData(train_dataset, bert_tokenizer, 128), batch_size=128, shuffle=True)
dev_loader = DataLoader(PoemData(dev_dataset, bert_tokenizer, 128), batch_size=32, shuffle=True)
model.fit(train_data=train_loader, epochs=10, save_dir='./checkpoint', save_freq=1, verbose=1, eval_data=dev_loader, eval_freq=1)


import numpy as np

class PoetryGen(object):
    """
    定义一个自动生成诗句的类，按照要求生成诗句
    model: 训练得到的预测模型
    tokenizer: 分词编码工具
    max_length: 生成诗句的最大长度，需小于等于model所允许的最大长度
    """
    def __init__(self, model, tokenizer, max_length=512):
        self.model = model
        self.tokenizer = tokenizer
        self.puncs = ['，', '。', '？', '；']
        self.max_length = max_length

    def generate(self, style='', head='', topk=2):
        """
        根据要求生成诗句
        style (str): 生成诗句的风格，写成诗句的形式，如“大漠孤烟直，长河落日圆。”
        head (str, list): 生成诗句的开头内容。若head为str格式，则head为诗句开始内容；
            若head为list格式，则head中每个元素为对应位置上诗句的开始内容（即藏头诗中的头）。
        topk (int): 从预测的topk中选取结果
        """
        head_index = 0
        style_ids = self.tokenizer.encode(style)['input_ids']
        # 去掉结束标记
        style_ids = style_ids[:-1]
        head_is_list = True if isinstance(head, list) else False
        if head_is_list:
            poetry_ids = self.tokenizer.encode(head[head_index])['input_ids']
        else:
            poetry_ids = self.tokenizer.encode(head)['input_ids']
        # 去掉开始和结束标记
        poetry_ids = poetry_ids[1:-1]
        break_flag = False
        while len(style_ids) + len(poetry_ids) <= self.max_length:
            next_word = self._gen_next_word(style_ids + poetry_ids, topk)
            # 对于一些符号，如[UNK], [PAD], [CLS]等，其产生后对诗句无意义，直接跳过
            if next_word in self.tokenizer.convert_tokens_to_ids(['[UNK]', '[PAD]', '[CLS]']):
                continue
            if head_is_list:
                if next_word in self.tokenizer.convert_tokens_to_ids(self.puncs):
                    head_index += 1
                    if head_index < len(head):
                        new_ids = self.tokenizer.encode(head[head_index])['input_ids']
                        new_ids = [next_word] + new_ids[1:-1]
                    else:
                        new_ids = [next_word]
                        break_flag = True
                else:
                    new_ids = [next_word]
            else:
                new_ids = [next_word]
            if next_word == self.tokenizer.convert_tokens_to_ids(['[SEP]'])[0]:
                break
            poetry_ids += new_ids
            if break_flag:
                break
        return ''.join(self.tokenizer.convert_ids_to_tokens(poetry_ids))

    def _gen_next_word(self, known_ids, topk):
        type_token = [0] * len(known_ids)
        mask = [1] * len(known_ids)
        sequence_length = len(known_ids)
        known_ids = paddle.to_tensor([known_ids], dtype='int64')
        type_token = paddle.to_tensor([type_token], dtype='int64')
        mask = paddle.to_tensor([mask], dtype='float32')
        logits = self.model.network.forward(known_ids, type_token, mask, sequence_length)
        # logits中对应最后一个词的输出即为下一个词的概率
        words_prob = logits[0, -1, :].numpy()
        # 依概率倒序排列后，选取前topk个词
        words_to_be_choosen = words_prob.argsort()[::-1][:topk]
        probs_to_be_choosen = words_prob[words_to_be_choosen]
        # 归一化
        probs_to_be_choosen = probs_to_be_choosen / sum(probs_to_be_choosen)
        word_choosen = np.random.choice(words_to_be_choosen, p=probs_to_be_choosen)
        return word_choosen

# 载入已经训练好的模型
net = PoetryBertModel('bert-base-chinese', 128)
model = paddle.Model(net)
model.load('./checkpoint/final')
poetry_gen = PoetryGen(model, bert_tokenizer)

def poetry_show(poetry):
    pattern = r"([，。；？])"
    text = re.sub(pattern, r'\1 ', poetry)
    for p in text.split():
        if p:
            print(p)

# 随机生成一首诗
poetry = poetry_gen.generate()
poetry_show(poetry)

# 生成特定风格的诗
poetry = poetry_gen.generate(style='感时花溅泪，恨别鸟惊心。')
poetry_show(poetry)

# 生成特定开头的诗
poetry = poetry_gen.generate(head='秋风秋雨')
poetry_show(poetry)

# 生成藏头诗
poetry = poetry_gen.generate(head=['好', '想', '睡', '觉'])
poetry_show(poetry)