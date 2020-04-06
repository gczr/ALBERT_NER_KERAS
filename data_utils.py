# coding: utf-8
import json
import numpy as np
from keras.utils import to_categorical

#读取json配置文件，返回的type为字典
def read_config(config_file):
    """"读取配置"""
    with open(config_file) as json_file:
        config = json.load(json_file)
    return config

config=read_config('config.json')
tag_id=config['tags']
id_tag = {v:k for k,v in tag_id.items()}
MAX_SEQ_LEN=config['MAX_SEQ_LEN']
train_file_path=config['train_file_path']
dev_file_path=config['dev_file_path']
test_file_path=config['test_file_path']


# 将原始样本数据做初步处理
def read_data(file_path):
    # 读取数据集
    with open(file_path, "r", encoding="utf-8") as f:
        content = [line.strip() for line in f.readlines()]

    #根据空行判断每个句子的分割点，找出分割点存在index列表里
    index = [-1]
    #空行代表是一个句子的分割，一般行都是"相 O"形式，里面有一个空格；而空行只是一个""，没有空格
    index.extend([i for i, _ in enumerate(content) if ' ' not in _])
    index.append(len(content))
  
    # 根据上述找好的每句话分割点的index位置，一句一句找出word和对应的tag
    sentences, tags = [], []
    for j in range(len(index)-1):
        word, tag = [], []
        #一个sentence就代表一句话
        sentence = content[index[j]+1: index[j+1]]
        for line in sentence:
            word.append(line.split()[0])
            tag.append(line.split()[-1])
        #句子原文用字符串形式连在一起
        sentences.append(''.join(word))
        #一句话的所有tag组成一个list，最后再统一放到大的tags列表里
        tags.append(tag)

    # 去除空的句子及标注序列，一般放在末尾
    sentences = [s for s in sentences if s]
    tags = [t for t in tags if t]

    return sentences, tags


# 将数据最后包装成可灌入keras的形式
def data_convert(word_encoder,file_path):

    sentences, tags_all = read_data(file_path)
    print("sentences length: %s " % len(sentences))
    print("last sentence: ", sentences[-1])

    # ALBERT ERCODING
    print("start ALBERT encoding")
    x = np.array([word_encoder(s) for s in sentences])
    print("end ALBERT encoding")

    # 对y值统一长度为MAX_SEQ_LEN
    y_new = []
    for tags in tags_all:
        tag_list = [tag_id[tag] for tag in tags]
        if len(tags) < MAX_SEQ_LEN:
            tag_list = tag_list + [0] * (MAX_SEQ_LEN-len(tags))
        else:
            tag_list = tag_list[: MAX_SEQ_LEN]

        y_new.append(tag_list)

    # 将y中的元素编码成ont-hot encoding
    y = np.empty(shape=(len(tags_all), MAX_SEQ_LEN, len(tag_id.keys())+1))

    for i, seq in enumerate(y_new):
        y[i, :, :] = to_categorical(seq, num_classes=len(tag_id.keys())+1)

    return x, y


# 读取训练集数据,将标签转换成id,并存成json文件，目的就是先把所有的tags找出来，存成配置文件，供后续训练的时候使用
def label2id():
    train_sents, train_tags = read_data(train_file_path)
    # 标签转换成id，并保存成文件
    unique_tags = []
    for seq in train_tags:
        for _ in seq:
            if _ not in unique_tags:
                unique_tags.append(_)

    label_id_dict = dict(zip(unique_tags, range(1, len(unique_tags) + 1)))

    with open("tags.json", "w", encoding="utf-8") as g:
        g.write(json.dumps(label_id_dict, ensure_ascii=False, indent=2))
