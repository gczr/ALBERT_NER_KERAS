
# coding: utf-8

# In[7]:


import json
import numpy as np
import pandas as pd
from keras_contrib.layers import CRF
from keras_contrib.losses import crf_loss
from keras_contrib.metrics import crf_accuracy, crf_viterbi_accuracy
from keras.models import load_model
from collections import defaultdict
from pprint import pprint
pd.set_option('max_columns', 600)
pd.set_option('max_rows', 500)

from data_utils import read_data,data_convert,tag_id,MAX_SEQ_LEN,id_tag
from albert_zh.extract_feature import BertVector


# In[3]:


# 利用ALBERT提取文本特征
bert_model = BertVector(pooling_strategy="NONE", max_seq_len=MAX_SEQ_LEN)
word_encoder = lambda text: bert_model.encode([text])["encodes"][0]

# 载入模型
custom_objects = {'CRF': CRF, 'crf_loss': crf_loss, 'crf_viterbi_accuracy': crf_viterbi_accuracy}
model = load_model("albert_bilstm_crf.h5", custom_objects=custom_objects)


# In[ ]:


# 输入句子，一个句子一个句子进行预测
while 1:
    # 输入句子
    text = input("Please enter an sentence: ").replace(' ', '')
    # 利用训练好的模型进行预测
    train_x = np.array([word_encoder(text)])
    y = np.argmax(model.predict(train_x), axis=2)
    y = [id_tag[id] for id in y[0] if id]
    
    
    words=[w for w in text][:MAX_SEQ_LEN]
    tags=y[:len(words)]
    df=pd.DataFrame({'Sentence':words,'NER':tags})
    # 输出预测结果
    print(df)


