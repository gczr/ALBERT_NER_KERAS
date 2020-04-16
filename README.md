### 利用ALBERT+BiLSTM+CRF模型实现序列标注算法
需要安装keras-contrib
pip install git+https://www.github.com/keras-team/keras-contrib.git

### 数据集

人民日报语料集，实体为人名、地名、组织机构名，数据集位于data/example.*;

### 说明
config.json：配置文件

MAX_SEQ_LEN：为albert的最大输入序列长度

tags：实体标注符号，最好从1开始，不要从0开始编码。

albert_model_train：模型训练脚本

albert_model_prdict：模型预测脚本
