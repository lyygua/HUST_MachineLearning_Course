import pandas as pd
import numpy as np
import jieba as jb
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer

print("start")
df1 = pd.read_csv("dataset/train.csv") #读入训练集数据
df1.fillna('', inplace=True)

train_data = []  # 训练数据
train_label = []  # 训练标签

#训练集中评论进行分词
for i in range(0, df1.shape[0]):
    text = df1["content"][i] + df1["comment_all"][i]
    train_data.append(' '.join(jb.cut(text, cut_all=False)))
    train_label.append(df1["label"][i])

test_data = []  # 测试数据
df = pd.read_csv("dataset/test.csv") #读取测试数据
df.fillna('', inplace=True)

#测试集中评论进行分词
for i in range(0, df.shape[0]):
    text = df["content"][i] + df["comment_all"][i]
    test_data.append(' '.join(jb.cut(text, cut_all=False)))

#引入停用词
stop_words = []
with open('stop_words.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    for line in lines:
        stop_words.append(line.strip())

#数据类型转换，方便后续操作
train_data = np.array(train_data)
train_label = np.array(train_label)
test_data = np.array(test_data)

vec = CountVectorizer(analyzer='word', stop_words=stop_words)  # 实例化向量化对象
train_sample = vec.fit_transform(train_data)  # 将训练集中的新闻向量化
test_sample = vec.transform(test_data)  # 将测试集中的新闻向量化

tfidf = TfidfTransformer()  # 实例化tf-idf对象
train_sample = tfidf.fit_transform(train_sample)  # 将训练集中的词频向量用tf-idf进行转换
test_sample = tfidf.transform(test_sample)  # 将测试集中的词频向量用tf-idf进行转换

reg = LinearSVC(dual=False) # 调用模型 
reg.fit(train_sample, train_label) #模型学习
predict = reg.predict(test_sample) #模型预测

np.savetxt('result/result_svm.txt', predict, fmt="%d", delimiter="\n")
print("finish")

