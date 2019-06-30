# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 10:58:39 2019

@author: Youmin
"""

import numpy as np
import pandas as pd
from gensim.models import word2vec
from sklearn.preprocessing import normalize

filename = 'C:/Users/Youmin/Desktop/train_data.sample'


'''
样本数据制作部分
'''
data_ = []
with open(filename,'r') as f:
    
    lines = f.readlines()
    for line in lines:
        line = line.split(',')
        line[0] = int(line[0]) # Query_id
        
        line[1] = line[1].split(' ') # Query
        line.append(len(line[1])) # line[5] Query_length
        
        line[2] = int(line[2]) # Query title id
        
        line[3] = line[3].split(' ') # title
        line.append(len(line[3])) # line[6] title_length
        line[4] = int(line[4].replace('\t\n','')) #label以及换行缩进符清洗
        data_.append(line)
# 至此，数据包含特征如下6个，除标签
data = pd.DataFrame(data_,columns=['query_id', 'query', 'query_title_id', 'title','label','query_length', 'title_length'])

# 统计特征扩充，groupby扩展
# 统计1：以label和query长度为基础进行划分，统计不同点击情况下不同query长度对title_length的影响
grouped_length = data['title_length'].groupby([data['label'],data['query_length']])
groupby_mean_mat = grouped_length.mean().unstack()

# 统计2：基于query_id的点击率以label和query_id作为基础进行划分，看看不同id被统计的概率是多少
grouped_click_id = data['query_id'].groupby([data['label'],data['query_id']])
groupby_click_mat = grouped_click_id.count().unstack().fillna(0)# 同时进行缺省值处理
clicked = groupby_click_mat.loc[1] # 被点击的数量
groupby_click_sum = groupby_click_mat.sum()
prob_click_id = round(clicked /groupby_click_sum,3)# 保留小数点3位
def click_prob(x):
    return prob_click_id[x]
def group_sum(x):
    return groupby_click_sum[x]
data['click_prob'] = data['query_id'].map(click_prob)
data['group_sum'] = data['query_id'].map(group_sum)

# 这里是一个粗概率，目前不考虑用标准化处理，希望模型可以通过参数自动学习
data['0_prob_titlen'] = data['query_length'].map(lambda x: groupby_mean_mat[x][0])
data['1_prob_titlen'] = data['query_length'].map(lambda x: groupby_mean_mat[x][1])

    # 构建groupby表格，考虑对列做标准化处理，表格表示形式如下：
#    query_length        1          2          3    ...   55    198        300
#    label                                          ...                       
#    0             12.868735  13.179517  13.449843  ...  17.0  11.0  14.300000
#    1             13.296875  13.525405  13.538568  ...  12.0   4.0  13.666667
# 至此，统计特征制作完毕

'''
#基于embedding的距离特征
'''

from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
from scipy.linalg import norm
import gensim

# 1.杰卡德系数计算
def jaccard_similarity(s1,s2):
    def add_space(s):
        return ' '.join(list(s))
    s1,s2 = add_space(s1),add_space(s2)
    cv = CountVectorizer(tokenizer=lambda s: s.split())
    corpus = [s1,s2]
    vectors = cv.fit_transform(corpus).toarray()
    numerator = np.sum(np.min(vectors, axis = 0))
    denominator = np.sum(np.max(vectors, axis = 0))
    return 1.0* numerator/denominator

# 2.TF距离计算
def tf_similarity(s1, s2):
    def add_space(s):
        return ' '.join(list(s))
    
    s1, s2 = add_space(s1), add_space(s2)
    cv = CountVectorizer(tokenizer = lambda s: s.split())
    corpus = [s1,s2]
    vectors = cv.fit_transform(corpus).toarray()
    return np.dot(vectors[0], vectors[1])/(norm(vectors[0])*norm(vectors[1]))

# 3. TF-IDF计算
def tfidf_similarity(s1, s2):
    def add_space(s):
        return ' '.join(list(s))
    s1,s2 = add_space(s1),add_space(s2)
    cv = TfidfVectorizer(tokenizer=lambda s: s.split())
    corpus = [s1,s2]
    vectors = cv.fit_transform(corpus).toarray()
    return np.dot(vectors[0],vectors[1])/(norm(vectors[0])*norm(vectors[1]))  


# 4. word2vec计算
#导入模型
#model_file = 'C:/Users/Youmin/Desktop/w2vmodel_title'
#model = gensim.models.Word2Vec.load(model_file)
#
#def vector_similariy(s1,s2):
#    def sentence_vector(s):
#        words = 原来用的结巴.lcut(s)
#        v = np.zeros(64)
#        for word in words:
#            v += model[word]
#        v /= len(words)
#        return v
#    v1, v2 = sentence_vector(s1), sentence_vector(s2)
#    return np.dot(v1, v2) / (norm(v1)*norm(v2))


#group_dist = data['query_id'].groupby([data['label'],data['query_id']])
#group_dist_mat = group_dist.unstack()# 同时进行缺省值处理



def f(x):
    # 将标签为1的各个query_id找出来匹配上，输入的m是query_id的下标
    l = data[(data['query_id']==m)&(data['label']==1)].index.tolist() #这里l返回的是data中index下标
    jaccard_dist = []
    tf_dist = []
    tfidf_dist = []
    for i in range(len(l)):
        #jaccard
        j_dist = jaccard_similarity(x,data['title'][l[i]])
        jaccard_dist.append(j_dist)
        
        #TF
        TF_dist = tf_similarity(x,data['title'][l[i]])
        tf_dist.append(TF_dist)
        
        #TF-IDF
        TfIdf_dist = tfidf_similarity(x,data['title'][l[i]])
        tfidf_dist.append(TfIdf_dist)
        
    jaccard = np.mean(jaccard_dist)
    tf_ = np.mean(tf_dist)
    tfidf = np.mean(tfidf_dist)
    return jaccard,tf_,tfidf
    #返回三个距离均值
    
#因此这里的关键是找出m
temp = pd.Series()
for m in range(len(data['query_id'].groupby(data['query_id']).count())):
    m = m+1
    dist_seq = data['title'][(data['query_id']==m)].map(f)# 这是x
    temp = pd.concat([temp,dist_seq])
#    jac = dist_seq['title'].map(lambda x:x[0])
#    tf = dist_seq['title'].map(lambda x:x[1])
#    idf = dist_seq['title'].map(lambda x:x[2])
#    
#    pd.concat([jac,tf,idf],axis=1)
i = 0
data['jaccard_dist'] = temp.map(lambda x: x[i])
i = 1
data['tf_dist'] = temp.map(lambda x: x[i])
i = 2
data['tfidf_dist'] = temp.map(lambda x: x[i])




'''
自定义基于20000样本训练集生成语料库
def main(data):

    num_features = 300    # Word vector dimensionality
    min_word_count = 10   # Minimum word count
    num_workers = 16       # Number of threads to run in parallel
    context = 10          # Context window size
    downsampling = 1e-3   # Downsample setting for frequent words
    sentences = data['title']

    model = word2vec.Word2Vec(sentences, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sg = 1, sample = downsampling)
    model.init_sims(replace=True)
    # 保存模型，供日後使用
    model.save("C:/Users/Youmin/Desktop/w2vmodel_title")
    
    # 可以在加载模型之后使用另外的句子来进一步训练模型
    # model = gensim.models.Word2Vec.load('/tmp/mymodel')
    # model.train(more_sentences)
main(data)
'''

'''
样本特征制作部分
'''
# 构建语料库——考虑部分数据
from gensim import corpora
from gensim import models

# 设置构建语料库的数据集大小
texts = data['title']
dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
tfidf_corpus = models.TfidfModel(corpus)
# 至此语料库就是tfidf_corpus
# tfidf_corpus[xxxxx]
#def corpus_map(x):
#    l = dictionary.doc2bow(x)
#    return tfidf_corpus[l]
#data['title_corpus'] = data['title'].map(corpus_map)


# LDA corpus
# set the topic words components
num_topic = 5
from gensim.models.ldamodel import LdaModel
lda = LdaModel(corpus,num_topics=num_topic)
def lda_map(x):
    l = dictionary.doc2bow(x)
    return lda[l]
temp_ = data['title'].map(lda_map)
temp_ = pd.DataFrame(dict([ (k,pd.Series(v).map(list)) for k,v in temp_.iteritems() ])).fillna(0).T#
l_ = list(range(num_topic))
l = [str(x) for x in l_]
temp_.columns = l

for i in range(len(temp_)):
    temp_.iloc[i] = temp_.iloc[i].map(lambda x: x[1] if x!=0 else 0)

data = pd.concat([data,temp_],axis=1)


'''
模型测试部分lgb+nn双输入
'''
#import lightgbm as lgb
#import tensorflow as tf























