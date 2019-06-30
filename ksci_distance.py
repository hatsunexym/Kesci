# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 09:23:19 2019

@author: Youmin
"""

# 相似度特征距离计算
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer
import numpy as np
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
导入模型
model_file = xxxxxx.bin路径
model = gensim.models.KeyedVectors.load_word2vec_format(model_file,binary=True)

def vector_similariy(s1,s2):
    def sentence_vector(s):
        words = 原来用的结巴.lcut(s)
        v = np.zeros(64)
        for word in words:
            v += model[word]
        v /= len(words)
        return v
    v1, v2 = sentence_vector(s1), sentence_vector(s2)
    return np.dot(v1, v2) / (norm(v1)*norm(v2))
    
    
    
    

    
