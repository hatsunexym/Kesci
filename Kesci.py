# -*- coding: utf-8 -*-
"""
Created on Fri Jun 21 09:36:50 2019

@author: Youmin
"""

# train -> training dataframe
# test -> test dataframe
import pandas as pd
import numpy as np

data = pd.read_csv('C:/Users/Youmin/Desktop/train_data.csv')
data.drop(data.columns[0], axis = 1, inplace = True)

label = data['label'].values.reshape(len(label),-1)

'''
特征组合部分
groupby
df.groupby('query_id').sum()/count()/mean()
'''



'''
内容向量化部分
'''


'''
模型训练部分
'''


'''
模型验证部分
'''


'''
模型测试部分
'''