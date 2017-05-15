#!user/bin/env python3
# -*- coding: utf-8 -*-

'''
Created on 20170515
@author: JohnHuiWB
'''
import sys
sys.path.append("../")
from SynXNLP.classify.NBayes import NBayes

def load_data_set():
	posting_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
					['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
					['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
					['stop', 'posting', 'stupid', 'worthless', 'garbage'],
					['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
					['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']
					]
	class_vec = [0, 1, 0, 1, 0, 1]
	return posting_list, class_vec

def create_vocab_list(data_set):
	vocab_set = set([])
	for doc in data_set:
		vocab_set = vocab_set | set(doc)
	return list(vocab_set)

def set_of_word_2_vec(vocab_list, input_set):
	return_vec = [0]*len(vocab_list)
	for word in input_set:
		if word in vocab_list:
			return_vec[vocab_list.index(word)] = 1
	return return_vec

data, classes = load_data_set()
my = create_vocab_list(data)

trainset = []
for i in range(len(classes)):
	trainset.append(set_of_word_2_vec(my, data[i]))

C = NBayes()
C.train(trainset, classes)
for i in range(len(classes)):
	result = C.predict(trainset[i])
	print(result)