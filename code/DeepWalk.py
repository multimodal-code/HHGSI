# deepwalk generate node embedding
# inputs: het_random_walk.txt
# outputs: node_net_embedding.txt

import string;
import re;
import random
import math
import numpy as np
from gensim.models import Word2Vec
from args import read_args
from itertools import *
dimen = 128
window = 5

args = read_args()
data_set = args.data_set

def read_random_walk_corpus():
	walks = []
	walks_with_group = []
	walks_no_group = []
	inputfile = open("../data/social_image/" + data_set + "/" + "het_random_walk.txt", "r")
	inputfile_with_group = open("../data/social_image/" + data_set + "/" + "het_random_walk_with_group.txt", "r")
	inputfile_no_group = open("../data/social_image/" + data_set + "/" + "het_random_walk_no_group.txt", "r")
	for line in inputfile:
		path = []
		node_list=re.split(' ',line)
		for i in range(len(node_list)):
			path.append(node_list[i])
		walks.append(path)
	for line in inputfile_with_group:
		path = []
		node_list=re.split(' ',line)
		for i in range(len(node_list)):
			path.append(node_list[i])
		walks_with_group.append(path)
	for line in inputfile_no_group:
		path = []
		node_list=re.split(' ',line)
		for i in range(len(node_list)):
			path.append(node_list[i])
		walks_no_group.append(path)
	inputfile.close()
	inputfile_with_group.close()
	inputfile_no_group.close()
	return walks, walks_with_group, walks_no_group


walk_corpus, walk_corpus_with_group, walk_corpus_no_group = read_random_walk_corpus()
model = Word2Vec(walk_corpus, vector_size = dimen, window = window, min_count = 0, workers = 2, sg = 1, hs = 0, negative = 5)
model_with_group = Word2Vec(walk_corpus_with_group, vector_size = dimen, window = window, min_count = 0, workers = 2, sg = 1, hs = 0, negative = 5)
model_no_group = Word2Vec(walk_corpus_no_group, vector_size = dimen, window = window, min_count = 0, workers = 2, sg = 1, hs = 0, negative = 5)


print("Output...")
model.wv.save_word2vec_format("../data/social_image/" + data_set + "/" + "node_net_embedding.txt")
model_with_group.wv.save_word2vec_format("../data/social_image/" + data_set + "/" + "node_net_embedding_with_group.txt")
model_no_group.wv.save_word2vec_format("../data/social_image/" + data_set + "/" + "node_net_embedding_no_group.txt")

