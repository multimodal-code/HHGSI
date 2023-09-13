import string
import re
import numpy as np
import os
import sys
import random
from itertools import *
import argparse
import json
from transformers import BertModel, BertTokenizer, BertConfig
from gensim.models.keyedvectors import KeyedVectors
import torch
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import multi_label_classification as MLC
import link_prediction as LP
import link_prediction_for_NUS as LPNUS
from args import read_args

def pca(feature, dim):
	p = PCA(n_components=dim)
	feature_add_dim = p.fit_transform(feature)[:, np.newaxis, :]
	feature_list = [i for i in feature_add_dim]
	return feature_list

def write_json(data,file_path):
    with open(file_path, 'w') as data_file:
        json.dump(data,data_file,indent=2)

def get_rel_json(file_path):
    if not os.path.exists(file_path):
        return {}
    with open(file_path) as data_file:
        return json.load(data_file)


args = read_args()

# multi-label classification
def multi_label_classification():

	i_feature_train_f = open(args.data_path + args.data_set + '/multi_label_classification/' + "i_feature_train_alpha_" + str(int(args.loss_weight_alpha * 10)) + "_beta_" + str(int(args.loss_weight_beta * 10)) + ".txt", "w")
	i_feature_test_f = open(args.data_path + args.data_set + '/multi_label_classification/' + "i_feature_test_alpha_" + str(int(args.loss_weight_alpha * 10)) + "_beta_" + str(int(args.loss_weight_beta * 10)) + ".txt", "w")
	embed_f = open(args.data_path + args.data_set + '/' + "node_embedding_alpha_" + str(int(args.loss_weight_alpha * 10)) + "_beta_" + str(int(args.loss_weight_beta * 10)) + ".txt", "r")

	available_index = []
	count = 0

	for line in islice(embed_f, 0, None):
		line = line.strip()
		node_id = re.split(' ', line)[0]
		index = int(node_id[1:])
		available_index.append(index)
		line = line.replace('i', '')
		if count % 5 != 0:
			i_feature_train_f.write(line.replace(' ', ',') + '\n')
		elif count % 10 != 0:
			i_feature_test_f.write(line.replace(' ', ',') + '\n')
		count += 1
	i_feature_train_f.close()
	i_feature_test_f.close()



	pretreatment_textCLEF_dir = '/home/hlf/code/social_image/dataset/relation/pretreatment_text' + args.data_set + '.json'
	name_index_dir = '/home/hlf/code/social_image/dataset/encodes/rel' + args.data_set + 'name2index.json'
	label_train_dir = args.data_path + args.data_set + "/multi_label_classification/i_label_train_alpha_" + str(int(args.loss_weight_alpha * 10)) + "_beta_" + str(int(args.loss_weight_beta * 10)) + ".txt"
	label_test_dir = args.data_path + args.data_set + "/multi_label_classification/i_label_test_alpha_" + str(int(args.loss_weight_alpha * 10)) + "_beta_" + str(int(args.loss_weight_beta * 10)) + ".txt"
	label_vocab = args.data_path + args.data_set + "/multi_label_classification/label_vocab_alpha_" + str(int(args.loss_weight_alpha * 10)) + "_beta_" + str(int(args.loss_weight_beta * 10)) + ".txt"

	label_vocab_file = open(label_vocab, 'w')
	label_train_file = open(label_train_dir, 'w')
	label_test_file = open(label_test_dir, 'w')

	with open(pretreatment_textCLEF_dir, 'r', encoding='utf-8') as json_file:
		data = json_file.read().encode(encoding='utf-8')
		pretreatment_textCLEF_dic = json.loads(data)
	with open(name_index_dir, 'r', encoding='utf-8') as json_file:
		data = json_file.read().encode(encoding='utf-8')
		name_index_dic = json.loads(data)

	labels = {}
	num = 0
	label_dic = {}
	for k, v in pretreatment_textCLEF_dic.items():
		for label in v['label']:
			if label not in labels:
				labels[label] = num
				str_txt = label + ' ' + str(num) + '\n'
				label_vocab_file.write(str_txt)
				num = num + 1

	for k, v in name_index_dic['id'].items():
		label_dic[k] = pretreatment_textCLEF_dic[k]['label']

	num = 0
	count = 0
	train_num = 0
	test_num = 0
	label_count = [0] * len(labels)
	for k, v in label_dic.items():
		label_list = [0] * len(labels)
		for label in v:
			label_list[labels[label]] = 1
			if num in available_index and count % 8 == 0:
				label_count[labels[label]] += 1
		for i in label_list:
			if i != 0 and i != 1:
				print(i)
		str_label = str(label_list)
		str_label = str_label.replace('[', '')
		str_label = str_label.replace(']', '')
		if num in available_index:
			if count % 5 != 0:
				label_train_file.write(str(num) + ',' + str_label + '\n')
				train_num += 1
			elif count % 10 != 0:
				label_test_file.write(str(num) + ',' + str_label + '\n')
				test_num += 1
			count += 1
		num += 1
	label_train_file.close()
	label_test_file.close()


	print("label_count: ", label_count)
	under_five_count = 0
	for i in label_count:
		if i < 4:
			under_five_count += 1
	print('under five count: ', under_five_count)

	return train_num, test_num, len(labels), label_count

def multi_label_classification_for_baseline():

	i_feature_train_f = open(args.data_path + args.data_set + '/multi_label_classification/'
	 + "i_feature_train_ALBEF.txt", "w")
	i_feature_test_f = open(args.data_path + args.data_set + '/multi_label_classification/'
	 + "i_feature_test_ALBEF.txt", "w")
	emb_dic_ALBEF_dir = args.data_path + args.data_set + '/multi_label_classification/' + "emb_dic_ALBEF.json"
	name_index_dir = '/home/hlf/code/social_image/dataset/encodes/rel' + args.data_set + 'name2index.json'
	emb_dic_ALBEF_dic = get_rel_json(emb_dic_ALBEF_dir)
	name_index_dic = get_rel_json(name_index_dir)
	
	nodeid2feature_dic = {}
	for k, v in emb_dic_ALBEF_dic.items():
		node_id = name_index_dic['id'][k]
		emb_str = str(v)
		emb_str = emb_str.replace('[', '')
		emb_str = emb_str.replace(']', '')
		nodeid2feature_dic[node_id] = emb_str
	nodeid2feature_list = sorted(nodeid2feature_dic.items(),key=lambda x:x[0])

	for id_f in nodeid2feature_list:
		if id_f[0] % 5 != 0:
			i_feature_train_f.write(str(id_f[0]) + ',' + id_f[1] + '\n')
		elif id_f[0] % 10 != 0:
			i_feature_test_f.write(str(id_f[0]) + ',' + id_f[1] + '\n')
	
	pretreatment_textCLEF_dir = '/home/hlf/code/social_image/dataset/relation/pretreatment_text' + args.data_set + '.json'
	label_train_dir = args.data_path + args.data_set + "/multi_label_classification/i_label_train_ALBEF.txt"
	label_test_dir = args.data_path + args.data_set + "/multi_label_classification/i_label_test_ALBEF.txt"
	label_vocab = args.data_path + args.data_set + "/multi_label_classification/label_vocab_ALBEF.txt"
	label_vocab_file = open(label_vocab, 'w')
	label_train_file = open(label_train_dir, 'w')
	label_test_file = open(label_test_dir, 'w')
	pretreatment_textCLEF_dic = get_rel_json(pretreatment_textCLEF_dir)

	labels = {}
	num = 0
	label_dic = {}
	for k, v in pretreatment_textCLEF_dic.items():
		for label in v['label']:
			if label not in labels:
				labels[label] = num
				str_txt = label + ' ' + str(num) + '\n'
				label_vocab_file.write(str_txt)
				num = num + 1

	for k, v in name_index_dic['id'].items():
		label_dic[k] = pretreatment_textCLEF_dic[k]['label']

	train_num = 0
	test_num = 0
	label_count = [0] * len(labels)
	for k, v in label_dic.items():
		node_id = name_index_dic['id'][k]
		label_list = [0] * len(labels)
		for label in v:
			label_list[labels[label]] = 1
		str_label = str(label_list)
		str_label = str_label.replace('[', '')
		str_label = str_label.replace(']', '')
		if node_id % 5 != 0:
			label_train_file.write(str(node_id) + ',' + str_label + '\n')
			train_num += 1
		elif node_id % 10 != 0:
			label_test_file.write(str(node_id) + ',' + str_label + '\n')
			test_num += 1
	label_train_file.close()
	label_test_file.close()

	return train_num, test_num, len(labels), label_count

def Link_prediction():
	embed_f = open(args.data_path + args.data_set + '/' + "node_embedding.txt", "r")
	het_neigh_f = open(args.data_path + args.data_set + '/' + "het_neigh_all.txt", "r")
	positive_sample_f = open(args.data_path + args.data_set + '/' + "link_prediction_positive_sample.txt", "r")

	positive_feat_1_f = open(args.data_path + args.data_set + '/link_prediction/' + "positive_feature_1.txt", "w")
	positive_feat_2_f = open(args.data_path + args.data_set + '/link_prediction/' + "positive_feature_2.txt", "w")
	negative_feat_1_f = open(args.data_path + args.data_set + '/link_prediction/' + "negative_feature_1.txt", "w")
	negative_feat_2_f = open(args.data_path + args.data_set + '/link_prediction/' + "negative_feature_2.txt", "w")
	embed_pkl_dic = {}
	het_neigh_dic = {}
	count = 0

	for line in islice(embed_f, 0, None):
		line = line.strip()
		node_id = re.split(' ', line)[0]
		index = int(node_id[1:])
		line = line.replace('i', '')
		embed_pkl_dic[index] = line.replace(' ', ',')


	for line in islice(het_neigh_f, 0, None):
		line = line.strip()
		node_id = re.split(':', line)[0]
		het_node_id = re.split(':', line)[1]
		index = int(node_id[1:])
		het_node_id = het_node_id.replace('i', '')
		het_neigh_list = list(map(int, re.split(',', het_node_id)))
		het_neigh_dic[index] = het_neigh_list


	for line in islice(positive_sample_f, 0, None):
		line = line.strip()
		id_1 = int(re.split(' ', line)[0])
		id_2 = int(re.split(' ', line)[1])
		feature_1 = embed_pkl_dic[id_1]
		count += 1
		positive_feat_1_f.write(embed_pkl_dic[id_1] + '\n')
		negative_feat_1_f.write(embed_pkl_dic[id_1] + '\n')
		feature_2 = embed_pkl_dic[id_2]
		positive_feat_2_f.write(embed_pkl_dic[id_2] + '\n')
		negative_id = random.randrange(0, args.I_n - 1, 1)
		while negative_id in het_neigh_dic[id_1] or negative_id not in embed_pkl_dic:
			negative_id = random.randrange(0, args.I_n - 1, 1)
		negative_feat_2_f.write(embed_pkl_dic[negative_id] + '\n')

	positive_feat_1_f.close()
	positive_feat_2_f.close()
	negative_feat_1_f.close()
	negative_feat_2_f.close()

	return count*2

def link_prediction_for_NUS():
	embed_f = open(args.data_path + args.data_set + '/' + "node_embedding_5.txt", "r")
	het_neigh_f = open(args.data_path + args.data_set + '/' + "het_neigh_all.txt", "r")
	positive_sample_f = open(args.data_path + args.data_set + '/link_prediction/' + "link_prediction_positive_sample.txt", "r")
	img_emb_clip_f = open(args.data_path + args.data_set + '/' + "img_emb_clip.txt", "r")
	embed_pkl_dir = args.data_path + args.data_set + '/link_prediction/' + "node_embedding_5.json"
	negative_sample_f = open(args.data_path + args.data_set + '/link_prediction/' + "link_prediction_negative_sample.txt", 'w')

	embed_dic = {}
	embed_clip_dic = {}
	het_neigh_dic = {}
	embed_pkl_dic = {}
	count = 0

	for line in islice(embed_f, 0, None):
		line = line.strip()
		node_id = re.split(' ', line)[0]
		index = int(node_id[1:])
		line = line.replace('i', '')
		embed_dic[index] = line.replace(' ', ',')

	for line in islice(img_emb_clip_f, 0, None):
		line = line.strip()
		node_id = re.split(' ', line)[0]
		index = int(node_id[:])
		embed_clip_dic[index] = line.replace(' ', ',')

	for line in islice(het_neigh_f, 0, None):
		line = line.strip()
		node_id = re.split(':', line)[0]
		het_node_id = re.split(':', line)[1]
		index = int(node_id[1:])
		het_node_id = het_node_id.replace('i', '')
		het_neigh_list = list(map(int, re.split(',', het_node_id)))
		het_neigh_dic[index] = het_neigh_list

	for line in islice(positive_sample_f, 0, None):
		line = line.strip()
		id_1 = int(re.split(' ', line)[0])
		id_2 = int(re.split(' ', line)[1])
		count += 1
		if id_1 not in embed_pkl_dic:
			try:
				embed_pkl_dic[id_1] = embed_dic[id_1]
			except KeyError:
				embed_pkl_dic[id_1] = embed_clip_dic[id_1]
		if id_2 not in embed_pkl_dic:
			try:
				embed_pkl_dic[id_2] = embed_dic[id_2]
			except KeyError:
				embed_pkl_dic[id_2] = embed_clip_dic[id_2]
		negative_id = random.randrange(0, args.I_n - 1, 1)
		while negative_id in het_neigh_dic[id_1] or negative_id not in embed_dic:
			negative_id = random.randrange(0, args.I_n - 1, 1)
		if negative_id not in embed_pkl_dic:
			embed_pkl_dic[negative_id] = embed_dic[negative_id]
		negative_sample_f.write(str(id_1) + ' ' + str(negative_id) + '\n')

	write_json(embed_pkl_dic, embed_pkl_dir)
	negative_sample_f.close()

	return count * 2 

def find_case():
	def string2float(embed_list):
		for i in range(len(embed_list)):
			embed_list[i] = float(embed_list[i])
		return np.array(embed_list)

	embed_f = open(args.data_path + args.data_set + '/' + "node_embedding.txt", "r")
	case_f = open(args.data_path + args.data_set + '/case/' + "case.txt", "w")
	embed_dic = {}
	index_list = []
	cos_sim_dic = {}
	for line in islice(embed_f, 0, None):
		line = line.strip()
		node_id = re.split(' ', line)[0]
		index = int(node_id[1:])
		index_list.append(index)
		line = line.replace('i', '')
		embed_list = line.split(' ')[1:]
		embed_dic[index] = string2float(embed_list)

	for i in index_list:
		c_emb = embed_dic[i]
		for j in index_list:
			p_emb = embed_dic[j]
			cos_sim = c_emb.dot(p_emb) / (np.linalg.norm(c_emb) * np.linalg.norm(p_emb))
			cos_sim_dic[j] = cos_sim
		cos_sim_list = sorted(cos_sim_dic.items(), key=lambda e:e[1])
		case_f.write(str(i) + '\n' + str(cos_sim_list[-10:]) + '\n' + str(cos_sim_list[:10]) + '\n\n')

	case_f.close()

	return 0



if __name__ == '__main__':
	print("------multi-label classification------")
	train_num, test_num, label_num, label_count = multi_label_classification()  # setup of author classification/clustering task
	MLC.model(train_num, test_num, label_num, label_count)
	print("------multi-label classification end------")

	# print("------multi-label classification for baseline------")
	# train_num, test_num, label_num, label_count = multi_label_classification_for_baseline()  # setup of author classification/clustering task
	# MLC.model_baseline(train_num, test_num, label_num, label_count)
	# print("------multi-label classification for baseline end------")

	# print("------link prediction------")
	# sample_count = Link_prediction()  # setup of author classification/clustering task
	# print(sample_count)
	# LP.model(sample_count)
	# print("------link prediction end------")

	# print("------link prediction for NUS------")
	# sample_count = link_prediction_for_NUS()  # setup of author classification/clustering task
	# print(sample_count)
	# LPNUS.model(sample_count)
	# print("------link prediction for NUS end------")


	# print("------find case------")
	# find_case()  # setup of author classification/clustering task
	# print("------find case end------")



