from args import read_args
import numpy
import sklearn
import csv
from sklearn.metrics import roc_auc_score
import json
import os
import re
from itertools import *

def get_rel_json(file_path):
    if not os.path.exists(file_path):
        return {}
    with open(file_path) as data_file:
        return json.load(data_file)

args = read_args()


def model(sample_count):
	positive_sample_dir = args.data_path + args.data_set + '/link_prediction/' + "link_prediction_positive_sample.txt"
	negative_sample_dir = args.data_path + args.data_set + '/link_prediction/' + "link_prediction_negative_sample.txt"
	emb_dir = args.data_path + args.data_set + '/link_prediction/' + "node_embedding_5.json"

	positive_sample_f = open(positive_sample_dir, 'r')
	negative_sample_f = open(negative_sample_dir, 'r')
	emb_dic = get_rel_json(emb_dir)

	predict = numpy.empty(sample_count)
	label = numpy.empty(sample_count)

	i = 0
	for line in islice(positive_sample_f, 0, None):
		line = line.strip()
		id_1 = int(re.split(' ', line)[0])
		id_2 = int(re.split(' ', line)[1])
		vec1_list = re.split(',', emb_dic[str(id_1)])[1:]
		vec2_list = re.split(',', emb_dic[str(id_2)])[1:]
		vec1 = numpy.asarray(vec1_list, dtype=numpy.float)
		vec2 = numpy.asarray(vec2_list, dtype=numpy.float)
		cos_sim = vec1.dot(vec2) / (numpy.linalg.norm(vec1) * numpy.linalg.norm(vec2))
		predict[i] = cos_sim
		label[i] = 1
		i += 1

	i = 0
	for line in islice(negative_sample_f, 0, None):
		line = line.strip()
		id_1 = int(re.split(' ', line)[0])
		id_2 = int(re.split(' ', line)[1])
		vec1_list = re.split(',', emb_dic[str(id_1)])[1:]
		vec2_list = re.split(',', emb_dic[str(id_2)])[1:]
		vec1 = numpy.asarray(vec1_list, dtype=numpy.float)
		vec2 = numpy.asarray(vec2_list, dtype=numpy.float)
		cos_sim = vec1.dot(vec2) / (numpy.linalg.norm(vec1) * numpy.linalg.norm(vec2))
		predict[int(sample_count / 2) + i] = cos_sim
		label[int(sample_count / 2) + i] = 0
		i += 1

	auc = roc_auc_score(label, predict)
	print(auc)

if __name__ == '__main__':
	model(650940)
