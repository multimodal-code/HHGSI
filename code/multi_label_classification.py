import random
import string
import re
import numpy
from itertools import *
import sklearn
from sklearn import ensemble
from sklearn import neighbors
from sklearn import neural_network
from sklearn import linear_model
import sklearn.metrics as Metric
import csv
import argparse
from args import read_args

args = read_args()


def compute_mAP(labels, outputs):
	AP = []
	for i in range(labels.shape[0]):
		AP.append(sklearn.metrics.average_precision_score(labels[i], outputs[i]))
	return numpy.mean(AP)


def load_data(data_file_name, n_features, n_samples):
	with open(data_file_name) as f:
		data_file = csv.reader(f)
		data = numpy.empty((n_samples, n_features))
		for i, d in enumerate(data_file):
			data[i] = numpy.asarray(d[:], dtype=numpy.float)
	f.close

	return data

def count_data(data_file_name):
	count = 0
	with open(data_file_name) as f:
		data_file = csv.reader(f)
		for i, d in enumerate(data_file):
			count += 1
	f.close
	return count


def model(train_num, test_num, label_num, label_count):

	i_feature_train_f = args.data_path + args.data_set + '/multi_label_classification/' + "i_feature_train_alpha_" + str(int(args.loss_weight_alpha * 10)) + "_beta_" + str(int(args.loss_weight_beta * 10)) + ".txt"
	i_label_train_f = args.data_path + args.data_set + '/multi_label_classification/' + "i_label_train_alpha_" + str(int(args.loss_weight_alpha * 10)) + "_beta_" + str(int(args.loss_weight_beta * 10)) + ".txt"
	i_feature_test_f = args.data_path + args.data_set + '/multi_label_classification/' + "i_feature_test_alpha_" + str(int(args.loss_weight_alpha * 10)) + "_beta_" + str(int(args.loss_weight_beta * 10)) + ".txt"
	i_label_test_f = args.data_path + args.data_set + '/multi_label_classification/' + "i_label_test_alpha_" + str(int(args.loss_weight_alpha * 10)) + "_beta_" + str(int(args.loss_weight_beta * 10)) + ".txt"

	train_num = count_data(i_label_train_f)
	test_num = count_data(i_label_test_f)

	print(train_num, test_num)

	features_train_data = load_data(i_feature_train_f, args.embed_d + 1, train_num)
	label_train_data = load_data(i_label_train_f, label_num + 1, train_num)
	features_train = features_train_data.astype(numpy.float32)[:, 1:]
	label_train = label_train_data.astype(numpy.float32)[:, 1:]

	learner = neural_network.MLPClassifier(max_iter=5000)
	for i in label_train:
		for j in i:
			if j != 0 and j != 1:
				print(j, 'erro')
	learner.fit(features_train, label_train)
	train_features = None
	train_target = None

	print("training finish!")



	features_test_data = load_data(i_feature_test_f, args.embed_d + 1, test_num)
	label_test_data = load_data(i_label_test_f, label_num + 1, test_num)
	test_id = features_test_data.astype(numpy.int32)[:, 0]
	features_test = features_test_data.astype(numpy.float32)[:, 1:]
	label_test = label_test_data.astype(numpy.float32)[:, 1:]
	test_predict = learner.predict(features_test)

	print("test prediction finish!")

	print("Micro-P: ")
	print(sklearn.metrics.precision_score(label_test, test_predict, average='micro', zero_division=1))

	print("Micro-R: ")
	print(sklearn.metrics.recall_score(label_test, test_predict, average='micro', zero_division=1))

	print("MicroF1: ")
	print(sklearn.metrics.f1_score(label_test, test_predict, average='micro', zero_division=1))

	print("Macro-P: ")
	print(sklearn.metrics.precision_score(label_test, test_predict, average='macro', zero_division=1))

	print("Macro-R: ")
	print(sklearn.metrics.recall_score(label_test, test_predict, average='macro', zero_division=1))

	print("MacroF1: ")
	print(sklearn.metrics.f1_score(label_test, test_predict, average='macro', zero_division=1))

	print("MAP: ")
	map = compute_mAP(label_test, test_predict)
	print(map)


def model_baseline(train_num, test_num, label_num, label_count):
	i_feature_train_f = args.data_path + args.data_set + '/multi_label_classification/' + "i_feature_train_ALBEF.txt"
	i_label_train_f = args.data_path + args.data_set + '/multi_label_classification/' + "i_label_train_ALBEF.txt"
	i_feature_test_f = args.data_path + args.data_set + '/multi_label_classification/' + "i_feature_test_ALBEF.txt"
	i_label_test_f = args.data_path + args.data_set + '/multi_label_classification/' + "i_label_test_ALBEF.txt"

	train_num = count_data(i_label_train_f)
	test_num = count_data(i_label_test_f)

	print(train_num, test_num)

	features_train_data = load_data(i_feature_train_f, 256 + 1, train_num)
	label_train_data = load_data(i_label_train_f, label_num + 1, train_num)
	features_train = features_train_data.astype(numpy.float32)[:, 1:]
	label_train = label_train_data.astype(numpy.float32)[:, 1:]

	learner = neural_network.MLPClassifier(max_iter=5000)
	for i in label_train:
		for j in i:
			if j != 0 and j != 1:
				print(j, 'erro')
	learner.fit(features_train, label_train)

	print("training finish!")

	features_test_data = load_data(i_feature_test_f, 256 + 1, test_num)
	label_test_data = load_data(i_label_test_f, label_num + 1, test_num)
	features_test = features_test_data.astype(numpy.float32)[:, 1:]
	label_test = label_test_data.astype(numpy.float32)[:, 1:]
	test_predict = learner.predict(features_test)

	print("test prediction finish!")

	print("Micro-P: ")
	print(sklearn.metrics.precision_score(label_test, test_predict, average='micro', zero_division=1))

	print("Micro-R: ")
	print(sklearn.metrics.recall_score(label_test, test_predict, average='micro', zero_division=1))

	print("MicroF1: ")
	print(sklearn.metrics.f1_score(label_test, test_predict, average='micro', zero_division=1))

	print("Macro-P: ")
	print(sklearn.metrics.precision_score(label_test, test_predict, average='macro', zero_division=1))

	print("Macro-R: ")
	print(sklearn.metrics.recall_score(label_test, test_predict, average='macro', zero_division=1))

	print("MacroF1: ")
	print(sklearn.metrics.f1_score(label_test, test_predict, average='macro', zero_division=1))

	print("MAP: ")
	map = compute_mAP(label_test, test_predict)
	print(map)


if __name__ == '__main__':
	if args.data_set == 'CLEF':
		model(2188, 548, 99, 1)
	if args.data_set == 'MIR':
		model(2188, 548, 14, 1)
	if args.data_set == 'PASCAL':
		model(2188, 548, 20, 1)
	if args.data_set == 'NUS':
		model(2188, 548, 81, 1)
