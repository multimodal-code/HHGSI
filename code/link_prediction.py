from args import read_args
import numpy
import sklearn
import csv
from sklearn.metrics import roc_auc_score

args = read_args()


def load_data(data_file_name, n_features, n_samples):
	with open(data_file_name) as f:
		data_file = csv.reader(f)
		data = numpy.empty((n_samples, n_features))
		for i, d in enumerate(data_file):
			data[i] = numpy.asarray(d[1:], dtype=numpy.float)
	f.close

	return data


def model(sample_count):
	positive_feat_1_f = args.data_path + args.data_set + '/link_prediction/' + "positive_feature_1.txt"
	positive_feat_2_f = args.data_path + args.data_set + '/link_prediction/' + "positive_feature_2.txt"
	negative_feat_1_f = args.data_path + args.data_set + '/link_prediction/' + "negative_feature_1.txt"
	negative_feat_2_f = args.data_path + args.data_set + '/link_prediction/' + "negative_feature_2.txt"
	positive_feat_1_data = load_data(positive_feat_1_f, args.embed_d, int(sample_count / 2))
	positive_feat_2_data = load_data(positive_feat_2_f, args.embed_d, int(sample_count / 2))
	negative_feat_1_data = load_data(negative_feat_1_f, args.embed_d, int(sample_count / 2))
	negative_feat_2_data = load_data(negative_feat_2_f, args.embed_d, int(sample_count / 2))

	predict = numpy.empty(sample_count)
	label = numpy.empty(sample_count)

	for i in range(int(sample_count / 2)):
		vec1 = positive_feat_1_data[i]
		vec2 = positive_feat_2_data[i]
		cos_sim = vec1.dot(vec2) / (numpy.linalg.norm(vec1) * numpy.linalg.norm(vec2))
		predict[i] = cos_sim
		label[i] = 1
	for i in range(int(sample_count / 2)):
		vec1 = negative_feat_1_data[i]
		vec2 = negative_feat_2_data[i]
		cos_sim = vec1.dot(vec2) / (numpy.linalg.norm(vec1) * numpy.linalg.norm(vec2))
		predict[int(sample_count / 2) + i] = cos_sim
		label[int(sample_count / 2) + i] = 0

	auc = roc_auc_score(label, predict)
	print(auc)



if __name__ == '__main__':
	model(29756)
