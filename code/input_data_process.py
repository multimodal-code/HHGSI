import six.moves.cPickle as pickle
import numpy as np
import argparse
import string
import re
import random
import math
from collections import Counter
from itertools import *
from args import read_args

commen_args = read_args()

parser = argparse.ArgumentParser(description = 'input data process')
parser.add_argument('--data_set', type = str, default = commen_args.data_set,
			   help = 'Data set utilized')
parser.add_argument('--I_n', type=int, default = commen_args.I_n,
					help='number of image node')
parser.add_argument('--data_path', type = str, default = '../data/social_image/',
				   help='path to data')
parser.add_argument('--walk_n', type = int, default = 10, 
			   help='number of walk per root node')
parser.add_argument('--walk_L', type = int, default = 30, 
			   help='length of each walk')
parser.add_argument('--window', type = int, default = 7,
			   help='window size for relation extration')
parser.add_argument('--T_split', type = int, default = 2012,
			   help = 'split time of train/test data')


args = parser.parse_args()
print(args)


class input_data(object):
	def __init__(self, args):
		self.args = args

		i_i_list = [[] for k in range(self.args.I_n)]
		i_i_with_group_list = [[] for k in range(self.args.I_n)]
		i_i_no_group_list = [[] for k in range(self.args.I_n)]

		relation_f = ["i_i_list.txt", "i_i_with_group_list.txt", "i_i_no_group_list.txt"]

		for i in range(len(relation_f)):
			f_name = relation_f[i]
			neigh_f = open(self.args.data_path + self.args.data_set + '/' + f_name, "r")
			for line in neigh_f:
				line = line.strip()
				node_id = int(re.split(':', line)[0])
				if node_id == 4:
					hh = 100
				neigh_list = re.split(':', line)[1]
				neigh_list_id = re.split(',', neigh_list)
				if f_name == 'i_i_list.txt':
					if neigh_list_id[0] != '':
						for j in range(len(neigh_list_id)):
							try:
								i_i_list[node_id].append('i' + str(neigh_list_id[j]))
							except IndexError:
								i_i_list[node_id].append('i' + str(neigh_list_id[j]))
				if f_name == 'i_i_with_group_list.txt':
					if neigh_list_id[0] != '':
						for j in range(len(neigh_list_id)):
							i_i_with_group_list[node_id].append('i' + str(neigh_list_id[j]))
				if f_name == 'i_i_no_group_list.txt':
					if neigh_list_id[0] != '':
						for j in range(len(neigh_list_id)):
							i_i_no_group_list[node_id].append('i' + str(neigh_list_id[j]))
			neigh_f.close()

		self.i_i_list = i_i_list
		self.i_i_with_group_list = i_i_with_group_list
		self.i_i_no_group_list = i_i_no_group_list



	def gen_het_rand_walk(self):
		het_walk_f = open(self.args.data_path + self.args.data_set + '/' + "het_random_walk.txt", "w")
		het_walk_with_group_f = open(self.args.data_path + self.args.data_set + '/' + "het_random_walk_with_group.txt", "w")
		het_walk_no_group_f = open(self.args.data_path + self.args.data_set + '/' + "het_random_walk_no_group.txt", "w")
		for i in range(self.args.walk_n):
			for j in range(self.args.I_n):
				if j == 4:
					hh = 100
				if len(self.i_i_list[j]):
					curNode = "i" + str(j)
					het_walk_f.write(curNode + " ")
					for l in range(self.args.walk_L - 1):
						if curNode[0] == 'i':
							curNode = int(curNode[1:])
							curNode = random.choice(self.i_i_list[curNode])
							het_walk_f.write(curNode + " ")
					het_walk_f.write("\n")
				if len(self.i_i_with_group_list[j]):
					curNode = "i" + str(j)
					het_walk_with_group_f.write(curNode + " ")
					for l in range(self.args.walk_L - 1):
						if curNode[0] == 'i':
							curNode = int(curNode[1:])
							curNode = random.choice(self.i_i_with_group_list[curNode])
							het_walk_with_group_f.write(curNode + " ")
					het_walk_with_group_f.write("\n")
				if len(self.i_i_no_group_list[j]):
					curNode = "i" + str(j)
					het_walk_no_group_f.write(curNode + " ")
					for l in range(self.args.walk_L - 1):
						if curNode[0] == 'i':
							curNode = int(curNode[1:])
							curNode = random.choice(self.i_i_no_group_list[curNode])
							het_walk_no_group_f.write(curNode + " ")
					het_walk_no_group_f.write("\n")
		het_walk_f.close()
		het_walk_with_group_f.close()
		het_walk_no_group_f.close()


	def gen_meta_rand_walk_APVPA(self):
		meta_walk_f = open(self.args.data_path + self.args.data_set + '/' + "meta_random_walk_APVPA_test.txt", "w")
		for i in range(self.args.walk_n):
			for j in range(self.args.A_n):
				if len(self.a_p_list_train[j]):
					curNode = "a" + str(j)
					preNode = "a" + str(j)
					meta_walk_f.write(curNode + " ")
					for l in range(self.args.walk_L - 1):
						if curNode[0] == "a":
							preNode = curNode
							curNode = int(curNode[1:])
							curNode = random.choice(self.a_p_list_train[curNode])
							meta_walk_f.write(curNode + " ")
						elif curNode[0] == "p":
							curNode = int(curNode[1:])
							if preNode[0] == "a":
								preNode = "p" + str(curNode)
								curNode = "p" + str(self.p_v[curNode])
								meta_walk_f.write(curNode + " ")
							else:
								preNode = "p" + str(curNode)
								curNode = random.choice(self.p_neigh_list_train[curNode])
								meta_walk_f.write(curNode + " ")
						elif curNode[0] == "v": 
							preNode = curNode
							curNode = int(curNode[1:])
							curNode = random.choice(self.v_p_list_train[curNode])
							meta_walk_f.write(curNode + " ")
					meta_walk_f.write("\n")
		meta_walk_f.close()



	def a_a_collaborate_train_test(self):
		a_a_list_train = [[] for k in range(self.args.A_n)]
		a_a_list_test = [[] for k in range(self.args.A_n)]
		p_a_list = [self.p_a_list_train, self.p_a_list_test]
		
		for t in range(len(p_a_list)):
			for i in range(len(p_a_list[t])):
				for j in range(len(p_a_list[t][i])):
					for k in range(j+1, len(p_a_list[t][i])):
						if t == 0:
							a_a_list_train[int(p_a_list[t][i][j][1:])].append(int(p_a_list[t][i][k][1:]))
							a_a_list_train[int(p_a_list[t][i][k][1:])].append(int(p_a_list[t][i][j][1:]))
						else:
							if len(a_a_list_train[int(p_a_list[t][i][j][1:])]) and len(a_a_list_train[int(p_a_list[t][i][k][1:])]):
								if int(p_a_list[t][i][k][1:]) not in a_a_list_train[int(p_a_list[t][i][j][1:])]:
									a_a_list_test[int(p_a_list[t][i][j][1:])].append(int(p_a_list[t][i][k][1:]))
								if int(p_a_list[t][i][j][1:]) not in a_a_list_train[int(p_a_list[t][i][k][1:])]:
									a_a_list_test[int(p_a_list[t][i][k][1:])].append(int(p_a_list[t][i][j][1:]))
		

		for i in range(self.args.A_n):
			a_a_list_train[i]=list(set(a_a_list_train[i]))
			a_a_list_test[i]=list(set(a_a_list_test[i]))

		a_a_list_train_f = open(args.data_path + "a_a_list_train.txt", "w")
		a_a_list_test_f = open(args.data_path + "a_a_list_test.txt", "w")
		a_a_list = [a_a_list_train, a_a_list_test]
		train_num = 0
		test_num = 0
		for t in range(len(a_a_list)):
			for i in range(len(a_a_list[t])):
				if len(a_a_list[t][i]):
					if t == 0:
						for j in range(len(a_a_list[t][i])):
							a_a_list_train_f.write("%d, %d, %d\n"%(i, a_a_list[t][i][j], 1))
							node_n = random.randint(0, self.args.A_n - 1)
							while node_n in a_a_list[t][i]: 
								node_n = random.randint(0, self.args.A_n - 1)
							a_a_list_train_f.write("%d, %d, %d\n"%(i, node_n, 0))
							train_num += 2
					else:
						for j in range(len(a_a_list[t][i])):
							a_a_list_test_f.write("%d, %d, %d\n"%(i, a_a_list[t][i][j], 1))
							node_n = random.randint(0, self.args.A_n - 1)
							while node_n in a_a_list[t][i] or node_n in a_a_list_train[i] or len(a_a_list_train[i]) == 0:
								node_n = random.randint(0, self.args.A_n - 1)
							a_a_list_test_f.write("%d, %d, %d\n"%(i, node_n, 0))	 
							test_num += 2
		a_a_list_train_f.close()
		a_a_list_test_f.close()

		print("a_a_train_num: " + str(train_num))
		print("a_a_test_num: " + str(test_num))


	def a_p_citation_train_test(self):
		p_time = [0] * args.P_n
		p_time_f = open(args.data_path + self.args.data_set + '/' + "p_time.txt", "r")
		for line in p_time_f:
			line = line.strip()
			p_id = int(re.split('\t',line)[0])
			time = int(re.split('\t',line)[1])
			p_time[p_id] = time + 2005
		p_time_f.close()

		a_p_cite_list_train = [[] for k in range(self.args.A_n)]
		a_p_cite_list_test = [[] for k in range(self.args.A_n)]
		a_p_list = [self.a_p_list_train, self.a_p_list_test]
		p_p_cite_list_train = self.p_p_cite_list_train
		p_p_cite_list_test = self.p_p_cite_list_test
		
		for t in range(len(a_p_list)):
			for i in range(len(a_p_list[t])):
				for j in range(len(a_p_list[t][i])):
					if t == 0:
						p_id = int(a_p_list[t][i][j][1:])
						for k in range(len(p_p_cite_list_train[p_id])):
							a_p_cite_list_train[i].append(int(p_p_cite_list_train[p_id][k][1:]))
					else:
						if len(self.a_p_list_train[i]):
							p_id = int(a_p_list[t][i][j][1:])
							for k in range(len(p_p_cite_list_test[p_id])):
								cite_index = int(p_p_cite_list_test[p_id][k][1:])
								if p_time[cite_index] < args.T_split and (cite_index not in a_p_cite_list_train[i]):
									a_p_cite_list_test[i].append(cite_index)


		for i in range(self.args.A_n):
			a_p_cite_list_train[i] = list(set(a_p_cite_list_train[i]))
			a_p_cite_list_test[i] = list(set(a_p_cite_list_test[i]))

		test_count = 0 
		a_p_cite_list_train_f = open(args.data_path + "a_p_cite_list_train.txt", "w")
		a_p_cite_list_test_f = open(args.data_path + "a_p_cite_list_test.txt", "w")
		a_p_cite_list = [a_p_cite_list_train, a_p_cite_list_test]
		train_num = 0
		test_num = 0
		for t in range(len(a_p_cite_list)):
			for i in range(len(a_p_cite_list[t])):
				if t == 0:
					for j in range(len(a_p_cite_list[t][i])):
						a_p_cite_list_train_f.write("%d, %d, %d\n"%(i, a_p_cite_list[t][i][j], 1))
						node_n = random.randint(0, self.args.P_n - 1)
						while node_n in a_p_cite_list[t][i] or node_n in a_p_cite_list_train[i]: 
							node_n = random.randint(0, self.args.P_n - 1)
						a_p_cite_list_train_f.write("%d, %d, %d\n"%(i, node_n, 0))
						train_num += 2
				else:
					for j in range(len(a_p_cite_list[t][i])):
						a_p_cite_list_test_f.write("%d, %d, %d\n"%(i, a_p_cite_list[t][i][j], 1))
						node_n = random.randint(0, self.args.P_n - 1)
						while node_n in a_p_cite_list[t][i] or node_n in a_p_cite_list_train[i]:
							node_n = random.randint(0, self.args.P_n - 1)
						a_p_cite_list_test_f.write("%d, %d, %d\n"%(i, node_n, 0))	 
						test_num += 2
		a_p_cite_list_train_f.close()
		a_p_cite_list_test_f.close()

		print("a_p_cite_train_num: " + str(train_num))
		print("a_p_cite_test_num: " + str(test_num))


	def a_v_train_test(self):
		a_v_list_train = [[] for k in range(self.args.A_n)]
		a_v_list_test = [[] for k in range(self.args.A_n)]
		a_p_list = [self.a_p_list_train, self.a_p_list_test]
		for t in range(len(a_p_list)):
			for i in range(len(a_p_list[t])):
				for j in range(len(a_p_list[t][i])):
					p_id = int(a_p_list[t][i][j][1:])
					if t == 0:
						a_v_list_train[i].append(self.p_v[p_id])
					else:
						if self.p_v[p_id] not in a_v_list_train[i] and len(a_v_list_train[i]):
							a_v_list_test[i].append(self.p_v[p_id])

		for k in range(self.args.A_n):
			a_v_list_train[k] = list(set(a_v_list_train[k]))
			a_v_list_test[k] = list(set(a_v_list_test[k]))

		a_v_list_train_f = open(args.data_path + "a_v_list_train.txt", "w")
		a_v_list_test_f = open(args.data_path + "a_v_list_test.txt", "w")
		a_v_list = [a_v_list_train, a_v_list_test]
		for t in range(len(a_v_list)):
			for i in range(len(a_v_list[t])):
				if t == 0:
					if len(a_v_list[t][i]):
						a_v_list_train_f.write(str(i)+":")
						for j in range(len(a_v_list[t][i])):
							a_v_list_train_f.write(str(a_v_list[t][i][j])+",")
						a_v_list_train_f.write("\n")
				else:
					if len(a_v_list[t][i]):
						a_v_list_test_f.write(str(i)+":")
						for j in range(len(a_v_list[t][i])):
							a_v_list_test_f.write(str(a_v_list[t][i][j])+",")
						a_v_list_test_f.write("\n")
		a_v_list_train_f.close()
		a_v_list_test_f.close()



if __name__ == '__main__':
	inputs = input_data(args)
	inputs.gen_het_rand_walk()
	print(args)


