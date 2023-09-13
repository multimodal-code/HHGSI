import six.moves.cPickle as pickle
import numpy as np
import string
import re
import random
import math
from collections import Counter
from itertools import *
from args import read_args


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
				neigh_list = re.split(':', line)[1]
				neigh_list_id = re.split(',', neigh_list)
				if f_name == 'i_i_list.txt':
					if neigh_list_id[0] != '':
						for j in range(len(neigh_list_id)):
							i_i_list[node_id].append('i'+str(neigh_list_id[j]))
				if f_name == 'i_i_with_group_list.txt':
					if neigh_list_id[0] != '':
						for j in range(len(neigh_list_id)):
							i_i_with_group_list[node_id].append('i'+str(neigh_list_id[j]))
				if f_name == 'i_i_no_group_list.txt':
					if neigh_list_id[0] != '':
						for j in range(len(neigh_list_id)):
							i_i_no_group_list[node_id].append('i'+str(neigh_list_id[j]))
			neigh_f.close()

		self.i_i_list = i_i_list
		self.i_i_with_group_list = i_i_with_group_list
		self.i_i_no_group_list = i_i_no_group_list

		if self.args.train_test_label != 2:
			self.triple_sample_i = self.compute_sample_p()

			img_embed = np.zeros((self.args.I_n, self.args.clip_f_d))
			i_e = open(self.args.data_path + self.args.data_set + "/" + "img_emb_clip.txt", "r")
			for line in islice(i_e, 1, None):
				values = line.split()
				index = int(values[0])
				embeds = np.asarray(values[1:], dtype='float32')
				img_embed[index] = embeds
			i_e.close()

			text_embed = np.zeros((self.args.I_n, self.args.clip_f_d))
			t_e = open(self.args.data_path + self.args.data_set + "/" + "text_emb_clip.txt", "r")
			for line in islice(t_e, 1, None):
				values = line.split()
				index = int(values[0])
				embeds = np.asarray(values[1:], dtype='float32')
				text_embed[index] = embeds
			t_e.close()

			self.img_embed = img_embed
			self.text_embed = text_embed


			i_net_embed = np.zeros((self.args.I_n, self.args.in_f_d))
			net_e_f = open(self.args.data_path + self.args.data_set + "/" + "node_net_embedding.txt", "r")
			for line in islice(net_e_f, 1, None):
				line = line.strip()
				index = re.split(' ', line)[0]
				if len(index) and index[0] == 'i':
					embeds = np.asarray(re.split(' ', line)[1:], dtype='float32')
					i_net_embed[int(index[1:])] = embeds
			net_e_f.close()
			self.i_net_embed = i_net_embed

			i_neigh_list = [[[] for i in range(self.args.I_n)] for j in range(3)]
			i_neigh_with_group_list = [[[] for i in range(self.args.I_n)] for j in range(3)]
			i_neigh_no_group_list = [[[] for i in range(self.args.I_n)] for j in range(3)]

			file_list = ["het_neigh.txt", "het_neigh_with_group.txt", "het_neigh_no_group.txt"]
			for file in file_list:
				het_neigh_f = open(self.args.data_path + self.args.data_set + '/' + file, "r")
				for line in het_neigh_f:
					line = line.strip()
					node_id = re.split(':', line)[0]
					neigh = re.split(':', line)[1]
					neigh_list = re.split(',', neigh)
					if node_id[0] == 'i' and len(node_id) > 1:
						for j in range(len(neigh_list)):
							if neigh_list[j][0] == 'i':
								if file == "het_neigh.txt":
									try:
										i_neigh_list[0][int(node_id[1:])].append(int(neigh_list[j][1:]))
									except ValueError:
										hh = 100
								if file == "het_neigh_with_group.txt":
									try:
										i_neigh_with_group_list[0][int(node_id[1:])].append(int(neigh_list[j][1:]))
									except ValueError:
										hh = 100
								if file == "het_neigh_no_group.txt":
									try:
										i_neigh_no_group_list[0][int(node_id[1:])].append(int(neigh_list[j][1:]))
									except ValueError:
										hh = 100
				het_neigh_f.close()
			i_neigh_list_top = [[[] for i in range(self.args.I_n)] for j in range(3)]
			i_neigh_with_group_list_top = [[[] for i in range(self.args.I_n)] for j in range(3)]
			i_neigh_no_group_list_top = [[[] for i in range(self.args.I_n)] for j in range(3)]
			top_k = [10]
			for i in range(self.args.I_n):
				for j in range(len(top_k)):
					i_neigh_list_temp = Counter(i_neigh_list[j][i])
					i_neigh_with_group_list_temp = Counter(i_neigh_with_group_list[j][i])
					i_neigh_no_group_list_temp = Counter(i_neigh_no_group_list[j][i])
					top_list = i_neigh_list_temp.most_common(top_k[j])
					top_with_group_list = i_neigh_with_group_list_temp.most_common(top_k[j])
					top_no_group_list = i_neigh_no_group_list_temp.most_common(top_k[j])
					neigh_size = 10
					for k in range(len(top_list)):
						i_neigh_list_top[j][i].append(int(top_list[k][0]))
					for k in range(len(top_with_group_list)):
						i_neigh_with_group_list_top[j][i].append(int(top_with_group_list[k][0]))
					for k in range(len(top_no_group_list)):
						i_neigh_no_group_list_top[j][i].append(int(top_no_group_list[k][0]))
					if len(i_neigh_list_top[j][i]) and len(i_neigh_list_top[j][i]) < neigh_size:
						for l in range(len(i_neigh_list_top[j][i]), neigh_size):
							i_neigh_list_top[j][i].append(random.choice(i_neigh_list_top[j][i]))
					if len(i_neigh_with_group_list_top[j][i]) and len(i_neigh_with_group_list_top[j][i]) < neigh_size:
						for l in range(len(i_neigh_with_group_list_top[j][i]), neigh_size):
							i_neigh_with_group_list_top[j][i].append(random.choice(i_neigh_with_group_list_top[j][i]))
					if len(i_neigh_no_group_list_top[j][i]) and len(i_neigh_no_group_list_top[j][i]) < neigh_size:
						for l in range(len(i_neigh_no_group_list_top[j][i]), neigh_size):
							i_neigh_no_group_list_top[j][i].append(random.choice(i_neigh_no_group_list_top[j][i]))


			i_neigh_list[:] = []

			self.i_neigh_list = i_neigh_list_top
			self.i_neigh_with_group_list = i_neigh_with_group_list_top
			self.i_neigh_no_group_list = i_neigh_no_group_list_top

			train_id_list = [[] for i in range(3)]
			for i in range(3):
				if i == 0:
					for l in range(self.args.I_n):
						if len(i_neigh_list_top[i][l]):
							train_id_list[i].append(l)
					self.i_train_id_list = np.array(train_id_list[i])	


	def het_walk_restart(self):
		i_neigh_list = [[] for k in range(self.args.I_n)]
		i_neigh_with_group_list = [[] for k in range(self.args.I_n)]
		i_neigh_no_group_list = [[] for k in range(self.args.I_n)]

		node_n = [self.args.I_n]
		for i in range(len(node_n)):
			for j in range(node_n[i]):
				if j == 4:
					hh = 100
				if i == 0:
					neigh_temp = self.i_i_list[j]
					neigh_temp_with_group = self.i_i_with_group_list[j]
					neigh_temp_no_group = self.i_i_no_group_list[j]
					neigh_train = i_neigh_list[j]
					neigh_train_with_group = i_neigh_with_group_list[j]
					neigh_train_no_group = i_neigh_no_group_list[j]
					curNode = "i" + str(j)
				if i == 0 and len(neigh_temp):
					neigh_L = 0
					i_L = 0
					while neigh_L < 100:
						rand_p = random.random()
						if rand_p > 0.5:
							if curNode[0] == "i":
								curNode = random.choice(self.i_i_list[int(curNode[1:])])
								neigh_train.append(curNode)
								neigh_L += 1
						else:
							if i == 0:
								curNode = ('i' + str(j))

				if i == 0 and len(neigh_temp_with_group):
					neigh_L = 0
					while neigh_L < 100:
						rand_p = random.random()
						if rand_p > 0.5:
							if curNode[0] == "i":
								try:
									curNode = random.choice(self.i_i_with_group_list[int(curNode[1:])])
								except IndexError:
									break
								neigh_train_with_group.append(curNode)
								neigh_L += 1
						else:
							if i == 0:
								curNode = ('i' + str(j))

				if i == 0 and len(neigh_temp_no_group):
					neigh_L = 0
					while neigh_L < 100:
						rand_p = random.random()
						if rand_p > 0.5:
							if curNode[0] == "i":
								try:
									curNode = random.choice(self.i_i_no_group_list[int(curNode[1:])])
								except IndexError:
									break
								neigh_train_no_group.append(curNode)
								neigh_L += 1
						else:
							if i == 0:
								curNode = ('i' + str(j))

		for i in range(len(node_n)):
			for j in range(node_n[i]):
				if i == 0:
					i_neigh_list[j] = list(i_neigh_list[j])
					i_neigh_with_group_list[j] = list(i_neigh_with_group_list[j])
					i_neigh_no_group_list[j] = list(i_neigh_no_group_list[j])

		file_list = ["het_neigh.txt", "het_neigh_with_group.txt", "het_neigh_no_group.txt"]
		for file in file_list:
			neigh_f = open(self.args.data_path + self.args.data_set + '/' + file, "w")
			for i in range(len(node_n)):
				for j in range(node_n[i]):
					if file == "het_neigh.txt":
						if i == 0:
							neigh_train = i_neigh_list[j]
							curNode = "i" + str(j)
						if i == 0 and len(neigh_train):
							neigh_f.write(curNode + ":")
							for k in range(len(neigh_train) - 1):
								neigh_f.write(neigh_train[k] + ",")
							neigh_f.write(neigh_train[-1] + "\n")
					if file == "het_neigh_with_group.txt":
						if i == 0:
							neigh_train = i_neigh_with_group_list[j]
							curNode = "i" + str(j)
						if i == 0 and len(neigh_train):
							neigh_f.write(curNode + ":")
							for k in range(len(neigh_train) - 1):
								neigh_f.write(neigh_train[k] + ",")
							neigh_f.write(neigh_train[-1] + "\n")
					if file == "het_neigh_no_group.txt":
						if i == 0:
							neigh_train = i_neigh_no_group_list[j]
							curNode = "i" + str(j)
						if i == 0 and len(neigh_train):
							neigh_f.write(curNode + ":")
							for k in range(len(neigh_train) - 1):
								neigh_f.write(neigh_train[k] + ",")
							neigh_f.write(neigh_train[-1] + "\n")
			neigh_f.close()


	def compute_sample_p(self):
		print("computing sampling ratio for each kind of triple ...")
		window = self.args.window
		walk_L = self.args.walk_L

		total_triple_n = [0.0]
		het_walk_f = open(self.args.data_path + self.args.data_set + '/' + "het_random_walk.txt", "r")
		centerNode = ''
		neighNode = ''

		for line in het_walk_f:
			line = line.strip()
			path = []
			path_list = re.split(' ', line)
			for i in range(len(path_list)):
				path.append(path_list[i])
			for j in range(walk_L):
				centerNode = path[j]
				if len(centerNode) > 1:
					if centerNode[0] == 'i':
						for k in range(j - window, j + window + 1):
							if k and k < walk_L and k != j:
								neighNode = path[k]
								if neighNode[0] == 'i':
									total_triple_n[0] += 1
		het_walk_f.close()

		for i in range(len(total_triple_n)):
			total_triple_n[i] = self.args.batch_s / (total_triple_n[i] * 10)
		print("sampling ratio computing finish.")

		return total_triple_n


	def sample_het_walk_triple(self):
		print ("sampling triple relations ...")
		triple_list = [[] for k in range(9)]
		window = self.args.window
		walk_L = self.args.walk_L
		I_n = self.args.I_n
		triple_sample_i = self.triple_sample_i

		het_walk_f = open(self.args.data_path + self.args.data_set + '/' + "het_random_walk.txt", "r")
		centerNode = ''
		neighNode = ''
		for line in het_walk_f:
			line = line.strip()
			path = []
			path_list = re.split(' ', line)
			for i in range(len(path_list)):
				path.append(path_list[i])
			for j in range(walk_L):
				centerNode = path[j]
				if len(centerNode) > 1:
					if centerNode[0]=='i':
						for k in range(j - window, j + window + 1):
							if k and k < walk_L and k != j:
								neighNode = path[k]
								if neighNode[0] == 'i' and random.random() < triple_sample_i[0]:
									negNode = random.randint(0, I_n - 1)
									while len(self.i_i_list[negNode]) == 0:
										negNode = random.randint(0, I_n - 1)
									triple = [int(centerNode[1:]), int(neighNode[1:]), int(negNode)]
									triple_list[0].append(triple)

		het_walk_f.close()

		return [triple_list[0]]

if __name__ == '__main__':
	args = read_args()
	print(args)
	a = input_data(args)
	a.het_walk_restart()
	print(a)


