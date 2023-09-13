import torch
import torch.optim as optim
import data_generator
import tools
from args import read_args
from IMGE.data import *
from pathlib import Path
from torch.autograd import Variable
import numpy as np
import random
torch.set_num_threads(2)
import os
import time
import json
import pickle


os.environ['CUDA_VISIBLE_DEVICES']='0'


class model_class(object):
	def __init__(self, args):
		super(model_class, self).__init__()
		self.args = args
		self.gpu = args.cuda

		input_data = data_generator.input_data(args = self.args)

		self.input_data = input_data

		if self.args.train_test_label == 2: 
			input_data.het_walk_restart()
			print ("neighbor set generation finish")
			exit(0)

		feature_list = [input_data.i_net_embed, input_data.img_embed, input_data.text_embed]

		for i in range(len(feature_list)):
			feature_list[i] = torch.from_numpy(np.array(feature_list[i])).float()

		if self.gpu:
			for i in range(len(feature_list)):
				feature_list[i] = feature_list[i].cuda()
		self.feature_list = feature_list
		i_neigh_list = input_data.i_neigh_list
		i_neigh_with_group_list = input_data.i_neigh_with_group_list
		i_neigh_no_group_list = input_data.i_neigh_no_group_list

		i_train_id_list = input_data.i_train_id_list

		if args.model_name != 'HHGSI_IMGE':
			self.model = tools.HHGSI(args, feature_list, i_neigh_list, i_train_id_list,
									  i_neigh_with_group_list, i_neigh_no_group_list)
		else:
			self.model = tools.HHGSI_IMGE(args, feature_list, i_neigh_list, i_train_id_list,
									  i_neigh_with_group_list, i_neigh_no_group_list)

		if self.gpu:
			self.model.cuda()
		self.parameters = filter(lambda p: p.requires_grad, self.model.parameters())
		self.optim = optim.Adam(self.parameters, lr=self.args.lr, weight_decay=0)
		self.model.init_weights()


	def model_train(self):
		print ('model training ...')
		if self.args.checkpoint != '':
			self.model.load_state_dict(torch.load(self.args.checkpoint))
		
		self.model.train()
		mini_batch_s = self.args.mini_batch_s
		embed_d = self.args.embed_d
		clip_f_d = self.args.clip_f_d

		for iter_i in range(self.args.train_iter_n):
			print ('iteration ' + str(iter_i) + ' ...')
			triple_list = self.input_data.sample_het_walk_triple()
			min_len = 1e10
			for ii in range(len(triple_list)):
				if len(triple_list[ii]) < min_len:
					min_len = len(triple_list[ii])
			batch_n = int(min_len / mini_batch_s)
			print ("batch_n:", batch_n)
			for k in range(batch_n):
				c_out = torch.zeros([len(triple_list), mini_batch_s, embed_d])
				p_out = torch.zeros([len(triple_list), mini_batch_s, embed_d])
				n_out = torch.zeros([len(triple_list), mini_batch_s, embed_d])
				d_l = torch.zeros([len(triple_list), mini_batch_s, clip_f_d])

				for triple_index in range(len(triple_list)):
					triple_list_temp = triple_list[triple_index]
					triple_list_batch = triple_list_temp[k * mini_batch_s : (k + 1) * mini_batch_s]
					if triple_list_batch == []:
						print('triple_list_batch', triple_list_batch)
					if args.model_name != 'HHGSI_IMGE':
						c_out_temp, p_out_temp, n_out_temp = self.model(triple_list_batch, triple_index)

						c_out[triple_index] = c_out_temp
						p_out[triple_index] = p_out_temp
						n_out[triple_index] = n_out_temp
						c_list = [k[0] for k in triple_list_batch]
						d_l[triple_index] = self.feature_list[1][c_list] 
					else:
						c_out_temp = self.model(triple_list_batch, triple_index)

						c_out[triple_index] = c_out_temp
						c_list = [k[0] for k in triple_list_batch]
						d_l[triple_index] = self.feature_list[1][c_list] 
				if iter_i % 20 == 0:
					hh = 1
				if args.model_name != 'HHGSI_IMGE':
					loss = tools.cross_entropy_loss(c_out, p_out, n_out, d_l, embed_d, self.args)
				else:
					loss = tools.cross_entropy_loss_IMGE(c_out, d_l)

				self.optim.zero_grad()
				loss.backward()
				self.optim.step()

				if k % 100 == 0:
					print ("loss: " + str(loss))

			if iter_i % self.args.save_model_freq == 0:
				torch.save(self.model.state_dict(), self.args.model_path + "HetGNN_" + str(iter_i) + ".pt")
				triple_index = 1
				self.model([], triple_index)
			print ('iteration ' + str(iter_i) + ' finish.')

if __name__ == '__main__':
	args = read_args()
	print("------arguments-------")
	for k, v in vars(args).items():
		print(k + ': ' + str(v))

	random.seed(args.random_seed)
	np.random.seed(args.random_seed)
	torch.manual_seed(args.random_seed)
	torch.cuda.manual_seed_all(args.random_seed)

	model_object = model_class(args)

	if args.train_test_label == 0:
		model_object.model_train()

