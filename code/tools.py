import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from args import read_args
import numpy as np
import string
import re
import math
from IMGE.transformer import GATEncoder
from IMGE.data import *
from IMGE.util import *
import time

args = read_args()


class HHGSI(nn.Module):
	def __init__(self, args, feature_list, i_neigh_list, i_train_id_list, i_neigh_with_group_list, i_neigh_no_group_list):
		super(HHGSI, self).__init__()
		embed_d = args.embed_d
		in_f_d = args.in_f_d
		self.args = args
		self.I_n = args.I_n
		self.feature_list = feature_list
		self.i_neigh_list = i_neigh_list
		self.i_neigh_with_group_list = i_neigh_with_group_list
		self.i_neigh_no_group_list = i_neigh_no_group_list
		self.i_train_id_list = i_train_id_list
		self.line_layer = nn.Linear(in_features = args.clip_f_d, out_features = args.in_f_d)
		self.decoder = nn.Linear(in_features=args.embed_d, out_features=args.clip_f_d)
		self.i_content_rnn = nn.LSTM(embed_d, int(embed_d / 2), 1, bidirectional=True)
		self.i_neigh_rnn = nn.LSTM(embed_d, int(embed_d / 2), 1, bidirectional=True)

		self.node_neigh_att = nn.Parameter(torch.ones(embed_d, 1), requires_grad=True)
		self.i_neigh_att = nn.Parameter(torch.ones(embed_d * 2, 1), requires_grad=True)

		self.mhatt_o2ox = clone(MultiHeadedAttention(4, 1024, 0.1, v=0, output=0), 1)
		self.res4o2ox = clone(SublayerConnectionv2(1024, 0.1), 1)


		self.softmax = nn.Softmax(dim=1)
		self.sigmoid = nn.Sigmoid()
		self.act_relu = nn.ReLU()
		self.act = nn.LeakyReLU()
		self.drop = nn.Dropout(p=0.5)
		self.bn = nn.BatchNorm1d(embed_d)
		self.embed_d = embed_d
		self.img_text_list, self.vocab_dic = read_multimodalgraph_data(args)
		self.GATEncoder = GATEncoder(128, 256, 4, len(self.vocab_dic), 0.5)

	def init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Linear) or isinstance(m, nn.Parameter):
				nn.init.xavier_normal_(m.weight.data)
				m.bias.data.fill_(0.1)

	def i_content_agg(self, id_batch):
		embed_d = self.embed_d
		img_text_list = [self.img_text_list[i] for i in id_batch[0]]
		if img_text_list == []:
			print(img_text_list)
		try:
			img_embed_batch = self.feature_list[1][id_batch]
		except RuntimeError:
			hh = 1
			img_embed_batch = self.feature_list[1][id_batch]
		if args.model_name != 'HHGSI_IHGNN':
			sources, source_masks, obj_feat, obj_mask, matrix = self.img_text_process(img_text_list, id_batch)

			img_text_embed_batch = self.GATEncoder.forward(sources, source_masks, img_embed_batch, obj_feat, None, obj_mask, matrix)

			concate_embed = torch.cat((img_embed_batch, img_text_embed_batch), \
									  1).view(len(id_batch[0]), 2, embed_d)
		else:
			concate_embed = torch.cat((img_embed_batch,), 1).view(len(id_batch[0]), 1, embed_d)

		concate_embed = torch.transpose(concate_embed, 0, 1)
		all_state, last_state = self.i_content_rnn(concate_embed)

		return torch.mean(all_state, 0)

	def node_neigh_agg(self, id_batch, node_type):
		embed_d = self.embed_d

		batch_s = int(len(id_batch[0]) / 10)

		if node_type == 1:
			try:
				neigh_agg = self.i_content_agg(id_batch).view(batch_s, 10, embed_d)
			except RuntimeError:
				neigh_agg = self.i_content_agg(id_batch).view(batch_s, 10, embed_d)
			all_state = self.res4o2ox[0](neigh_agg, self.mhatt_o2ox[0](neigh_agg, neigh_agg, neigh_agg))

		neigh_agg = torch.mean(all_state, 1).view(batch_s, embed_d)

		return neigh_agg

	def node_het_agg(self, id_batch, node_type): 
		if args.model_name != 'EHGSI_GNN':
			i_neigh_batch = [[0] * 10] * len(id_batch)
			i_neigh_with_group_batch = [[0] * 10] * len(id_batch)
			i_neigh_no_group_batch = [[0] * 10] * len(id_batch)
			for i in range(len(id_batch)):
				if node_type == 1:
					i_neigh_batch[i] = self.i_neigh_list[0][id_batch[i]]
					i_neigh_with_group_batch[i] = self.i_neigh_with_group_list[0][id_batch[i]]
					i_neigh_no_group_batch[i] = self.i_neigh_no_group_list[0][id_batch[i]]

			i_neigh_batch = np.reshape(i_neigh_batch, (1, -1))
			if i_neigh_batch.size == 0:
				hh = 0
			i_agg_batch = self.node_neigh_agg(i_neigh_batch, 1)
			while [] in i_neigh_with_group_batch:
				i_neigh_with_group_batch.remove([])
			i_neigh_with_group_batch = np.reshape(i_neigh_with_group_batch, (1, -1))
			if i_neigh_with_group_batch.size != 0:
				i_agg_with_group_batch = self.node_neigh_agg(i_neigh_with_group_batch, 1)
			else:
				i_agg_with_group_batch = i_agg_batch
			while [] in i_neigh_no_group_batch:
				i_neigh_no_group_batch.remove([])
			i_neigh_no_group_batch = np.reshape(i_neigh_no_group_batch, (1, -1))
			if i_neigh_no_group_batch.size != 0:
				i_agg_no_group_batch = self.node_neigh_agg(i_neigh_no_group_batch, 1)
			else:
				i_agg_no_group_batch = i_agg_batch

			# attention module
			id_batch = np.reshape(id_batch, (1, -1))
			if node_type == 1:
				c_agg_batch = self.i_content_agg(id_batch)

			c_agg_batch_2 = torch.cat((c_agg_batch, c_agg_batch), 1).view(len(c_agg_batch), self.embed_d * 2)
			i_agg_batch_2 = torch.cat((c_agg_batch, i_agg_batch), 1).view(len(c_agg_batch), self.embed_d * 2)
			cat_tensor_1 = torch.zeros(c_agg_batch.size(0) - i_agg_with_group_batch.size(0), args.embed_d).cuda()
			i_agg_with_group_batch = torch.cat((i_agg_with_group_batch, cat_tensor_1), 0)
			i_agg_batch_2_with_group = torch.cat((c_agg_batch, i_agg_with_group_batch), 1).view(len(c_agg_batch), self.embed_d * 2)
			cat_tensor_2 = torch.zeros(c_agg_batch.size(0) - i_agg_no_group_batch.size(0), args.embed_d).cuda()
			i_agg_no_group_batch = torch.cat((i_agg_no_group_batch, cat_tensor_2), 0)
			i_agg_batch_2_no_group = torch.cat((c_agg_batch, i_agg_no_group_batch), 1).view(len(c_agg_batch), self.embed_d * 2)

			concate_embed = torch.cat((c_agg_batch_2, i_agg_batch_2, i_agg_batch_2_with_group,
									   i_agg_batch_2_no_group), 1).view(len(c_agg_batch), 4, self.embed_d * 2)
			if node_type == 1:
				atten_w = self.act(torch.bmm(concate_embed, self.i_neigh_att.unsqueeze(0).expand(len(c_agg_batch), \
																								 *self.i_neigh_att.size())))
			atten_w = self.softmax(atten_w).view(len(c_agg_batch), 1, 4)

			# weighted combination
			concate_embed = torch.cat((c_agg_batch, i_agg_batch, i_agg_with_group_batch,
									   i_agg_no_group_batch), 1).view(len(c_agg_batch), 4, self.embed_d)
		else:
			i_neigh_batch = [[0] * 10] * len(id_batch)
			for i in range(len(id_batch)):
				if node_type == 1:
					i_neigh_batch[i] = self.i_neigh_list[0][id_batch[i]]

			i_neigh_batch = np.reshape(i_neigh_batch, (1, -1))
			if i_neigh_batch.size == 0:
				hh = 0
			i_agg_batch = self.node_neigh_agg(i_neigh_batch, 1)

			# attention module
			id_batch = np.reshape(id_batch, (1, -1))
			if node_type == 1:
				c_agg_batch = self.i_content_agg(id_batch)

			c_agg_batch_2 = torch.cat((c_agg_batch, c_agg_batch), 1).view(len(c_agg_batch), self.embed_d * 2)
			i_agg_batch_2 = torch.cat((c_agg_batch, i_agg_batch), 1).view(len(c_agg_batch), self.embed_d * 2)

			concate_embed = torch.cat((c_agg_batch_2, i_agg_batch_2), 1).view(len(c_agg_batch), 2, self.embed_d * 2)
			if node_type == 1:
				atten_w = self.act(torch.bmm(concate_embed, self.i_neigh_att.unsqueeze(0).expand(len(c_agg_batch), *self.i_neigh_att.size())))
			atten_w = self.softmax(atten_w).view(len(c_agg_batch), 1, 2)

			# weighted combination
			concate_embed = torch.cat((c_agg_batch, i_agg_batch), 1).view(len(c_agg_batch), 2, self.embed_d)

		weight_agg_batch = torch.bmm(atten_w, concate_embed).view(len(c_agg_batch), self.embed_d)

		return weight_agg_batch

	def het_agg(self, triple_index, c_id_batch, pos_id_batch, neg_id_batch):
		embed_d = self.embed_d

		if triple_index == 0:
			c_agg = self.node_het_agg(c_id_batch, 1)
			p_agg = self.node_het_agg(pos_id_batch, 1)
			n_agg = self.node_het_agg(neg_id_batch, 1)
		elif triple_index == 1:  # save learned node embedding
			embed_file = open(self.args.data_path + self.args.data_set + '/' + "node_embedding_alpha_" + str(int(args.loss_weight_alpha * 10)) + "_beta_" + str(int(args.loss_weight_beta * 10)) + ".txt", "w")
			save_batch_s = self.args.mini_batch_s
			for i in range(3):
				if i != 0:
					continue
				if i == 0:
					batch_number = int(len(self.i_train_id_list) / save_batch_s)
				for j in range(batch_number):
					if i == 0:
						id_batch = self.i_train_id_list[j * save_batch_s: (j + 1) * save_batch_s]
						out_temp = self.node_het_agg(id_batch, 1)
					out_temp = out_temp.data.cpu().numpy()
					for k in range(len(id_batch)):
						index = id_batch[k]
						if i == 0:
							embed_file.write('i' + str(index) + " ")
						for l in range(embed_d - 1):
							embed_file.write(str(out_temp[k][l]) + " ")
						embed_file.write(str(out_temp[k][-1]) + "\n")

				if i == 0:
					id_batch = self.i_train_id_list[batch_number * save_batch_s:]
					if id_batch.size == 0:
						embed_file.close()
						continue
					out_temp = self.node_het_agg(id_batch, 1)
				out_temp = out_temp.data.cpu().numpy()
				for k in range(len(id_batch)):
					index = id_batch[k]
					if i == 0:
						embed_file.write('i' + str(index) + " ")
					for l in range(embed_d - 1):
						embed_file.write(str(out_temp[k][l]) + " ")
					embed_file.write(str(out_temp[k][-1]) + "\n")
			embed_file.close()
			return [], [], []

		return c_agg, p_agg, n_agg

	def aggregate_all(self, triple_list_batch, triple_index):
		c_id_batch = [x[0] for x in triple_list_batch]
		pos_id_batch = [x[1] for x in triple_list_batch]
		neg_id_batch = [x[2] for x in triple_list_batch]

		c_agg, pos_agg, neg_agg = self.het_agg(triple_index, c_id_batch, pos_id_batch, neg_id_batch)

		return c_agg, pos_agg, neg_agg

	def forward(self, triple_list_batch, triple_index):
		c_out, p_out, n_out = self.aggregate_all(triple_list_batch, triple_index)
		return c_out, p_out, n_out

	def img_text_process(self, img_text_list, id_batch):
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		batch_len = len(id_batch[0])
		srcpadid = self.vocab_dic['<pad>']
		start = time.time()
		topk = 1
		thre = 0.0
		objdim = args.objdim
		t1 = time.time()
		src = self.text_to_id(img_text_list, batch_len)
		sources, source_masks = prepare_sources(src, srcpadid)
		sources = sources.to(device)
		source_masks = source_masks.to(device)
		boxfeats = [img_text_list[i]['features'] for i in range(0, batch_len)]
		boxprobs = [img_text_list[i]['objects_conf'] for i in range(0, batch_len)]
		imgs = [img_text_list[i]['img_id'] for i in range(0, batch_len)]
		aligns = [img_text_list[i]['graph'] for i in range(0, batch_len)]
		regions_num = []
		for align in aligns:
			if len(align) == 0:
				regions_num.append(1)
			else:
				regions_num.append(align[-1][1] + 1)

		obj_feat = sources.new_zeros(sources.size(0), 15, topk,
									 objdim).float()
		obj_mask = source_masks.new_zeros(sources.size(0), 15 * topk) 

		matrix = sources.new_zeros(sources.size(0), sources.size(1), 15 * topk).float()

		for ib, img in enumerate(imgs):
			boxfeat = torch.tensor(boxfeats[ib]).reshape(-1, topk, objdim)
			img_boxprobs = torch.tensor(boxprobs[ib])
			ge_thre = (img_boxprobs >= thre).byte() 
			ge_thre[list(range(0, ge_thre.size(0), 5))] = 1 
			obj_mask[ib, :15 * topk] = ge_thre[:15 * topk]
			obj_feat[ib, :15 * topk] = boxfeat[:15 * topk, :topk]
			for item in aligns[ib]:
				objixs = sources.new_tensor([n + item[1] * topk for n in range(topk)])
				if item[0] >= args.fix_length:
					continue
				matrix[ib, item[0], objixs] = ge_thre[objixs].float().cuda()

		obj_feat = obj_feat.view(sources.size(0), -1, objdim)
		obj_mask = obj_mask.unsqueeze(1)
		obj_feat = obj_feat.to(device)
		obj_mask = obj_mask.to(device)
		matrix = matrix.to(device)

		return sources, source_masks, obj_feat, obj_mask, matrix

	def text_to_id(self, img_text_list, batch_len):
		text_list = [img_text_list[i]['text'] for i in range(0, batch_len)]
		text_id_list = []
		for i in range(0, len(text_list)):
			text = text_list[i]
			text_id_list.append([])
			word_list = text.split(' ')
			count = 0
			for word in word_list:
				if count < args.fix_length:
					if word in self.vocab_dic:
						text_id_list[i].append(self.vocab_dic[word])
					else:
						text_id_list[i].append(self.vocab_dic['<unk>'])
				else:
					break
				count += 1
			while count < 50:
				text_id_list[i].append(self.vocab_dic['<pad>'])
				count += 1
		text_id = torch.tensor(text_id_list)
		return text_id

class HHGSI_IMGE(nn.Module):
	def __init__(self, args, feature_list, i_neigh_list, i_train_id_list, i_neigh_with_group_list,
				 i_neigh_no_group_list):
		super(HHGSI_IMGE, self).__init__()
		embed_d = args.embed_d
		in_f_d = args.in_f_d
		self.args = args
		self.I_n = args.I_n
		self.feature_list = feature_list
		self.i_neigh_list = i_neigh_list
		self.i_neigh_with_group_list = i_neigh_with_group_list
		self.i_neigh_no_group_list = i_neigh_no_group_list
		self.i_train_id_list = i_train_id_list

		self.line_layer = nn.Linear(in_features=args.clip_f_d, out_features=args.in_f_d)
		self.decoder = nn.Linear(in_features=args.embed_d, out_features=args.clip_f_d)
		self.i_content_rnn = nn.LSTM(embed_d, int(embed_d / 2), 1, bidirectional=True)
		self.i_neigh_rnn = nn.LSTM(embed_d, int(embed_d / 2), 1, bidirectional=True)

		self.node_neigh_att = nn.Parameter(torch.ones(embed_d, 1), requires_grad=True)
		self.i_neigh_att = nn.Parameter(torch.ones(embed_d * 2, 1), requires_grad=True)

		self.mhatt_o2ox = clone(MultiHeadedAttention(4, 1024, 0.1, v=0, output=0), 1)
		self.res4o2ox = clone(SublayerConnectionv2(1024, 0.1), 1)

		self.softmax = nn.Softmax(dim=1)
		self.sigmoid = nn.Sigmoid()
		self.act_relu = nn.ReLU()
		self.act = nn.LeakyReLU()
		self.drop = nn.Dropout(p=0.5)
		self.bn = nn.BatchNorm1d(embed_d)
		self.embed_d = embed_d
		self.img_text_list, self.vocab_dic = read_multimodalgraph_data(args)
		self.GATEncoder = GATEncoder(128, 256, 4, len(self.vocab_dic), 0.5)

	def init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Linear) or isinstance(m, nn.Parameter):
				nn.init.xavier_normal_(m.weight.data)
				m.bias.data.fill_(0.1)


	def i_content_agg(self, id_batch):
		embed_d = self.embed_d
		img_text_list = [self.img_text_list[i] for i in id_batch[0]]
		if img_text_list == []:
			print(img_text_list)
		sources, source_masks, obj_feat, obj_mask, matrix = self.img_text_process(img_text_list, id_batch)
		try:
			img_embed_batch = self.feature_list[1][id_batch]
		except RuntimeError:
			hh = 1
			img_embed_batch = self.feature_list[1][id_batch]

		img_text_embed_batch = self.GATEncoder.forward(sources, source_masks, img_embed_batch, obj_feat, None, obj_mask,
													   matrix)
		concate_embed = torch.cat((img_embed_batch, img_text_embed_batch), \
								  1).view(len(id_batch[0]), 2, embed_d)

		concate_embed = torch.transpose(concate_embed, 0, 1)
		all_state, last_state = self.i_content_rnn(concate_embed)

		all_state = torch.mean(all_state, 0)

		return all_state

	def node_het_agg(self, id_batch): 

		id_batch = np.reshape(id_batch, (1, -1))

		c_agg_batch = self.i_content_agg(id_batch)


		return c_agg_batch

	def het_agg(self, triple_index, c_id_batch):
		embed_d = self.embed_d

		if triple_index == 0:
			c_agg = self.node_het_agg(c_id_batch)
		elif triple_index == 1:  # save learned node embedding
			embed_file = open(self.args.data_path + self.args.data_set + '/' + "node_embedding_alpha_" + str(int(args.loss_weight_alpha * 10)) + "_beta_" + str(int(args.loss_weight_beta * 10)) + ".txt", "w")
			save_batch_s = self.args.mini_batch_s
			for i in range(3):
				if i != 0:
					continue
				if i == 0:
					batch_number = int(len(self.i_train_id_list) / save_batch_s)
				for j in range(batch_number):
					if i == 0:
						id_batch = self.i_train_id_list[j * save_batch_s: (j + 1) * save_batch_s]
						out_temp = self.node_het_agg(id_batch)
					out_temp = out_temp.data.cpu().numpy()
					for k in range(len(id_batch)):
						index = id_batch[k]
						if i == 0:
							embed_file.write('i' + str(index) + " ")
						for l in range(embed_d - 1):
							embed_file.write(str(out_temp[k][l]) + " ")
						embed_file.write(str(out_temp[k][-1]) + "\n")

				if i == 0:
					id_batch = self.i_train_id_list[batch_number * save_batch_s:]
					if id_batch.size == 0:
						embed_file.close()
						continue
					out_temp = self.node_het_agg(id_batch)
				out_temp = out_temp.data.cpu().numpy()
				for k in range(len(id_batch)):
					index = id_batch[k]
					if i == 0:
						embed_file.write('i' + str(index) + " ")
					for l in range(embed_d - 1):
						embed_file.write(str(out_temp[k][l]) + " ")
					embed_file.write(str(out_temp[k][-1]) + "\n")
			embed_file.close()
			return [], [], []

		return c_agg

	def aggregate_all(self, triple_list_batch, triple_index):
		c_id_batch = [x[0] for x in triple_list_batch]

		c_agg = self.het_agg(triple_index, c_id_batch)

		return c_agg

	def forward(self, triple_list_batch, triple_index):
		c_out = self.aggregate_all(triple_list_batch, triple_index)
		return c_out

	def img_text_process(self, img_text_list, id_batch):
		device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		batch_len = len(id_batch[0])
		srcpadid = self.vocab_dic['<pad>']
		start = time.time()
		topk = 1 
		thre = 0.0 
		objdim = args.objdim
		t1 = time.time()
		src = self.text_to_id(img_text_list, batch_len)
		sources, source_masks = prepare_sources(src, srcpadid)
		sources = sources.to(device)
		source_masks = source_masks.to(device)
		boxfeats = [img_text_list[i]['features'] for i in range(0, batch_len)]
		boxprobs = [img_text_list[i]['objects_conf'] for i in range(0, batch_len)]
		imgs = [img_text_list[i]['img_id'] for i in range(0, batch_len)]
		aligns = [img_text_list[i]['graph'] for i in range(0, batch_len)]
		regions_num = []
		for align in aligns:
			if len(align) == 0:
				regions_num.append(1)
			else:
				regions_num.append(align[-1][1] + 1)

		obj_feat = sources.new_zeros(sources.size(0), 15, topk,
									 objdim).float()
		obj_mask = source_masks.new_zeros(sources.size(0), 15 * topk)

		matrix = sources.new_zeros(sources.size(0), sources.size(1), 15 * topk).float()

		for ib, img in enumerate(imgs):
			boxfeat = torch.tensor(boxfeats[ib]).reshape(-1, topk, objdim)
			img_boxprobs = torch.tensor(boxprobs[ib])
			ge_thre = (img_boxprobs >= thre).byte()
			ge_thre[list(range(0, ge_thre.size(0), 5))] = 1
			obj_mask[ib, :15 * topk] = ge_thre[:15 * topk]
			obj_feat[ib, :15 * topk] = boxfeat[:15 * topk, :topk]
			for item in aligns[ib]:
				objixs = sources.new_tensor([n + item[1] * topk for n in range(topk)])
				if item[0] >= args.fix_length:
					continue
				matrix[ib, item[0], objixs] = ge_thre[objixs].float().cuda()

		obj_feat = obj_feat.view(sources.size(0), -1, objdim)
		obj_mask = obj_mask.unsqueeze(1)
		obj_feat = obj_feat.to(device)
		obj_mask = obj_mask.to(device)
		matrix = matrix.to(device)

		return sources, source_masks, obj_feat, obj_mask, matrix

	def text_to_id(self, img_text_list, batch_len):
		text_list = [img_text_list[i]['text'] for i in range(0, batch_len)]
		text_id_list = []
		for i in range(0, len(text_list)):
			text = text_list[i]
			text_id_list.append([])
			word_list = text.split(' ')
			count = 0
			for word in word_list:
				if count < args.fix_length:
					if word in self.vocab_dic:
						text_id_list[i].append(self.vocab_dic[word])
					else:
						text_id_list[i].append(self.vocab_dic['<unk>'])
				else:
					break
				count += 1
			while count < 50:
				text_id_list[i].append(self.vocab_dic['<pad>'])
				count += 1
		text_id = torch.tensor(text_id_list)
		return text_id

def cross_entropy_loss(c_embed_batch, pos_embed_batch, neg_embed_batch, d_label_batch, embed_d, args):
	batch_size = c_embed_batch.shape[0] * c_embed_batch.shape[1]

	c_embed = c_embed_batch.view(batch_size, 1, embed_d)
	pos_embed = pos_embed_batch.view(batch_size, embed_d, 1)
	neg_embed = neg_embed_batch.view(batch_size, embed_d, 1)

	out_p = torch.bmm(c_embed, pos_embed)
	out_n = - torch.bmm(c_embed, neg_embed)

	sum_p = F.logsigmoid(out_p)
	sum_n = F.logsigmoid(out_n)
	c_embed_batch = c_embed_batch.squeeze(dim = 0)
	d_label_batch = d_label_batch.squeeze(dim = 0)
	target = torch.ones(c_embed_batch.size(0))
	Reconstruction_loss = F.cosine_embedding_loss(c_embed_batch, d_label_batch, target)

	loss_sum = -args.loss_weight_alpha * (sum_p + sum_n) + args.loss_weight_beta * Reconstruction_loss


	return loss_sum.mean()

def cross_entropy_loss_IMGE(c_embed_batch, d_label_batch):
	target = torch.ones(c_embed_batch.size(0))
	Reconstruction_loss = F.cosine_embedding_loss(c_embed_batch, d_label_batch, target)
	return Reconstruction_loss

