import re
import torch
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F

import time
import subprocess
import pickle
from .optimizer import NoamOpt, CommonOpt
from .util import *


class GATEncoder(nn.Module):
	def __init__(self, d_model, d_hidden, n_heads, src_vocab, input_dp, dropout=0.1, layer=2):
		super(GATEncoder, self).__init__()
		self.layer = layer
		self.dp = dropout
		self.d_model = d_model
		self.hid = d_hidden
		self.src_vocab = src_vocab
		self.input_dp = input_dp
		objcnndim = 2048

		self.src_emb_pos = nn.Sequential(Embeddings(d_model, src_vocab), PositionalEncoding(d_model, input_dp))

		self.trans_obj = nn.Sequential(Linear(objcnndim, d_model), nn.ReLU(), nn.Dropout(dropout),
									   Linear(d_model, d_model), nn.ReLU(), nn.Dropout(dropout))

		# text
		self.mhatt_x = clone(MultiHeadedAttention(n_heads, d_model, dropout), layer)
		self.ffn_x = clone(PositionwiseFeedForward(d_model, d_hidden), layer)
		self.res4ffn_x = clone(SublayerConnectionv2(d_model, dropout), layer)
		self.res4mes_x = clone(SublayerConnectionv2(d_model, dropout), layer)

		# img
		self.mhatt_o = clone(MultiHeadedAttention(n_heads, d_model, dropout, v=0, output=0), layer)
		self.ffn_o = clone(PositionwiseFeedForward(d_model, d_hidden), layer)
		self.res4mes_o = clone(SublayerConnectionv2(d_model, dropout), layer)
		self.res4ffn_o = clone(SublayerConnectionv2(d_model, dropout), layer)

		self.mhatt_x2o = clone(Linear(d_model * 2, d_model), layer)
		self.mhatt_o2x = clone(Linear(d_model * 2, d_model), layer)
		self.xgate = clone(SublayerConnectionv2(d_model, dropout), layer)
		self.ogate = clone(SublayerConnectionv2(d_model, dropout), layer)

		# visual guide cross attention
		self.mhatt_o2o = clone(Linear(1024, 1920), layer)
		self.mhatt_o2ox = clone(MultiHeadedAttention(n_heads, d_model, dropout, v=0, output=0), layer)
		self.res4o2ox = clone(SublayerConnectionv2(d_model, dropout), layer)

		self.Linearlayer = nn.Sequential(
			nn.Linear(1920, 512, bias=True),
			nn.ReLU(inplace=True),
			nn.Dropout(p=0.5, inplace=False),
			nn.Linear(512, 1024, bias=True)
		)


	def forward(self, x, mask, o_clip, *objs):
		x = self.src_emb_pos(x) 
		#              B 1 O     B T O
		obj_feats, _, obj_mask, matrix = objs

		o = self.trans_obj(obj_feats)
		matrix = matrix.unsqueeze(-1)
		# B O T
		matrix4obj = torch.transpose(matrix, 1, 2)

		batch, objn, xn = matrix4obj.size(0), matrix4obj.size(1), matrix4obj.size(2)

		for i in range(self.layer):
			# Textual Self-attention, for text node
			newx = self.res4mes_x[i](x, self.mhatt_x[i](x, x, x, mask))

			# Visual Self-attention, for image node
			newo = self.res4mes_o[i](o, self.mhatt_o[i](o, o, o, obj_mask))
			testo = newo

			x_ep = newx.unsqueeze(1).expand(batch, objn, xn, newx.size(-1))
			newo_ep = newo.unsqueeze(2).expand(batch, objn, xn, o.size(-1))
			# B O T H
			o2x_gates = torch.sigmoid(self.mhatt_o2x[i](torch.cat((x_ep, newo_ep), -1)))
			o2x = (o2x_gates * matrix4obj * x_ep).sum(2)

			newo = self.ogate[i](newo, o2x)

			# visual guide cross attention
			o_all = self.mhatt_o2o[i](o_clip)
			try:
				o_all = o_all.view(batch, objn, newo.size(-1))
			except RuntimeError:
				o_all = o_all.view(batch, objn, newo.size(-1))
			newo = self.res4o2ox[i](newo, self.mhatt_o2ox[i](newo, o_all, o_all, obj_mask))

			o = self.res4ffn_o[i](newo, self.ffn_o[i](newo))


		o = o.view(o.size()[0], -1) 
		output = self.Linearlayer(o)

		output = torch.sigmoid(output)

		return  output


def print_params(model):
	print('total parameters:', sum([np.prod(list(p.size())) for p in model.parameters()]))



def getBinaryTensor(imgTensor, boundary=0.5):
	one = torch.ones_like(imgTensor)
	zero = torch.zeros_like(imgTensor)
	return torch.where(imgTensor > boundary, one, zero)
