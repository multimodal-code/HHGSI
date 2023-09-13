import math
from collections import OrderedDict, Counter
from itertools import chain

import torch
import os

from torchtext import data, datasets
from torchtext.data import Example
from contextlib import ExitStack
import pickle


def read_multimodalgraph_data(args):
	img_text_file = open(args.data_path + args.data_set + '/' + "img_text.pkl", "rb")
	vocab_file = open(args.data_path + args.data_set + '/' + "vocab.pkl", "rb")
	img_text_dic = pickle.load(img_text_file, encoding='iso-8859-1')
	vocab_dic = pickle.load(vocab_file, encoding='iso-8859-1')
	img_text_list = list(img_text_dic.values())
	return img_text_list, vocab_dic


# load the dataset + reversible tokenization
class NormalField(data.Field):

	def build_vocab(self, *args, **kwargs):
		counter = Counter()
		sources = []
		for arg in args:
			sources += [getattr(arg, name) for name, field in
						arg.fields.items() if field is self]
		for data in sources:
			for x in data:
				if not self.sequential:
					x = [x]
				try:
					counter.update(x)
				except TypeError:
					counter.update(chain.from_iterable(x))
		specials = list(OrderedDict.fromkeys(
			tok for tok in [self.unk_token, self.pad_token, self.init_token,
							self.eos_token] + kwargs.pop('specials', [])
			if tok is not None))
		self.vocab = self.vocab_cls(counter, specials=specials, **kwargs)

	def reverse(self, batch, unbpe=True, returen_token=False):
		if not self.batch_first:
			batch.t_()

		with torch.cuda.device_of(batch):
			batch = batch.tolist()

		batch = [[self.vocab.itos[ind] for ind in ex] for ex in batch]  # denumericalize

		def trim(s, t):
			sentence = []
			for w in s:
				if w == t:
					break
				sentence.append(w)
			return sentence

		batch = [trim(ex, self.eos_token) for ex in batch]  # trim past frst eos

		def filter_special(tok):
			return tok not in (self.init_token, self.pad_token)

		if unbpe:
			batch = [" ".join(filter(filter_special, ex)).replace("@@ ", "") for ex in batch]
		else:
			batch = [" ".join(filter(filter_special, ex)) for ex in batch]

		if returen_token:
			batch = [ex.split() for ex in batch]
		return batch


class GraphField(data.Field):
	def preprocess(self, x):
		return x.strip()

	def process(self, x, device=None):
		batch_imgs = []
		batch_alighs = []
		region_num = []
		for i in x:
			i = i.split()
			img = i[0]
			align = i[1:]
			align = list(map(lambda item: list(map(int, item.split('-'))), align))
			batch_imgs.append(img)
			batch_alighs.append(align)
			if len(align) == 0:
				region_num.append(1)
			else:
				region_num.append(align[-1][-1] + 1)
		return batch_imgs, batch_alighs, region_num


class TranslationDataset(data.Dataset):
	"""Defines a dataset for machine translation."""

	@staticmethod
	def sort_key(ex):
		return data.interleave_keys(len(ex.src), len(ex.trg))

	def __init__(self, path, exts, fields, **kwargs):
		"""Create a TranslationDataset given paths and fields.
		Arguments:
			path: Common prefix of paths to the data files for both languages.
			exts: A tuple containing the extension to path for each language.
			fields: A tuple containing the fields that will be used for data
				in each language.
			Remaining keyword arguments: Passed to the constructor of
				data.Dataset.
		"""
		if not isinstance(fields[0], (tuple, list)):
			fields = [('src', fields[0]), ('trg', fields[1])]

		src_path, trg_path = tuple(os.path.expanduser(path + '.' + x) for x in exts)

		examples = []
		with open(src_path, encoding='utf-8') as src_file, open(trg_path, encoding='utf-8') as trg_file:
			for src_line, trg_line in zip(src_file, trg_file):
				src_line, trg_line = src_line.strip(), trg_line.strip()
				if src_line != '' and trg_line != '':
					examples.append(data.Example.fromlist(
						[src_line, trg_line], fields))

		super(TranslationDataset, self).__init__(examples, fields, **kwargs)

	@classmethod
	def splits(cls, path, exts, fields, root='.data', train='train', validation='val', test='test', **kwargs):
		"""Create dataset objects for splits of a TranslationDataset.
		Arguments:
			root: Root dataset storage directory. Default is '.data'.
			exts: A tuple containing the extension to path for each language.
			fields: A tuple containing the fields that will be used for data
				in each language.
			train: The prefix of the train data. Default: 'train'.
			validation: The prefix of the validation data. Default: 'val'.
			test: The prefix of the test data. Default: 'test'.
			Remaining keyword arguments: Passed to the splits method of
				Dataset.
		"""

		train_data = None if train is None else cls(
			os.path.join(path, train), exts, fields, **kwargs)
		val_data = None if validation is None else cls(
			os.path.join(path, validation), exts, fields, **kwargs)
		test_data = None if test is None else cls(
			os.path.join(path, test), exts, fields, **kwargs)
		return tuple(d for d in (train_data, val_data, test_data)
					 if d is not None)


class ParallelDataset(data.Dataset):

	@staticmethod
	def sort_key(ex):
		return data.interleave_keys(len(ex.src), 0)

	def __init__(self, path, exts, fields, max_len=None, **kwargs):
		assert len(exts) == len(fields), 'N parallel dataset must match'
		self.N = len(fields)

		if not isinstance(fields[0], (tuple, list)):
			newfields = [('src', fields[0]), ('lb', fields[1])]
			for i in range(len(exts) - 2):
				newfields.append(('extra_{}'.format(i), fields[2 + i]))
			fields = newfields

		paths = tuple(os.path.expanduser(path + '.' + x) for x in exts)
		examples = []

		with ExitStack() as stack:
			files = [stack.enter_context(open(fname, encoding='utf-8')) for fname in paths]
			for i, lines in enumerate(zip(*files)):
				lines = [line.strip() for line in lines]
				if not any(line == '' for line in lines):
					example = Example.fromlist(lines, fields)
					examples.append(example)
		super(ParallelDataset, self).__init__(examples, fields, **kwargs)


if __name__ == '__main__':
	hh = read_multimodalgraph_data(1)
	print(hh)