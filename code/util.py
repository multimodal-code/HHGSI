# coding:utf-8
import os
import numpy as np
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms
from options import options
import pickle



class Dataloader():
	def __init__(self, opt):
		self.opt = opt
		self.dirs = ['train', 'test', 'testing']

		self.means = [0, 0, 0]
		self.stdevs = [0, 0, 0]

		self.transform = transforms.Compose([transforms.Resize(opt.isize),
											 transforms.CenterCrop(opt.isize),
											 transforms.ToTensor(), 
											 ])

		self.dataset = {x: ImageFolder(os.path.join(opt.dataroot, x), self.transform) for x in self.dirs}

	def get_mean_std(self, type, mean_std_path):
		num_imgs = len(self.dataset[type])
		for data in self.dataset[type]:
			img = data[0]
			for i in range(3):
				self.means[i] += img[i, :, :].mean()
				self.stdevs[i] += img[i, :, :].std()

		self.means = np.asarray(self.means) / num_imgs
		self.stdevs = np.asarray(self.stdevs) / num_imgs

		print("{} : normMean = {}".format(type, self.means))
		print("{} : normstdevs = {}".format(type, self.stdevs))

		with open(mean_std_path, 'wb') as f:
			pickle.dump(self.means, f)
			pickle.dump(self.stdevs, f)
			print('pickle done')


if __name__ == '__main__':
	opt = options().parse()
	dataloader = Dataloader(opt)
	for x in dataloader.dirs:
		mean_std_path = 'mean_std_value_' + x + '.pkl'
		dataloader.get_mean_std(x, mean_std_path)