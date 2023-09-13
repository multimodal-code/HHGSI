import argparse

def read_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_path', type = str, default = '/home/hlf/code/open_source/hhgsi_data/',
				   help='path to data')
	parser.add_argument('--model_path', type = str, default = './model_save/',
				   help='path to save model')
	parser.add_argument('--data_set', type=str, default='CLEF',
						help='Data set utilized')
	parser.add_argument('--clip_model', type = str, default='RN50',
						help="the model clip use")
	parser.add_argument('--I_n', type=int, default= 4994, # clef: 3291 MIR: 4994 PASCAL: 6259 NUS: 71569
						help='number of image node')
	parser.add_argument('--clip_f_d', type = int, default= 1024,
					help = 'clip feature dimension')
	parser.add_argument('--in_f_d', type = int, default = 128,
				   help = 'input feature dimension')
	parser.add_argument('--embed_d', type = int, default = 1024,
				   help = 'embedding dimension')
	parser.add_argument('--lr', type = int, default = 0.01, 
				   help = 'learning rate')
	parser.add_argument('--batch_s', type = int, default = 3000, 
				   help = 'batch size')
	parser.add_argument('--mini_batch_s', type = int, default = 20, 
				   help = 'mini batch size')
	parser.add_argument('--train_iter_n', type = int, default = 100, 
				   help = 'max number of training iteration')
	parser.add_argument('--walk_n', type = int, default = 10, 
				   help='number of walk per root node')
	parser.add_argument('--walk_L', type = int, default = 30, 
				   help='length of each walk')
	parser.add_argument('--window', type = int, default = 10,
				   help='window size for relation extration')
	parser.add_argument("--random_seed", default = 10, type = int)
	parser.add_argument('--train_test_label', type= int, default = 0,
				   help='train/test label: 0 - train, 1 - test, 2 - code test/generate negative ids for evaluation')
	parser.add_argument('--save_model_freq', type = float, default = 10,
				   help = 'number of iterations to save model')
	parser.add_argument("--cuda", default = 1, type = int)
	parser.add_argument("--checkpoint", default = '', type=str)
	parser.add_argument('--objdim', type=int, default=2048)
	parser.add_argument('--fix_length', type=int, default="50", help='length of text')
	parser.add_argument('--if_link_pre', type=int, default="0", help='Whether to conduct link prediction experiments')
	parser.add_argument('--remove_edge_rate', type=float, default="0.5", help='Delete edge ratio')
	parser.add_argument('--loss_weight_alpha', type=float, default=0.8, help="weight of similar loss")
	parser.add_argument('--loss_weight_beta', type=float, default=0.8, help="weight of reconstruction loss")
	parser.add_argument('--model_name', type=str, default='HHGSI', help="Ablation experiment. HHGSI_IMGE, HHGSI_IHGNN, EHGSI_GNN, HHGSI")


	args = parser.parse_args()

	return args
