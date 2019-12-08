"""Experiments with random initialization

Train Axon model with random initialization and 
save erros to pkl-file. The considered functions 
are the same as in experiments.ipynb
"""
from axon_model import init_weights
from axon_model import train_random_model

import torch
import numpy as np 
import argparse
import pickle
import os

parser = argparse.ArgumentParser('Train axon network with random initialization')
parser.add_argument('function', type=str, default='x2', help='possible values: x2, sqrt, exp, sin, 2d, diff')
parser.add_argument('--K', type=int, default=10, help='the maximal number of basis functions to add')
parser.add_argument('--num_epochs', type=int, default='1000')
parser.add_argument('--eps', type=float, help='epsilon for equation -eps^2 u''+u = 1, u(0)=u(1)=0')


# solution for -eps^2 u''+u = 1, u(0)=u(1)=0
def u(x, eps=0.05):
    a = (1-np.exp(1/eps))/(np.exp(2/eps)-1)
    b = (np.exp(1/eps)-np.exp(2/eps))/(np.exp(2/eps)-1)
    return a*np.exp(x/eps)+b*np.exp(-x/eps) + 1

def f_2d(x):
    return np.sqrt(xs[:,0]**2+xs[:,1]**2).astype(np.float64)

function_mapping_1d = {'x2': lambda x: x**2, 
					   'sqrt': lambda x: np.sqrt(x),
					   'exp': lambda x: np.exp(-x),
					   'sin': lambda x: np.sin(20*x)}



if __name__=="__main__":
	if torch.cuda.is_available():
		device = 'cuda:0'
	else:
		device = 'cpu'
	args = parser.parse_args()
	K = args.K
	num_epochs = args.num_epochs
	fname = 'error_'+args.function+'.pkl' # file to save errors
	#xs = np.linspace(0,1, 1000)[:,None]
	#err = train_random_model(xs, lambda x: np.sin(20*x), 10, 1)
	if args.function == 'diff':
		xs = np.linspace(0,1,1000).reshape(-1,1)
		eps = args.eps
		errors = {}
		errors[eps] = []

		for k in range(1,K+1):
			err_k = train_random_model(xs, lambda x: u(x, eps), k, num_epochs, device=device)
			errors[eps].append(err_k)
		if os.path.exists(fname):
			with open(fname,'rb') as f:
				errors1 = pickle.load(f)
				#print(errors1)
			with open(fname,'wb') as f:
				#print(errors1['error'].update(errors))
				errors1['error'].update(errors)
				pickle.dump(errors1, f)
		else:
			with open(fname,'wb') as f:
				pickle.dump({'error': errors}, f)

	elif args.function == '2d':
		x = np.linspace(-1,1,100)
		xx, yy = np.meshgrid(x,x)
		xs = np.hstack([xx.flatten()[:,np.newaxis], yy.flatten()[:,np.newaxis]])
		#ys = f_2d(xs)
		errors = []

		for k in range(1,K+1):
			err_k = train_random_model(xs, f_2d, k, num_epochs, device=device)
			errors.append(err_k)
			with open(fname,'wb') as f:
				pickle.dump({'error': errors}, f)
	elif args.function in function_mapping_1d.keys():
		xs = np.linspace(0,1,1000).reshape(-1,1)
		f = function_mapping_1d[args.function]
		errors = []

		for k in range(1,K+1):
			err_k = train_random_model(xs, f, k, num_epochs, device=device)
			errors.append(err_k)
		with open(fname,'wb') as f:
			pickle.dump({'error': errors}, f)
	else:
		raise NotImplementedError
	


