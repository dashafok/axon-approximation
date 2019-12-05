import numpy as np
from axon_approximation import axon_algorithm
from axon_model import AxonNetwork
import torch

if __name__ == "__main__":
	xs = np.linspace(0,1, 1000)[:,None]
	ys = np.sin(xs).flatten()
	bs, bs_coefs, r, coefs, norms, errs = axon_algorithm(xs, ys, 10)
	model = AxonNetwork(torch.from_numpy(xs.astype(np.float32)), torch.from_numpy(ys.astype(np.float32)), bs_coefs, r, coefs, norms, bs)
	print("Error:{0:.6f}".format(np.linalg.norm(model(torch.from_numpy(xs.astype(np.float32))).data.numpy().squeeze() - ys)/ np.linalg.norm(ys)))