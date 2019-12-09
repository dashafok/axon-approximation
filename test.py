import numpy as np
from axon_approximation import axon_algorithm
from axon_model import AxonNetwork, train_random_model
import torch
import nevergrad as ng


# returns OnePlusOne optimizer for given variable shape, other arguments can be specified
def get_opt_oneplus_one(n, **kwargs):
    return ng.optimizers.OnePlusOne(instrumentation=n, **kwargs)

if __name__ == "__main__":
	xs = np.linspace(0,1, 1000)[:,None]
	ys = np.sin(20*xs).flatten()
	bs, bs_coefs, r, coefs, norms, errs = axon_algorithm(xs, ys, 10, lambda n: get_opt_oneplus_one(n, budget=1200))
	model = AxonNetwork(torch.from_numpy(xs.astype(np.float32)), torch.from_numpy(ys.astype(np.float32)), bs_coefs, r, coefs, norms, bs)
	print("Error:{0:.6f}".format(np.linalg.norm(model(torch.from_numpy(xs.astype(np.float32))).data.numpy().squeeze() - ys)/ np.linalg.norm(ys)))
	err = train_random_model(xs, lambda x: np.sin(20*x), 10, 1)
	print(err)