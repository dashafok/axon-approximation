import numpy as np
import nevergrad as ng


def relu(x):
	z = x
	z[x <= 0] = 0.0
	return z

def objective(w, x, res, nonlinearity=relu):
	'''
	objective function for axon_algorithm
	'''
	new_bas = nonlinearity(x@w)
	# first, orthogonalize:
	new_bas = new_bas - x@(x.T@new_bas)
	if (np.dot(new_bas.flatten(), new_bas.flatten()) < 1e-7):
		return 100
	return -(new_bas.flatten()@res.flatten())**2/(new_bas.flatten()@new_bas.flatten()) + 1e-8*(np.dot(new_bas.flatten(), new_bas.flatten())-1)**2


def axon_algorithm(xs, ys, K):
	'''
	Greedy algorithm for function approximation	

	Args:
		xs: numpy array of points
		ys: function values
		K: number of basis function to compute

	Returns:
		bs: basis values
		bs_coef: list of coeffients for constructing each following basis function
		r: R matrix form QR decomposition of [1,x]
		orth_coef: coefficitients to perform Gram-Schmitdt orthogonalization
		orth_norms: norms of basis functions
		errors: list of relative errors for each new basis function
	'''

	# intial basis [1,x]:
	
	ns = xs.shape[0]
	bs = np.hstack([np.ones(ns)[:,None], xs])
	bs, r = np.linalg.qr(bs)

	bs_coef = []
	orth_coef = []
	orth_norms = []
	errors = []
	res = ys - bs@bs.T@ys

	for i in range(K):

		# solve optimization problem:
		optimizer = ng.optimizers.OnePlusOne(instrumentation=bs.shape[1], budget=1200)
		opt_res = optimizer.minimize(lambda x: objective(x, bs, res))
		x0 = opt_res.args[0]
		x0 = x0/np.linalg.norm(x0)
		
		# orthogonalize:
		new_bas = relu(bs@x0)

		# orthogonalize and remember coefficients, whiche are later needed for inference:
		orth_coef.append(bs.T@new_bas)
		new_bas = new_bas - bs@bs.T@new_bas

		# normalize and remember norms, as they will be needed for inference:
		orth_norms.append(np.linalg.norm(new_bas))
		new_bas = new_bas/np.linalg.norm(new_bas)

		bs = np.hstack([bs, new_bas.reshape(-1, 1)])
		bs_coef.append(x0)

		res = ys - bs@bs.T@ys
		errors.append(np.linalg.norm(res)/np.linalg.norm(ys))
	
	return bs, bs_coef, r, orth_coef, orth_norms, errors




