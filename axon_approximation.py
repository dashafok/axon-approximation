import numpy as np
import nevergrad as ng


def relu(x):
	z = x
	z[x <= 0] = 0.0
	return z

def repu(x, q):
	z = np.power(x,q)
	z[x <= 0] = 0.0
	return z

def obj(w, x, res, nonlinearity):
 	'''
 	Objective function for axon_algorithm (condisdered in the paper)
 	'''
 	if (np.dot(w, w) < 1e-7):
 		return 100
 	return -(nonlinearity(x@w)@res)**2/w.T@x.T@x@w + 1e-8*(np.dot(w, w)-1)**2


def obj_new(w, x, res, nonlinearity):
	'''
	Modified objective function for axon_algorithm 
	'''
	new_bas = nonlinearity(x@w)
	# first, orthogonalize:
	new_bas = new_bas - x@(x.T@new_bas)
	if (np.dot(new_bas.flatten(), new_bas.flatten()) < 1e-7):
		return 100
	return -(new_bas.flatten()@res.flatten())**2/(new_bas.flatten()@new_bas.flatten()) + 1e-8*(np.dot(new_bas.flatten(), new_bas.flatten())-1)**2


def axon_algorithm(xs, ys, K, get_optimizer, new_obj=True, nonlinearity=relu, **optimizer_args):
	'''
	Greedy algorithm for function approximation	from paper

	Args:
		xs: numpy array of points
		ys: function values
		K: number of basis function to compute
		optimizer: optimizer for minimization problem
		get_optimizer: a function returning optimizer, depending on the number of basis functions
		new_obj: True to use modified objective
		nonlinearity: nonlinearity to use, default - relu
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
	if new_obj:
		objective = obj
	else:
		objective = obj_new
	for i in range(K):

		# solve optimization problem:
		optimizer = get_optimizer(bs.shape[1])
		opt_res = optimizer.minimize(lambda x: objective(x, bs, res, nonlinearity=nonlinearity))
		x0 = opt_res.args[0]
		x0 = x0/np.linalg.norm(x0)
		
		new_bas = nonlinearity(bs@x0)
		
		# orthogonormalize
		c_orth = [bs.T@new_bas]
		new_bas = new_bas - bs@bs.T@new_bas
		norm_orth = [np.linalg.norm(new_bas)]
		new_bas = new_bas/np.linalg.norm(new_bas)
		

		if new_obj:
			# reorthonomalize
			for _ in range(2):
				c_orth.append(bs.T@new_bas)
				new_bas = new_bas - bs@bs.T@new_bas
				norm_orth.append(np.linalg.norm(new_bas))
				new_bas = new_bas/np.linalg.norm(new_bas)		
		# remember coefficients, as they will be needed for inference:
		orth_norms.append(norm_orth)
		orth_coef.append(c_orth)

		bs = np.hstack([bs, new_bas.reshape(-1, 1)])
		bs_coef.append(x0)

		res = ys - bs@bs.T@ys
		errors.append(np.linalg.norm(res)/np.linalg.norm(ys))
	
	return bs, bs_coef, r, orth_coef, orth_norms, errors

