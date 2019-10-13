import numpy as np
import itertools
import time

# standardize fod to have summation one
def fod_stand(fod):
	fod_st = fod.copy()
	fod_st[fod_st < 0] = 0
	fod_st /= np.sum(fod_st)
	return fod_st

# Hellinger distance between two pdfs f1 and f2
def hellinger_dis(f1, f2):
	return max(np.sqrt(0.5 * np.sum((np.sqrt(f1) - np.sqrt(f2))**2)), 1e-4)

# l2 distance between two pdfs f1 and f2
def l2_dis(f1, f2):
	return max(np.linalg.norm(f1-f2), 1e-4)

# calculate minimum-nearest-neighbor distance for each voxel on 2D or 3D grid
# fod: FODs on 2D or 3D grid
# weight_min: minimum-nearest-neighbor distance on 2D or 3D grid
def MNN_Dist(fod, dis_func):

	n_index = fod.shape[:-1]
	index_list = list(np.ndindex(n_index))
	fod_st = np.zeros(fod.shape)
	weight_min = np.zeros(n_index)

	for k in index_list:
		fod_st[k] = fod_stand(fod[k])

	for k in index_list:
		weight_cand = []
		for i in range(len(n_index)):
			if k[i] < n_index[i]-1:
				k_nb = np.array(k)
				k_nb[i] += 1
				weight_cand.append(dis_func(fod_st[k], fod_st[tuple(k_nb)]))
			if k[i] > 0:
				k_nb = np.array(k)
				k_nb[i] -= 1
				weight_cand.append(dis_func(fod_st[k], fod_st[tuple(k_nb)]))
		weight_min[k] = np.min(np.array(weight_cand))

	return weight_min

# calculate kernel weighted DWI signal at kth voxel
def DWI_Weight(k, DWI, h, b, fod_st, dis_func, weight_ratio, dis_scale):

	n_index = DWI.shape[:-1]
	weight_loc = np.zeros(n_index)
	weight_st = np.zeros(n_index)

	range_low = np.maximum(np.zeros(len(n_index)), np.ceil(np.array(k)-h)).astype(int)
	range_up = np.minimum(n_index, np.floor(np.array(k)+h+1)).astype(int)
	range_list = list(itertools.product(*[range(range_low[i], range_up[i]) for i in range(len(n_index))]))

	for l in range_list:
		weight_loc[l] = max(0, 1 - sum((dis_scale*(np.array(k)-np.array(l)))**2) / h**2)
		if weight_loc[l] > 0:
			weight_st[l] = np.exp(-b / weight_ratio[k]**2 * dis_func(fod_st[k], fod_st[l])**2)
	weight_st[k] = 1
	weight = weight_loc * weight_st
	weight /= np.sum(weight)

	DWI_weight = np.zeros(DWI.shape[-1])
	for l in range_list:
		DWI_weight += weight[l] * DWI[l]

	return DWI_weight