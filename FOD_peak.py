import numpy as np
from scipy.sparse.csgraph import connected_components
import math

# detect number of peaks and their positions in FOD
# idx: indices of nearest neighboring grid points for each grid point
# nbhd: consider only the top nbhd nearest neighboring grid points to identify local maximal peaks
# thresh: eliminate local peaks lower than thresh * highest peak
# degree: merge local peaks within degree together (use mean position as final position)
# peak_cut: largest number of detected peaks (with top peak values)
def FOD_Peak(fod, idx, nbhd, thresh, degree, pos, sampling_index, return_peak_pos, peak_cut = float('inf')):

	available_index = np.ones(len(fod))
	peak_idx = []
	for i in sampling_index:
		if available_index[i] == 1:
			nbidx = idx[i, :nbhd]
			low_nbidx = nbidx[fod[nbidx] < fod[i]]
			available_index[low_nbidx] = 0
			if len(low_nbidx) == nbhd-1:
				peak_idx.append(i)
	peak_idx = np.array(peak_idx)

	if len(peak_idx) == 0:
		if return_peak_pos:
			return 0, np.array([[]]*3)
		else:
			return 0

	peak_idx = peak_idx[fod[peak_idx] > thresh*np.max(fod)]
	if len(peak_idx) > peak_cut:
		peak_idx = peak_idx[np.argsort(-fod[peak_idx])[:peak_cut]]
	peak_pos = pos[:, peak_idx]
	peak_value = fod[peak_idx]

	peak_dis = peak_pos.T.dot(peak_pos)
	peak_comp = (peak_dis > math.cos(degree*math.pi/180)) | (peak_dis < -math.cos(degree*math.pi/180))

	num_comp, idx_comp = connected_components(peak_comp)
	if return_peak_pos:
		peak_pos_final = np.zeros((3, num_comp))
		for i in range(num_comp):
			peak_pos_tmp = peak_pos[:, idx_comp==i].dot(peak_value[idx_comp==i])
			peak_pos_final[:, i] = peak_pos_tmp/np.linalg.norm(peak_pos_tmp)
		return num_comp, peak_pos_final
	else:
		return num_comp

# detect whether each voxel has at least one nearest-neighbor voxels with the same number of detected peaks on 2D/3D grid
def Num_Indicator(num_mat, active_index_list):

	n_index = num_mat.shape
	num_indicator = np.zeros(n_index)
	for k in active_index_list:
		for i in range(len(n_index)):
			if k[i] < n_index[i]-1:
				k_nb = np.array(k)
				k_nb[i] += 1
				if num_mat[k] == num_mat[tuple(k_nb)]:
					num_indicator[k] = 1
					break
			if k[i] > 0:
				k_nb = np.array(k)
				k_nb[i] -= 1
				if num_mat[k] == num_mat[tuple(k_nb)]:
					num_indicator[k] = 1
					break
		
	return num_indicator