import numpy as np
from FOD_peak import FOD_Peak

# specify parameters for peak detection in angular_error_region
pos_mat = np.load("pos.npz")
pos = pos_mat["pos_plot"]
sampling_index = pos_mat["sampling_index"]
dis = pos.T.dot(pos)
idx = np.zeros(dis.shape, dtype=int)
for i in range(dis.shape[0]):
	idx[i, :] = np.argsort(-dis[i, :])
nbhd = 40
thresh = 0.25
degree = 5
peak_cut = 4

# specify path of FOD estimation results
fod_path = "/Users/jlyang/Documents/FOD/DMRI_code/Results_new/ROI_HCPM2/"  # need SPECIFY case by case

#fod_name = "123K_3e3_b4"
#save_name = "123K_3e3_b4_sm6"
# load FOD estimation result (including voxel-wise and spacially-smoothed)
#fod_all = np.load(fod_path+"fod_all_{}.npy".format(fod_name))  # need SPECIFY case by case
#fod_s = np.squeeze(fod_all[..., 6, :])

fod_name = "123K_3e3_b4"
save_name = "123K_3e3_b4_sm6"
#save_name = "retest_123K_3e3_b4_sm6"
# load FOD estimation result (including voxel-wise and spacially-smoothed)
fod_all = np.load(fod_path+"fod_all_{}.npy".format(fod_name))  # need SPECIFY case by case
#fod_s = np.squeeze(fod_all[..., 0, :])
fod_s = np.squeeze(fod_all[..., 6, :])
#fod_s = np.squeeze(fod_all[:, 1:, :-1, 0, :])
#fod_s = np.squeeze(fod_all[:, 1:, :-1, 6, :])

# set dimensions of each voxel according to real data
#xgrid_sp, ygrid_sp, zgrid_sp = 1, 1, 1
xgrid_sp, ygrid_sp, zgrid_sp = 1.25, 1.25, 1.25
#xgrid_sp, ygrid_sp, zgrid_sp = 1.3672, 1.3672, 2.7

n1, n2, n3, _ = fod_s.shape
braingrid = np.zeros((3, n1, n2, n3))
for i in range(n1):
	for j in range(n2):
		for k in range(n3):
			braingrid[:, i, j, k] = [(i-0.5*(n1-1))*xgrid_sp, (j-0.5*(n2-1))*ygrid_sp, (k-0.5*(n3-1))*zgrid_sp]

n_fiber, rmap = np.zeros(n1*n2*n3), np.zeros(n1*n2*n3)
vec, loc, map = np.array([[]]*3).T, np.array([[]]*3).T, np.array([])

for k in range(n3):
	for j in range(n2):
		for i in range(n1):
			ind = k*n1*n2+j*n1+i
			n_fiber[ind], peak_pos = FOD_Peak(fod_s[i, j, k], idx, nbhd, thresh, degree, pos, sampling_index, True, peak_cut)
			if n_fiber[ind] > 0:
				vec = np.vstack((vec, peak_pos.T))
				loc = np.vstack((loc, np.tile(braingrid[:, i, j, k], (int(n_fiber[ind]), 1))))
				map = np.concatenate((map, [ind+1]*int(n_fiber[ind])))
			else:
				vec = np.vstack((vec, [np.nan]*3))
				loc = np.vstack((loc, braingrid[:, i, j, k]))
				map = np.concatenate((map, [ind+1]))

n_fiber2 = np.array([])
for i in range(n1*n2*n3):
	rmap[i] = np.where(map == (i+1))[0][0]+1
	n_fiber2 = np.concatenate((n_fiber2, [n_fiber[i]]*max(int(n_fiber[i]), 1)))

np.savez(fod_path+"track_{}.npz".format(save_name), vec = vec.T.reshape(-1), loc = loc.T.reshape(-1), 
	n_fiber = n_fiber, n_fiber2 = n_fiber2, map = map, rmap = rmap, braingrid = braingrid.T.reshape(-1), 
	grid_sp = np.array([xgrid_sp, ygrid_sp, zgrid_sp]).astype('float'), ndim = np.array([n1, n2, n3]).astype('float'))