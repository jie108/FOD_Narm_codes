import numpy as np
from simu_region import hellinger_dis_region, angular_error_region

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

# specify path of FOD estimation results
fod_path = "/Users/jlyang/Documents/FOD/DMRI_code/Results_new/ROI/"  # need SPECIFY case by case

# load theta_fib, phi_fib, fib_indi and dirac_plot
ROI_mat = np.load(fod_path+'ROI.npz')  # need SPECIFY case by case
theta_fib, phi_fib, fib_indi = ROI_mat['theta_fib'], ROI_mat['phi_fib'], ROI_mat['fib_indi']
dirac_plot = ROI_mat['dirac_plot']

# load FOD estimation result (using noiseless DWI)
fod_all_no = np.load(fod_path+"fod_all_no.npy")  # need SPECIFY case by case
# evaluate FOD estimation result by calculating Hellinger distance between estimated FOD and true FOD (diff_true)
hellinger_dis_region(np.expand_dims(fod_all_no, axis=-2), fod_all_no, dirac_plot)
# evaluate FOD estimation result by calcuating angular errors
angular_error_region(fod_all_no, theta_fib, phi_fib, fib_indi, idx, nbhd, thresh, degree, pos, sampling_index)

# load FOD estimation result (including voxel-wise and spacially-smoothed)
fod_all = np.load(fod_path+"fod_all_a15_b2_stop.npy")  # need SPECIFY case by case
# evaluate FOD estimation result by calculating Hellinger distance between estimated FOD and estimated FOD with 
# noiseless DWI (diff_noiseless), and Hellinger distance between estimated FOD and true FOD (diff_true)
hellinger_dis_region(fod_all, fod_all_no, dirac_plot)
# evaluate FOD estimation result (voxel-wise) by calcuating angular errors
angular_error_region(fod_all[..., 0, :], theta_fib, phi_fib, fib_indi, idx, nbhd, thresh, degree, pos, sampling_index)
# evaluate FOD estimation result (spacially smoothed) at step 6 by calcuating angular errors
angular_error_region(fod_all[..., 6, :], theta_fib, phi_fib, fib_indi, idx, nbhd, thresh, degree, pos, sampling_index)
# evaluate FOD estimation result (spacially smoothed) at step S by calcuating angular errors
angular_error_region(fod_all[..., -1, :], theta_fib, phi_fib, fib_indi, idx, nbhd, thresh, degree, pos, sampling_index)