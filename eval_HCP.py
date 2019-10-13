import numpy as np
import pandas as pd
from DWI_weight import fod_stand, hellinger_dis
from FOD_peak import FOD_Peak

# evaluate FOD estimation result by calculating Hellinger distance between estimated FOD and estimated FOD with 
# noiseless DWI (diff_noiseless), and Hellinger distance between estimated FOD and true FOD (diff_true)
def hellinger_dis_retest(fod_all, fod_all_retest):

	n_index, S = fod_all.shape[:-2], fod_all.shape[-2]-1
	diff = np.zeros(n_index + (S+1,))

	for k in list(np.ndindex(n_index)):
		for s in range(S+1):
			diff[k][s] = hellinger_dis(fod_stand(fod_all[k][s]), fod_stand(fod_all_retest[k][s]))

	diff_axis = tuple(range(len(n_index)))
	diff_mean, diff_std = diff.mean(axis=diff_axis), diff.std(axis=diff_axis, ddof=1)

	diff = {"smooth_round":range(S+1), "diff(mean)":diff_mean, "diff(std)":diff_std}
	diff_table = pd.DataFrame(diff, columns=diff.keys())
	diff_table["smooth_round"] = diff_table["smooth_round"].apply(lambda x: "{:.0f}".format(x))
	diff_table.iloc[:, 1:] = diff_table.iloc[:, 1:].applymap(lambda x: "{:.3f}".format(x))
	print(diff_table)


# evaluate FOD estimation result by calcuating angular errors
def angular_error_retest(fod, fod_retest, idx, nbhd, thresh, degree, pos, sampling_index, peak_cut):

	n_index = fod.shape[:-1]
	index_list = list(np.ndindex(n_index))

	# nfib: number of fibers in estimated FOD using peak detection function FOD_Peak
	# peak_pos: 3D coordinates of peaks in estimated FOD using peak detection function FOD_Peak
	nfib, peak_pos = np.zeros(n_index), {}
	for k in index_list:
		nfib[k], peak_pos[k] = FOD_Peak(fod[k], idx, nbhd, thresh, degree, pos, sampling_index, True, peak_cut)
	n0fib, n1fib, n2fib = np.sum(nfib==0), np.sum(nfib==1), np.sum(nfib==2)

	nfib_retest, peak_pos_retest = np.zeros(n_index), {}
	for k in index_list:
		nfib_retest[k], peak_pos_retest[k] = FOD_Peak(fod_retest[k], idx, nbhd, thresh, degree, pos, sampling_index, True, peak_cut)

	angle_error_1fib, angle_error_2fib, sep_error = [], [], []
	ind_1fib, ind_2fib = [], []
	for k in index_list:
		if nfib[k] > 0 and peak_pos[k].shape[1] == peak_pos_retest[k].shape[1]:
			if peak_pos[k].shape[1] == 1:
				ind_1fib.append(k)
				angle_error_1fib.append(np.arccos(np.minimum(np.abs(peak_pos[k].T.dot(peak_pos_retest[k])), 1))/np.pi*180)
			if peak_pos[k].shape[1] == 2:
				ind_2fib.append(k)
				angle_mat = np.arccos(np.minimum(np.abs(peak_pos[k].T.dot(peak_pos_retest[k])), 1))/np.pi*180
				angle_error_cand1, angle_error_cand2 = [angle_mat[0,0], angle_mat[1,1]], [angle_mat[0,1], angle_mat[1,0]]
				if np.sum(angle_error_cand1) <= np.sum(angle_error_cand2):
					angle_error_2fib += angle_error_cand1 
				else:
					angle_error_2fib += angle_error_cand2
				sep_error.append(np.abs(np.arccos(np.abs(peak_pos[k][:,0].T.dot(peak_pos[k][:,1]))) 
					- np.arccos(np.abs(peak_pos_retest[k][:,0].T.dot(peak_pos_retest[k][:,1]))))/np.pi*180)

	if n0fib>0:
		#print("0-fiber: number {}, correct/under/over_rate {:.2f}/{:.2f}/{:.2f}".format(
		#	n0fib, np.sum(nfib_retest[nfib==0]==0)/n0fib, np.sum(nfib_retest[nfib==0]<0)/n0fib, np.sum(nfib_retest[nfib==0]>0)/n0fib))
		print("0-fiber: overlap_ratio {:.2f}".format(
			np.sum((nfib==0)&(nfib_retest==0)) / np.sum((nfib==0)|(nfib_retest==0))))
	if n1fib>0:
		#print("1-fiber: number {}, correct/under/over_rate {:.2f}/{:.2f}/{:.2f}, angle_error(mean/median) {:.2f}/{:.2f}".format(
		#	n1fib, np.sum(nfib_retest[nfib==1]==1)/n1fib, np.sum(nfib_retest[nfib==1]<1)/n1fib, np.sum(nfib_retest[nfib==1]>1)/n1fib, 
		#	np.mean(angle_error_1fib), np.median(angle_error_1fib)))
		print("1-fiber: overlap_ratio {:.2f}, angle_error(mean/median) {:.2f}/{:.2f}".format(
			np.sum((nfib==1)&(nfib_retest==1)) / np.sum((nfib==1)|(nfib_retest==1)), np.mean(angle_error_1fib), np.median(angle_error_1fib)))
	if n2fib>0:
		#print("2-fiber: number {}, correct/under/over_rate {:.2f}/{:.2f}/{:.2f}, angle_error(mean/median) {:.2f}/{:.2f}, separation_error(mean/median) {:.2f}/{:.2f}".format(
		#	n2fib, np.sum(nfib_retest[nfib==2]==2)/n2fib, np.sum(nfib_retest[nfib==2]<2)/n2fib, np.sum(nfib_retest[nfib==2]>2)/n2fib, 
		#	np.mean(angle_error_2fib), np.median(angle_error_2fib), np.mean(sep_error), np.median(sep_error)))
		print("2-fiber: overlap_ratio {:.2f}, angle_error(mean/median) {:.2f}/{:.2f}, separation_error(mean/median) {:.2f}/{:.2f}".format(
			np.sum((nfib==2)&(nfib_retest==2)) / np.sum((nfib==2)|(nfib_retest==2)),
			np.mean(angle_error_2fib), np.median(angle_error_2fib), np.mean(sep_error), np.median(sep_error)))

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
fod_path = "/Users/jlyang/Documents/FOD/DMRI_code/Results_new/ROI_HCP2/"  # need SPECIFY case by case

# load FOD estimation result (including voxel-wise and spacially-smoothed)
fod_all = np.load(fod_path+"fod_all_123K_3e3_b4.npy")  # need SPECIFY case by case
fod_all_retest = np.load(fod_path+"fod_all_retest_123K_3e3_b4.npy")  # need SPECIFY case by case
fod_all_retest = fod_all_retest[:, 1:, :-1]  # ROI2 (retest)
# evaluate FOD estimation result by calculating Hellinger distance between estimated FOD and estimated FOD with 
# noiseless DWI (diff_noiseless), and Hellinger distance between estimated FOD and true FOD (diff_true)
hellinger_dis_retest(fod_all, fod_all_retest)
# evaluate FOD estimation result (voxel-wise) by calcuating angular errors
angular_error_retest(fod_all[..., 0, :], fod_all_retest[..., 0, :], idx, nbhd, thresh, degree, pos, sampling_index, peak_cut)
# evaluate FOD estimation result (spacially smoothed) at step 6 by calcuating angular errors
angular_error_retest(fod_all[..., 6, :], fod_all_retest[..., 6, :], idx, nbhd, thresh, degree, pos, sampling_index, peak_cut)
# evaluate FOD estimation result (spacially smoothed) at step S by calcuating angular errors
angular_error_retest(fod_all[..., -1, :], fod_all_retest[..., -1, :], idx, nbhd, thresh, degree, pos, sampling_index, peak_cut)