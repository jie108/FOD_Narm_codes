import numpy as np
import scipy.io as spio
from scipy.linalg import cho_solve, cho_factor
from scipy.stats import chi2
from DWI_weight import *
from FOD_peak import *
from pyclasso import SN_CLasso
from multiprocessing import Pool
import time

# load data
data_type = "ROI"  # "ROI"/"ROIs"/"ROI3D"
noiseless = False
ADNI_data = False
HCP_data = False
#ROI_mat = spio.loadmat("{}.mat".format(data_type), squeeze_me=True)  # load data generated in matlab
ROI_mat = np.load("{}.npz".format(data_type))  # load data generated in python

# extract components in loaded data
Constraint = ROI_mat["Constraint"]  # constraint matrix (on half sphere) (half of vertex_SN)
design_SN = ROI_mat["design_SN"]  # design matrix
vertex_SN = ROI_mat["vertex_SN"]  # SN basis evaluated on dense grid (convert SN coefficients onto dense grid)
DWI = ROI_mat["DWI"]  # DWI signals
Constraint_T = np.ascontiguousarray(Constraint.T)  # transform for efficiency in C
design_SN_T = np.ascontiguousarray(design_SN.T)
vertex_SN = np.ascontiguousarray(vertex_SN)
DWI = np.ascontiguousarray(DWI)
if noiseless:
	DWI_noiseless = ROI_mat["DWI_noiseless"]
	DWI_noiseless = np.ascontiguousarray(DWI_noiseless)
#if real_data:
#	mask = ROI_mat["mask"]

n_index = DWI.shape[:-1]  # tuple to record ROI dimension
l, p = vertex_SN.shape  # l: number of dense grid points, p: number of SN basis

# grid for lambda selection and stopping rule
# grid search for lambda stops once the average relative change of RSS across stop_percent*L consecutive steps is 
# smaller than stop_thresh
if ADNI_data or HCP_data:
	L = 120
	log_seq = np.linspace(0.5, -5, L)
	lambda_seq = 10**log_seq  # lambda grid
	rho_seq = lambda_seq*np.maximum(np.ceil(-log_seq), np.ones(L))  # rho grid in ADMM
	stop_percent = 0.05
	stop_thresh = 1e-3 if ADNI_data else 3e-3
else:
	L = 100
	log_seq = np.linspace(0, -5, L)
	lambda_seq = 10**log_seq
	rho_seq = 2*lambda_seq
	stop_percent = 0.05
	stop_thresh = 1e-3

# parameters in ADMM
ep_a = 1e-4  # absolute tolerance in ADMM stopping rule
ep_r = 1e-2  # relative tolerance in ADMM stopping rule
maxit = 5000  # maximum iteration steps in ADMM iteration
verbose = 0

# neighborhood expansion parameter in smoothing
ch = 1.15
if ADNI_data:
	dis_scale = np.array([1, 1, 2])
else:
	dis_scale = np.ones(len(n_index))
# maximum smoothing step
if noiseless:
	S = 0  # S=0 if no smoothing
else:
	S = 10 if len(n_index) == 2 else 6  # S=10 in 2D grid and S=6 in 3D grid

# other parameters in smoothing
b = 2 if len(n_index) == 2 else 4  # gamma parameter in similarity kernel; b=2 in 2D grid and b=4 in 3D grid
dis_func = hellinger_dis  # distance function in similarity kernel
scaling = True
q = 0.15  # parameter in weight rescaling (rescale top q and bottom q extreme smoothing cases)
stop_MNN = True  # stopping rule based on minimum-nearest-neighbor distance
stop_seq = False  # stopping rule in PMARM
a_seq = 3  # parameter in stopping rule in PMARM

parallel = True

# parameters for peak detection in smoothing
#pos_mat = spio.loadmat("pos.mat", squeeze_me=True)  # load data generated in matlab
pos_mat = np.load("pos.npz")  # load data generated in python
pos = pos_mat["pos_plot"]  # positions of grid points on dense grid
sampling_index = pos_mat["sampling_index"]  # indices of half-sphere grid points
dis = pos.T.dot(pos)
idx = np.zeros(dis.shape, dtype=int)
for i in range(l):
	idx[i, :] = np.argsort(-dis[i, :])  # indices of nearest neighboring grid points for each grid point
nbhd = 40  # consider only the top nbhd nearest neighboring grid points to identify local maximal peaks
thresh = 0.25  # eliminate local peaks lower than thresh * highest peak
degree = 5  # merge local peaks within degree together

index_list = list(np.ndindex(n_index))  # linearized indices of voxels on ROI
# numpy arrays to store results
beta_all = np.zeros(n_index + (S+1, p))  # SN coefficients in S smoothing steps
fod_all = np.zeros(n_index + (S+1, l))  # fod evaluations on dense grid in S smoothing steps
lambda_all = -np.ones(n_index + (S+1,))  # indices of selected lambda in lambda grid in S smoothing steps
#beta_all = {}
#fod_all = {}
# intermediate information in smoothing steps
stop_s = -np.ones(n_index)  # total steps of smoothing procedure (-1 if not stopped)
weight_min = np.zeros(n_index + (S+1,))  # minimum-nearest-neighbor distance in S smoothing steps
num_peak = np.zeros(n_index)  # number of peaks by peak detection algorithm
num_indicator = np.zeros(n_index + (S+1,))  # whether at least one adjacent voxels have the same number of peaks

# record computational time for each component
time_DWI_Weight = 0
time_MNN_Dist = 0
time_FOD_Peak = 0
time_num_indicator = 0
time_stop_MNN = 0


start_time = time.time()

# prepare the inverse matrices common for every ADMM step for every lambda/rho
Ip = np.eye(p)
XX = design_SN_T.dot(design_SN_T.T)
CC = Constraint_T.dot(Constraint_T.T) + Ip
inv_seq = np.zeros((L, p, p))
for i in range(L):
	inv_seq[i] = cho_solve(cho_factor(XX+rho_seq[i]*CC, lower=True), Ip)
inv_seq = np.ascontiguousarray(inv_seq)

print("Time used for matrix inversion: {:.2f}s".format(time.time() - start_time))

start_time = time.time()

# start smoothing steps
for s in range(S+1):

	start_time_s = time.time()

	active_index_list = [k for k in index_list if stop_s[k] < 0]  # indices of voxels that still need smoothing
	#active_index_list = [k for k in index_list if stop_s[k] < 0 and mask[k] > 0]
	stop_index_list = [k for k in index_list if stop_s[k] >= 0]  # indices of voxels that smoothing has stopped

	# calculate kernel weighted DWI signals
	start_time_tmp = time.time()
	if s == 0:
		DWI_w = DWI if not noiseless else DWI_noiseless
	else:
		# calculate weight ratio matrix for weight rescaling
		fod_prev = fod_all[..., s-1, :]
		fod_st = np.zeros(fod_prev.shape)
		DWI_w = np.zeros(DWI.shape)
		for k in index_list:
			fod_st[k] = fod_stand(fod_prev[k])
		if scaling:
			weight_min_prev = weight_min[..., s-1]
			weight_ratio_1 = np.minimum(weight_min_prev / np.percentile(weight_min_prev, q*100), np.ones(n_index))
			weight_ratio_2 = np.maximum(weight_min_prev / np.percentile(weight_min_prev, (1-q)*100), np.ones(n_index))
			weight_ratio = weight_ratio_1 * weight_ratio_2
		else:
			weight_ratio = np.ones(n_index)

		# calculate weighted DWI signals
		def DWI_Weight_Wrapper(k):
			return DWI_Weight(k, DWI, ch**s, b, fod_st, dis_func, weight_ratio, dis_scale)

		if parallel:
			pool = Pool(16)
			result_list = pool.map(DWI_Weight_Wrapper, active_index_list)
			pool.close()
			pool.join()
			for k in range(len(active_index_list)):
				DWI_w[active_index_list[k]] = result_list[k]
		else:
			for k in active_index_list:
				DWI_w[k] = DWI_Weight_Wrapper(k)
	time_DWI_Weight += time.time() - start_time_tmp

	# estimate FOD by fitting SN_CLasso on kernel weighted DWI signals
	def SN_CLasso_Wrapper(k):
		#print("Estimating: {}".format(k))
		return SN_CLasso(
			DWI_w[k], design_SN_T, Constraint_T, vertex_SN, inv_seq, lambda_seq, rho_seq, stop_percent, stop_thresh, 
			ep_a, ep_r, maxit, verbose)

	if parallel:
		pool = Pool(4)
		result_list = pool.map(SN_CLasso_Wrapper, active_index_list)
		pool.close()
		pool.join()
		for k in range(len(active_index_list)):
			beta_all[active_index_list[k]][s], fod_all[active_index_list[k]][s], lambda_all[active_index_list[k]][s], _ = result_list[k]
			#beta_all[active_index_list[k]], fod_all[active_index_list[k]], _ = result_list[k]
	else:
		for k in active_index_list:
			beta_all[k][s], fod_all[k][s], lambda_all[k][s], _ = SN_CLasso_Wrapper(k)

	# for voxels reached stopping, simply copy the results from the previous smoothing step
	for k in stop_index_list:
		beta_all[k][s], fod_all[k][s] = beta_all[k][s-1], fod_all[k][s-1]

	# calculate num_indicator (whether at least one adjacent voxels have the same number of peaks) in stopping rule
	if S != 0:
		start_time_tmp = time.time()
		weight_min[..., s] = MNN_Dist(fod_all[..., s, :], dis_func)  # minimum-nearest-neighbor distance at step s
		time_MNN_Dist += time.time() - start_time_tmp
		start_time_tmp = time.time()
		def FOD_Peak_Wrapper(k):
			return FOD_Peak(fod_all[k][s], idx, nbhd, thresh, degree, pos, sampling_index, False)
		if parallel:
			pool = Pool(16)
			result_list = pool.map(FOD_Peak_Wrapper, active_index_list)
			pool.close()
			pool.join()
			for k in range(len(active_index_list)):
				num_peak[active_index_list[k]] = result_list[k]  # number of detected peaks at step s
		else:
			for k in active_index_list:
				num_peak[k] = FOD_Peak_Wrapper(k)
		time_FOD_Peak += time.time() - start_time_tmp
		start_time_tmp = time.time()
		num_indicator[..., s] = Num_Indicator(num_peak, active_index_list)  # num_indicator at step s
		time_num_indicator += time.time() - start_time_tmp

	# check stopping rule based on minimum-nearest-neighbor distance
	start_time_tmp = time.time()
	if stop_MNN and s >= 2:
		for k in active_index_list:
			if min(weight_min[k][s], weight_min[k][s-1]) >= weight_min[k][s-2] and num_indicator[k][s-2] == 1:
				stop_s[k] = s-2
				beta_all[k][s], fod_all[k][s] = beta_all[k][s-2], fod_all[k][s-2]
				beta_all[k][s-1], fod_all[k][s-1] = beta_all[k][s-2], fod_all[k][s-2]
			elif s == S and weight_min[k][s] >= weight_min[k][s-1] and num_indicator[k][s-1] == 1:
				stop_s[k] = s-1
				beta_all[k][s], fod_all[k][s] = beta_all[k][s-1], fod_all[k][s-1]
		weight_min[..., s] = MNN_Dist(fod_all[..., s, :], dis_func)
	time_stop_MNN += time.time() - start_time_tmp

	# check stopping rule in PMARM
	if stop_seq and s >= 1:
		for k in active_index_list:
			if dis_func(fod_stand(fod_all[k][s]), fod_stand(fod_all[k][s-1])) > chi2.ppf(0.6/s, 1) * a_seq:
				stop_s[k] = s-1
				beta_all[k][s], fod_all[k][s] = beta_all[k][s-1], fod_all[k][s-1]

	print("s = {}, time used = {:.2f}s, stopped = {:.2f}%".format(
		s, time.time() - start_time_s, np.mean(stop_s != -1)*100))

# print computational time for each component
print("Total time used: {:.2f}s".format(time.time() - start_time))
print("Time used for DWI_Weight: {:.2f}s".format(time_DWI_Weight))
print("Time used for MNN_Dist: {:.2f}s".format(time_MNN_Dist))
print("Time used for FOD_Peak: {:.2f}s".format(time_FOD_Peak))
print("Time used for num_indicator: {:.2f}s".format(time_num_indicator))
print("Time used for stop_MNN: {:.2f}s".format(time_stop_MNN))

# save results
if not noiseless:
	np.save("beta_all.npy", beta_all)
	np.save("fod_all.npy", fod_all)
	np.save("lambda_all.npy", lambda_all)
else:
	beta_all, fod_all, lambda_all = np.squeeze(beta_all), np.squeeze(fod_all), np.squeeze(lambda_all)
	np.save("beta_all_no.npy", beta_all)
	np.save("fod_all_no.npy", fod_all)
	np.save("lambda_all_no.npy", lambda_all)