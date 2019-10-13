## Feb 8, 2018
## comments by Jie 
##import packages 
import numpy as np
import scipy.io as spio
from scipy.linalg import cho_solve, cho_factor
from scipy.stats import chi2
from DWI_weight import *
from FOD_peak import *
from pyclasso import SN_CLasso  ## defined in pyclasso.pyx, a wrapper function to the sn_classo in classo.c file
from multiprocessing import Pool
import time

## read in data 
path="FOD_files"
data_type = "ROI" ## "ROI3D", "ROIs"
noiseless = False
real_data = False
ROI_mat = spio.loadmat("{}/{}.mat".format(path,data_type), squeeze_me=True)
type(ROI_mat)

## components in ROI.mat 
Constraint = ROI_mat["Constraint"]  ## C matrix  (on half sphere): 1281 by 511: half of vertex_SN
Constraint.shape 
design_SN = ROI_mat["design_SN"]  ## design matrix: 41 by 511, SN basis evaluated on the gradient directions 
design_SN.shape
vertex_SN = ROI_mat["vertex_SN"]  ## 511 SN basis evaluated on the 2562 equal angle grid (on full sphere) 
vertex_SN.shape
DWI = ROI_mat["DWI"]  ## data 
DWI.shape ## 10 by 10 grid, each has 41 DWI observations 

##transform for efficiency in C , and make ascontiguousarray for C input 
Constraint_T = np.ascontiguousarray(Constraint.T)
design_SN_T = np.ascontiguousarray(design_SN.T)
vertex_SN = np.ascontiguousarray(vertex_SN)
DWI = np.ascontiguousarray(DWI)

## for noiseless or real data situations 
if noiseless:
	DWI_noiseless = ROI_mat["DWI_noiseless"]
	DWI_noiseless = np.ascontiguousarray(DWI_noiseless)

if real_data:
    mask = ROI_mat["mask"]

## 
n_index = DWI.shape[:-1] ##tuple to record  ROI dimensions: 10 by 10 
l, p = vertex_SN.shape ## number of grid points and number of SN basis 

## define tuning parameter grid and stopping rule 
if not real_data:
	L = 100
	log_seq = np.linspace(0, -5, L)
	lambda_seq = 10**log_seq
	rho_seq = 2*lambda_seq
	stop_percent = 0.05
	stop_thresh = 1e-3
else:
	L = 120
	log_seq = np.linspace(0.5, -5, L)
	lambda_seq = 10**log_seq
	rho_seq = lambda_seq*np.maximum(np.ceil(-log_seq), np.ones(L))
	stop_percent = 0.05
	stop_thresh = 1e-3

## ADMM parameters 
ep_a = 1e-4
ep_r = 1e-2
maxit = 5000
verbose = 0

## smoothing neighborhood expansion parameter 
ch = 1.15
if not real_data:
	dis_scale = np.ones(len(n_index))
else:
	dis_scale = np.array([1, 1, 2])

## maximum smoothing steps	
## if no smoothing, set S = 0 
#S = 10 if not noiseless else 0  
S=0 ## no smoothing
S=2

## smoothing weights and stopping rule 
b = 2  ## spatial kernel b parameter  
dis_func = hellinger_dis ##  dissimilarity kernel, defined in DWI_weight.py 
scaling = True ## 
q = 0.15 ## top q percent 
stop_MMN = True  ## our stopping proposal 
stop_seq = False ## Zhu stopping proposal based on chi
a_seq = 3 ## Zhu stopping criterion parameter  

## set parallel 
#parallel = True
parallel=False

## for peak detection in the smoothing steps: 
pos_mat = spio.loadmat("{}/pos.mat".format(path), squeeze_me=True)
pos = pos_mat["pos_plot"]   ## 3 by 2562: evaluation grid 3D coordinates 
pos.shape 

sampling_index = pos_mat["sampling_index"]  ## corresponds to half sphere grid 
sampling_index.shape

dis = pos.T.dot(pos) ## pos^T pos: inner product distance matrix 
idx = np.zeros(dis.shape, dtype=int)
for i in range(l):
	idx[i, :] = np.argsort(-dis[i, :])  ## find the index of the nearest grid point

##  peak detection algrithm parameters 
nbhd = 40
thresh = 0.25
degree = 5
indicator_degree = 10

## 
index_list = list(np.ndindex(n_index)) ## linearized indice of the voxels on the ROI

## define numpy arrays to store results 
beta_all = np.zeros(n_index + (S+1, p))  ## SN coefficients for S smoothing steps 
fod_all = np.zeros(n_index + (S+1, l))  ## fod evaluation on the grid for S smoothing steps 

## for real data use {} as the ROI is regular, so use dictionary  
#beta_all = {} 
#fod_all = {}  

## intermediate information in smoothing steps 
stop_s = -np.ones(n_index)   ## for each voxel, whether stopped? 
weight_min = np.zeros(n_index + (S+1,)) ## minimum nearest neighbor weight 
num_peak = np.zeros(n_index) ## number of peaks by peak detection algorithm 
num_indicator = np.zeros(n_index + (S+1,))  ## whether adjacent voxels have the same number of peaks; if not , keep smoothing (even other stopping criteria are met)

## computational time for each step 
time_DWI_Weight = 0
time_MNN_Dist = 0
time_FOD_Peak = 0
time_num_indicator = 0
time_stop_MMN = 0


## prepare the "inverse" matrix common for every ADMM step for every lambda/rho 
start_time = time.time()

Ip = np.eye(p)
XX = design_SN_T.dot(design_SN_T.T)
CC = Constraint_T.dot(Constraint_T.T) + Ip
inv_seq = np.zeros((L, p, p)) ## use to store the 
for i in range(L):
	inv_seq[i] = cho_solve(cho_factor(XX+rho_seq[i]*CC, lower=True), Ip)

inv_seq = np.ascontiguousarray(inv_seq)  ## prepare for C input 
print("Time used for matrix inversion: {:.2f}s".format(time.time() - start_time))


## start fitting: smoothing steps 
start_time = time.time()

for s in range(S+1):
	start_time_s = time.time()
	active_index_list = [k for k in index_list if stop_s[k] < 0]  ## indices of voxels that still need smoothing 
	#active_index_list = [k for k in index_list if stop_s[k] < 0 and mask[k] > 0] ## for real data
	stop_index_list = [k for k in index_list if stop_s[k] >= 0] ## indices of voxels that smoothing has stopped 

## calculate weights 
	start_time_tmp = time.time()
	if s == 0:   
        ## first step: 
		DWI_w = DWI if not noiseless else DWI_noiseless
	else:      
		## subsequent smoothing steps 
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
##			
		def DWI_Weight_Wrapper(k):
			return DWI_Weight(k, DWI, ch**s, b, fod_st, dis_func, weight_ratio, dis_scale)
##
		if parallel:
			#pool = Pool(16) ##server call 16 thread
			pool=Pool(2) ##PC
			result_list = pool.map(DWI_Weight_Wrapper, active_index_list)
			pool.close()
			pool.join()
			for k in range(len(active_index_list)):
				DWI_w[active_index_list[k]] = result_list[k]
		else:
			for k in active_index_list:
				DWI_w[k] = DWI_Weight_Wrapper(k)
	time_DWI_Weight += time.time() - start_time_tmp
##
## fitting FOD by classo
	def SN_CLasso_Wrapper(k):
		print("Estimating: {}".format(k))
		return SN_CLasso(
			DWI_w[k], design_SN_T, Constraint_T, vertex_SN, inv_seq, lambda_seq, rho_seq, stop_percent, stop_thresh, 
			ep_a, ep_r, maxit, verbose)
##
	if parallel:
		#pool = Pool(4) ## call 4 threads, each with 8 cores 
		pool=Pool(2) ## PC  
		result_list = pool.map(SN_CLasso_Wrapper, active_index_list)
		pool.close()
		pool.join()
		for k in range(len(active_index_list)):
			beta_all[active_index_list[k]][s], fod_all[active_index_list[k]][s], _ = result_list[k] ## for smoothing;
			#beta_all[active_index_list[k]], fod_all[active_index_list[k]], _ = result_list[k] ## no smoothing 
	else:
		for k in active_index_list:
			beta_all[k][s], fod_all[k][s], record = SN_CLasso_Wrapper(k)
##
##  for voxel reached stopping, simply copy the results from the previous step 
	for k in stop_index_list:
		beta_all[k][s], fod_all[k][s] = beta_all[k][s-1], fod_all[k][s-1]
##
## peak detection: not run for voxelwise estimation, i.e., when S==0
	if S != 0:
		start_time_tmp = time.time()
		weight_min[..., s] = MNN_Dist(fod_all[..., s, :], dis_func)
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
				num_peak[active_index_list[k]] = result_list[k]
		else:
			for k in active_index_list:
				num_peak[k] = FOD_Peak_Wrapper(k)
		time_FOD_Peak += time.time() - start_time_tmp
		start_time_tmp = time.time()
		num_indicator[..., s] = Num_Indicator(num_peak, indicator_degree, active_index_list)
		time_num_indicator += time.time() - start_time_tmp
##
## check stopping criteria: our proposal based on MMN
	start_time_tmp = time.time()
	if stop_MMN and s >= 2:
		for k in active_index_list:
			if min(weight_min[k][s], weight_min[k][s-1]) >= weight_min[k][s-2] and num_indicator[k][s-2] == 1:
				stop_s[k] = s-2
				beta_all[k][s], fod_all[k][s] = beta_all[k][s-2], fod_all[k][s-2]
				beta_all[k][s-1], fod_all[k][s-1] = beta_all[k][s-2], fod_all[k][s-2]
			elif s == S and weight_min[k][s] >= weight_min[k][s-1] and num_indicator[k][s-1] == 1:
				stop_s[k] = s-1
				beta_all[k][s], fod_all[k][s] = beta_all[k][s-1], fod_all[k][s-1]
		weight_min[..., s] = MNN_Dist(fod_all[..., s, :], dis_func)
	time_stop_MMN += time.time() - start_time_tmp
##
## check stopping criteria: Zhu proposal 
	if stop_seq and s >= 1:
		for k in active_index_list:
			if dis_func(fod_stand(fod_all[k][s]), fod_stand(fod_all[k][s-1])) > chi2.ppf(0.6/s, 1) * a_seq:
				stop_s[k] = s-1
				beta_all[k][s], fod_all[k][s] = beta_all[k][s-1], fod_all[k][s-1]
##
	print("s = {}, time used = {:.2f}s, stopped = {:.2f}%".format(
		s, time.time() - start_time_s, np.mean(stop_s != -1)*100))

## print time being used 
print("Total time used: {:.2f}s".format(time.time() - start_time))
print("Time used for DWI_Weight: {:.2f}s".format(time_DWI_Weight))
print("Time used for MNN_Dist: {:.2f}s".format(time_MNN_Dist))
print("Time used for FOD_Peak: {:.2f}s".format(time_FOD_Peak))
print("Time used for num_indicator: {:.2f}s".format(time_num_indicator))
print("Time used for stop_MMN: {:.2f}s".format(time_stop_MMN))

## save results 
if not noiseless:
	np.save("beta_all.npy", beta_all)
	np.save("fod_all.npy", fod_all)
else:
	#beta_all, fod_all = np.squeeze(beta_all), np.squeeze(fod_all)
	np.save("beta_all_no.npy", beta_all)
	np.save("fod_all_no.npy", fod_all)

## execute this code in terminal: > python ./classo_DWI.py 