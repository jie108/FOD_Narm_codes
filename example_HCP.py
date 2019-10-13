import numpy as np
import nibabel as nib
##python package to read .nii file; similar as NIfTI” package from MatLab, however with flipped image
from real_data import gauss_newton_dwi
from sphere_mesh import spmesh
from sphere_harmonics import spharmonic, Rmatrix
from sphere_needlets import SNvertex, Ctran
import warnings
warnings.filterwarnings("ignore")

# load data
#data_path = 'HCPM/test/'  # HCP test data
data_path = 'HCPM/retest/'  # HCP retest data
bvec_raw = np.loadtxt(data_path+'bvecs')  # gradient directions along which DWI signals are measured
bval_raw = np.loadtxt(data_path+'bvals')  # b-values
img_data_raw = nib.load(data_path+'data.nii.gz').get_data().astype('float64')  # original DWI signals
img_data_raw = img_data_raw[::-1]  # flip img_data_raw to make it consistent with img_data_raw in matlab
mask = nib.load(data_path+'data_brain.nii.gz').get_data().astype('float64')
mask = mask[::-1]  # flip mask to make it consistent with mask in matlab

# standardize b-values according to bval_list
bval_list = np.array([1000, 2000, 3000])
bval_raw[np.abs(bval_raw)<100] = 0
for b in bval_list:
	bval_raw[np.abs(bval_raw-b)<100] = b

# separate DWI signals corresponding to b0 from the original DWI signals
bval_indi = np.array([i for i in range(len(bval_raw)) if bval_raw[i] in bval_list])
img_data_all, img_b0_all = img_data_raw[..., bval_indi], img_data_raw[..., bval_raw == 0]
bvec, bval = bvec_raw[..., bval_indi], bval_raw[bval_indi]

# pre-analysis for estimating b_factor and ratio_response
x_pre, y_pre, z_pre = np.arange(20, 50), np.arange(65, 105), np.arange(50, 90)
img_data_pre, img_b0_pre = img_data_all[np.ix_(x_pre, y_pre, z_pre)], img_b0_all[np.ix_(x_pre, y_pre, z_pre)]

# estimation of sigma and S0
# assume additive error (valid when SNR is large) 
# (assume additive error as well in nonlinear regression in single tensor model)
img_b0_pre_indi = img_b0_pre.min(axis=3)>0
sigma_pre, S0_pre = img_b0_pre.std(axis=3, ddof=1)*img_b0_pre_indi, img_b0_pre.mean(axis=3)*img_b0_pre_indi

# single tensor model
S_data_pre = (img_data_pre.T/S0_pre.T).T
S_data_pre[S_data_pre<=0] = np.min(S_data_pre[S_data_pre>0])
S_data_pre[S_data_pre==float("inf")] = np.max(S_data_pre[S_data_pre<float("inf")])

n_index = S_data_pre.shape[:-1]
eval_pre = np.ones(n_index+(3,))  # eigenvalues of D matrix for each voxel
index_list = list(np.ndindex(n_index))
for k in index_list:
	D_matrix = gauss_newton_dwi(bvec, bval, S_data_pre[k])
	if not np.isnan(D_matrix).any():
		eval_pre[k] = np.sort(np.linalg.eig(D_matrix)[0])

FA_pre = np.sqrt(((eval_pre[..., 0]-eval_pre[..., 1])**2+(eval_pre[..., 1]-eval_pre[..., 2])**2
	+(eval_pre[..., 2]-eval_pre[..., 0])**2)/(2*(eval_pre[..., 0]**2+eval_pre[..., 1]**2+eval_pre[..., 2]**2)))
eval_ratio_pre = eval_pre[..., 2]*2/(eval_pre[..., 0]+eval_pre[..., 1])
eval_pre_indi = (FA_pre<1) * (FA_pre>0.8) * ((eval_pre>0).all(axis=3)) * (eval_pre[..., 1]/eval_pre[..., 0]<1.5)

eval_select_pre = list(zip(eval_pre[eval_pre_indi], eval_ratio_pre[eval_pre_indi]))
eval_select_pre.sort(key = lambda x: x[1])
eval_select_median_pre = eval_select_pre[len(eval_select_pre)//2]
# b_factor is a multiplicative factor used for adjusting the scale of -b*u^T*D*u in function "myresponse",
# leading to an appropriate R matrix
# in function "myresponse", D is assumed to have eigenvalues 1/ratio, 1/ratio and 1, however in practice, 
# the largest eigenvalue of D might be, say lambda. we then set b_factor=lambda, and change b value by b=b*b_factor,
# thus -b*u^T*D*u in function "myresponse" will have the correct value
b_factor, ratio_response = eval_select_median_pre[0][2], eval_select_median_pre[1]
# end pre-analysis for estimating b_factor and ratio_response

# real data range
#bval_list = np.array([1000, 2000, 3000])  # include b value 1K, 2K and 3K 
#bval_list = np.array([1000])  # include b value 1K
bval_list = np.array([3000])  # include b value 3K
bval_indi = np.array([i for i in range(len(bval_raw)) if bval_raw[i] in bval_list])
img_data_all, img_b0_all = img_data_raw[..., bval_indi], img_data_raw[..., bval_raw == 0]
bvec, bval = bvec_raw[..., bval_indi], bval_raw[bval_indi]

#x_range, y_range, z_range = np.arange(55, 75), np.arange(85, 105), np.arange(65, 85)  # ROI1
#x_range, y_range, z_range = np.arange(55, 75), np.arange(85, 106), np.arange(65, 85)  # ROI1 (retest)
#x_range, y_range, z_range = np.arange(53, 68), np.arange(117, 132), np.arange(60, 75)  # ROI2
#x_range, y_range, z_range = np.arange(53, 68), np.arange(117, 133), np.arange(59, 75)  # ROI2 (retest)
x_range, y_range, z_range = np.arange(53, 68), np.arange(115, 130), np.arange(61, 76)  # ROI2 (retest) (HCPM)
#x_range, y_range, z_range = np.arange(59, 74), np.arange(90, 105), np.arange(70, 85)  # ROI3 (subregion of ROI1)
img_data, img_b0 = img_data_all[np.ix_(x_range, y_range, z_range)], img_b0_all[np.ix_(x_range, y_range, z_range)]
img_mask = mask[np.ix_(x_range, y_range, z_range)]

# estimation of sigma and S0
sigma, S0 = img_b0.std(axis=3, ddof=1), img_b0.mean(axis=3)
img_b0_indi = np.invert((img_b0.min(axis=3)>0)*(img_mask>0))
sigma[img_b0_indi], S0[img_b0_indi] = 1, 1

img_data[img_data<0] = np.min(img_data[img_data>0])
DWI = (img_data.T/S0.T).T  # normalize DWI of each voxel by its estimated S0 and then set S0=1 in response function

lmax = 8  # maximum spherical harmonic level
jmax = 3  # maximum spherical needlet level (corresponding to lmax)
J_plot = 5  # vertex level for graphing and representation

# spherical coordinates of bvec
theta_bvec = np.arccos(bvec[2])
phi_bvec = np.arctan2(bvec[1], bvec[0])
phi_bvec += 2*np.pi*(phi_bvec<0)
# coordinates of half equal-angle grid at vertex level J_plot
pos_constraint, theta_constraint, phi_constraint, sampling_index = spmesh(J_plot, half = 1)
# coordinates of entire equal-angle grid at vertex level J_plot
pos_plot, theta_plot, phi_plot = spmesh(J_plot, half = 0)

# Phi matrix: evaluate SH basis on grid determined by bvec (convert SH coefficients onto grid determined by bvec)
SH_matrix = spharmonic(theta_bvec, phi_bvec, lmax)
# R matrix
R_matrix = {}
for b in bval_list:
	R_matrix[b] = Rmatrix(b*b_factor, ratio_response, lmax)
# design matrix for symmetrized spherical harmonics up to level lmax
design_SH = SH_matrix.copy()
for b in bval_list:
	design_SH[bval == b] = design_SH[bval == b].dot(R_matrix[b])

# constraint matrix
Constraint = SNvertex(theta_constraint, phi_constraint, jmax)
# convert SN coefficients onto dense grid determined by J_plot
vertex_SN = SNvertex(theta_plot, phi_plot, jmax)
# beta = Cst_matrix'*f, beta: SN coefficients, f: SH coefficients
Cst_matrix = Ctran(lmax)
# f = C_matrix*beta
C_matrix = np.linalg.solve(Cst_matrix.dot(Cst_matrix.T), Cst_matrix)
# design matrix for symmetrized spherical needlets up to level jmax
design_SN = design_SH.dot(C_matrix)

np.savez("ROI_HCP.npz", Constraint = Constraint, design_SN = design_SN, vertex_SN = vertex_SN, DWI = DWI, mask = img_mask)