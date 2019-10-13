import numpy as np
from sphere_mesh import spmesh
from sphere_harmonics import spharmonic, Rmatrix
from sphere_needlets import SNvertex, Ctran
from simu_region import region2D, region3D, fiber_crossing

lmax = 8  # maximum spherical harmonic level
jmax = 3  # maximum spherical needlet level (corresponding to lmax)
J_obs = 2.5  # vertex level for deciding number of observations (number of gradient directions)
J_plot = 5  # vertex level for graphing and representation
b_response = 3  # first b value in single tensor model for response function (b_response = 0 if no first b value)
b_response2 = 3+1e-15  # second b value in single tensor model for response function (b_response2 = 0 if no second b value)
ratio_response = 10  # shape in single tensor model for response function

# coordinates of half equal-angle grid at vertex level J_obs
pos_obs, theta_obs, phi_obs, sampling_index_obs = spmesh(J_obs, half = 1)
# coordinates of half equal-angle grid at vertex level J_obs+0.5 and complement of coordinates at vertex level J_obs
pos_obs_all, theta_obs_all, phi_obs_all, _ = spmesh(int(J_obs+0.5), half = 1)
sampling_index_obs2 = [i for i in range(len(theta_obs_all)) if i not in sampling_index_obs]
theta_obs2, phi_obs2 = theta_obs_all[sampling_index_obs2], phi_obs_all[sampling_index_obs2]
# coordinates of half equal-angle grid at vertex level J_plot
pos_constraint, theta_constraint, phi_constraint, sampling_index = spmesh(J_plot, half = 1)
# coordinates of entire equal-angle grid at vertex level J_plot
pos_plot, theta_plot, phi_plot = spmesh(J_plot, half = 0)

# convert SH coefficients onto dense grid determined by J_plot
SH_matrix_plot = spharmonic(theta_plot, phi_plot, lmax)
# design matrix for symmetrized spherical harmonics
design_SH = {}
# design matrix for symmetrized spherical harmonics for first b value
if b_response:
	# Phi matrix: evaluate SH basis on grid determined by J_obs (convert SH coefficients onto grid determined by J_obs)
	SH_matrix = spharmonic(theta_obs, phi_obs, lmax)
	# R matrix
	R_matrix = Rmatrix(b_response, ratio_response, lmax)
	# design matrix for symmetrized spherical harmonics up to level lmax
	design_SH[b_response] = SH_matrix.dot(R_matrix)
# design matrix for symmetrized spherical harmonics for second b value
if b_response2:
	# Phi matrix: evaluate SH basis on grid determined by J_obs (convert SH coefficients onto grid determined by J_obs)
	SH_matrix = spharmonic(theta_obs2, phi_obs2, lmax)
	# R matrix
	R_matrix = Rmatrix(b_response2, ratio_response, lmax)
	# design matrix for symmetrized spherical harmonics up to level lmax
	design_SH[b_response2] = SH_matrix.dot(R_matrix)
if len(design_SH.keys()) == 1:
	design_SH = design_SH[b_response] if b_response else design_SH[b_response2]
elif len(design_SH.keys()) == 2:
	# stack design matrices for first b value and second b value
	design_SH = np.vstack((design_SH[b_response], design_SH[b_response2]))

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

n = 10  # size of simulation region
sigma = 0.05  # noise level (SNR=1/sigma)
matlab_randn = True  # set matlab_randn=True if we want DWI signal (with noise) from python the same as that from matlab

# DWI signals (noiseless and with Rician noise) and SH coefficients of dirac functions
# in 2D simulation region with 1-fiber and 2-fiber
r = 0.6  # radius of smaller circle which determines the region for fiber generation schema
DWI_noiseless, DWI = {}, {}
if b_response:
	DWI_noiseless[b_response], DWI[b_response], SH_coef, theta_fib, phi_fib, fib_indi = region2D(
		n, theta_obs, phi_obs, b_response, ratio_response, lmax, r, sigma, matlab_randn)
if b_response2:
	DWI_noiseless[b_response2], DWI[b_response2], SH_coef, theta_fib, phi_fib, fib_indi = region2D(
		n, theta_obs2, phi_obs2, b_response2, ratio_response, lmax, r, sigma)
if len(DWI.keys()) == 1:
	DWI_noiseless = DWI_noiseless[b_response] if b_response else DWI_noiseless[b_response2]
	DWI = DWI[b_response] if b_response else DWI[b_response2]
elif len(DWI.keys()) == 2:
	DWI_noiseless = np.concatenate((DWI_noiseless[b_response], DWI_noiseless[b_response2]), axis=-1)
	DWI = np.concatenate((DWI[b_response], DWI[b_response2]), axis=-1)
dirac_plot = SH_coef.dot(SH_matrix_plot.T)
np.savez("ROI_{}{}K.npz".format(b_response, b_response2), Constraint = Constraint, design_SN = design_SN, 
	vertex_SN = vertex_SN, DWI = DWI, DWI_noiseless = DWI_noiseless, theta_fib = theta_fib, phi_fib = phi_fib, 
	fib_indi = fib_indi, dirac_plot = dirac_plot)

# DWI signals (noiseless and with Rician noise) and SH coefficients of dirac functions
# in 2D simulation region with 0-fiber, 1-fiber and 2-fiber
r = 0.8
DWI_noiseless, DWI = {}, {}
if b_response:
	DWI_noiseless[b_response], DWI[b_response], SH_coef, theta_fib, phi_fib, fib_indi = region2D(
		n, theta_obs, phi_obs, b_response, ratio_response, lmax, r, sigma, matlab_randn)
if b_response2:
	DWI_noiseless[b_response2], DWI[b_response2], SH_coef, theta_fib, phi_fib, fib_indi = region2D(
		n, theta_obs2, phi_obs2, b_response2, ratio_response, lmax, r, sigma)
if len(DWI.keys()) == 1:
	DWI_noiseless = DWI_noiseless[b_response] if b_response else DWI_noiseless[b_response2]
	DWI = DWI[b_response] if b_response else DWI[b_response2]
elif len(DWI.keys()) == 2:
	DWI_noiseless = np.concatenate((DWI_noiseless[b_response], DWI_noiseless[b_response2]), axis=-1)
	DWI = np.concatenate((DWI[b_response], DWI[b_response2]), axis=-1)
dirac_plot = SH_coef.dot(SH_matrix_plot.T)
np.savez("ROIs_{}{}K.npz".format(b_response, b_response2), Constraint = Constraint, design_SN = design_SN, 
	vertex_SN = vertex_SN, DWI = DWI, DWI_noiseless = DWI_noiseless, theta_fib = theta_fib, phi_fib = phi_fib, 
	fib_indi = fib_indi, dirac_plot = dirac_plot)

# DWI signals (noiseless and with Rician noise) and SH coefficients of dirac functions
# in 3D simulation region with 0-fiber, 1-fiber and 2-fiber
r = 0.6
DWI_noiseless, DWI = {}, {}
if b_response:
	DWI_noiseless[b_response], DWI[b_response], SH_coef, theta_fib, phi_fib, fib_indi = region3D(
		n, theta_obs, phi_obs, b_response, ratio_response, lmax, r, sigma, matlab_randn)
if b_response2:
	DWI_noiseless[b_response2], DWI[b_response2], SH_coef, theta_fib, phi_fib, fib_indi = region3D(
		n, theta_obs2, phi_obs2, b_response2, ratio_response, lmax, r, sigma)
if len(DWI.keys()) == 1:
	DWI_noiseless = DWI_noiseless[b_response] if b_response else DWI_noiseless[b_response2]
	DWI = DWI[b_response] if b_response else DWI[b_response2]
elif len(DWI.keys()) == 2:
	DWI_noiseless = np.concatenate((DWI_noiseless[b_response], DWI_noiseless[b_response2]), axis=-1)
	DWI = np.concatenate((DWI[b_response], DWI[b_response2]), axis=-1)
dirac_plot = SH_coef.dot(SH_matrix_plot.T)
np.savez("ROI3D_{}{}K.npz".format(b_response, b_response2), Constraint = Constraint, design_SN = design_SN, 
	vertex_SN = vertex_SN, DWI = DWI, DWI_noiseless = DWI_noiseless, theta_fib = theta_fib, phi_fib = phi_fib, 
	fib_indi = fib_indi, dirac_plot = dirac_plot)

# identical two-fiber cases including DWI signals (noiseless and with Rician noise) 
# and SH coefficients of dirac functions in 2D simulation region
angle = np.pi/4  # separation angle between two fibers
DWI_noiseless, DWI = {}, {}
if b_response:
	DWI_noiseless[b_response], DWI[b_response], SH_coef, theta_fib, phi_fib, fib_indi = fiber_crossing(
		n, theta_obs, phi_obs, b_response, ratio_response, lmax, angle, sigma)
if b_response2:
	DWI_noiseless[b_response2], DWI[b_response2], SH_coef, theta_fib, phi_fib, fib_indi = fiber_crossing(
		n, theta_obs2, phi_obs2, b_response2, ratio_response, lmax, angle, sigma, seed = 1)
if len(DWI.keys()) == 1:
	DWI_noiseless = DWI_noiseless[b_response] if b_response else DWI_noiseless[b_response2]
	DWI = DWI[b_response] if b_response else DWI[b_response2]
elif len(DWI.keys()) == 2:
	DWI_noiseless = np.concatenate((DWI_noiseless[b_response], DWI_noiseless[b_response2]), axis=-1)
	DWI = np.concatenate((DWI[b_response], DWI[b_response2]), axis=-1)
dirac_plot = SH_coef.dot(SH_matrix_plot.T)
np.savez("fiber_crossing_{}{}K.npz".format(b_response, b_response2), Constraint = Constraint, design_SN = design_SN, 
	vertex_SN = vertex_SN, DWI = DWI, DWI_noiseless = DWI_noiseless, theta_fib = theta_fib, phi_fib = phi_fib, 
	fib_indi = fib_indi, dirac_plot = dirac_plot)

np.savez("pos.npz", pos_plot = pos_plot, sampling_index = sampling_index)