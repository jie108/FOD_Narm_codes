import numpy as np
import pandas as pd
from sphere_harmonics import spharmonic_eval, spharmonic, myresponse
from DWI_weight import fod_stand, hellinger_dis
from FOD_peak import FOD_Peak

# add Rician noise on noiseless DWI
# if matlab_randn=True, use pre-stored normal-distributed random values generated in matlab
# if matlab_randn=False, generate normal-distributed random values in python
def Rician_noise(dwi_noiseless, sigma, matlab_randn = False, seed = 0):

	if matlab_randn:
		error = sigma*np.load("matlab_randn.npy")
		if dwi_noiseless.ndim == 3:
			error = error[:, 0]
	else:
		np.random.seed(seed)
		error = sigma*np.random.randn(*((2,)+dwi_noiseless.shape))

	dwi = dwi_noiseless.copy()
	dwi = np.sqrt((dwi + error[0])**2 + error[1]**2)

	return dwi

# generate DWI signals (noiseless and with Rician noise) and SH coefficients of dirac functions 
# on equal-angle grid specified by (theta, phi) in 2D simulation region
# n is the size of simulation region and r is the radius of smaller circle
# DWI_noiseless: region size * region size * number of grid points, DWI signals without noise
# DWI: region size * region size * number of grid points, DWI signals with Rician noise
# SH_coef: region size * region size * number of symmetrized SH basis, SH coefficients of dirac functions
# theta_fib: region size * region size * 2, polar angle of fibers generated from schema 1 and schema 2
# phi_fib: region size * region size * 2, azimuthal angle of fibers generated from schema 1 and schema 2
# fib_indi: fib_indi[i,j]=0: voxel_ij contains no fibers
#			fib_indi[i,j]=1: voxel_ij contains fiber generated from schema 1
#			fib_indi[i,j]=2: voxel_ij contains fiber generated from schema 2
#			fib_indi[i,j]=3: voxel_ij contains fibers generated from schema 1 and schema 2 (crossing fibers)
def region2D(n, theta, phi, b, ratio, lmax, r = 0.6, sigma = 0.05, matlab_randn = False, seed = 0):

	r_mesh = np.linspace(1./(2*n), 1-1./(2*n), n)
	r_y, r_x = np.meshgrid(r_mesh, r_mesh)

	# fiber generation schema 1
	# set bottom right vertex as center, and draw 1/4 circle with radius 1 and r
	# points being selected should satisfy:
	# lower edge of its square should be lower than intersection between radius-1 circle and right edge of its square
	# i.e., y-0.05<sqrt(1^2-(1-(x+0.05))^2)
	# upper edge of its square should be higher than intersection between radius-r circle and left edge of its square
	# i.e., y+0.05>sqrt(r^2-(1-(x-0.05))^2)
	fib_indi_1 = ((1-(r_x+0.05))**2+(r_y-0.05)**2 < 1-1e-6) & ((1-(r_x-0.05))**2+(r_y+0.05)**2 > r**2+1e-6)
	# fiber generation schema 2
	# set bottom left vertex as center, and draw 1/4 circle with radius 1 and r
	# points being selected should satisfy:
	# lower edge of its square should be lower than intersection between radius-1 circle and left edge of its square
	# i.e., y-0.05<sqrt(1^2-(x-0.05)^2)
	# upper edge of its square should be higher than intersection between radius-r circle and right edge of its square
	# i.e., y+0.05>sqrt(r^2-(x+0.05)^2)
	fib_indi_2 = ((r_x-0.05)**2+(r_y-0.05)**2 < 1-1e-6) & ((r_x+0.05)**2+(r_y+0.05)**2 > r**2+1e-6)

	# fib_indi[i,j]=0: voxel_ij contains no fibers
	# fib_indi[i,j]=1: voxel_ij contains fiber generated from schema 1
	# fib_indi[i,j]=2: voxel_ij contains fiber generated from schema 2
	# fib_indi[i,j]=3: voxel_ij contains fibers generated from schema 1 and schema 2 (crossing fibers)
	# simulation region plot can be obtained by rotating fib_indi 90 degree conter-clockwisely
	fib_indi = fib_indi_1 + 2*fib_indi_2

	# spherical coordinates of fiber directions in each voxel
	# coordinate system here is within each voxel, with x-axis perpendicular to the screen, 
	# and is different from coordinate system for generating fibers voxelwisely in simulation region
	theta_fib_1 = np.arctan2(r_x, 1-r_y)  # polar angle of fiber 1
	theta_fib_2 = np.arctan2(r_x, r_y)+np.pi/2 # polar angle of fiber 2
	theta_fib = np.stack((theta_fib_1, theta_fib_2), axis=-1)
	phi_fib = np.ones((n, n, 2))*np.pi/2  # azimuthal angle of fiber 1 and fiber 2

	# calculate noiseless DWI, DWI (with Rician noise) and SH coefficients of dirac functions
	DWI_noiseless = np.zeros((n, n, len(theta)))
	SH_coef = np.zeros((n, n, int((lmax+1)*(lmax+2)/2)))
	for i in range(n):
		for j in range(n):
			if fib_indi[i, j] == 0:
				DWI_noiseless[i, j] = np.exp(-b)  # D becomes identity matrix in single tensor model
				SH_coef[i, j, 0] = spharmonic_eval(0, 0, 0, 0).real
			elif fib_indi[i, j] == 1:
				for l in range(len(theta)):
					DWI_1 = myresponse(b, ratio, theta_fib[i, j, 0], phi_fib[i, j, 0], theta[l], phi[l])
					DWI_noiseless[i, j, l] = DWI_1
				SH_coef[i, j] = spharmonic(theta_fib[i, j, :1], phi_fib[i, j, :1], lmax)
			elif fib_indi[i, j] == 2:
				for l in range(len(theta)):
					DWI_2 = myresponse(b, ratio, theta_fib[i, j, 1], phi_fib[i, j, 1], theta[l], phi[l])
					DWI_noiseless[i, j, l] = DWI_2
				SH_coef[i, j] = spharmonic(theta_fib[i, j, 1:], phi_fib[i, j, 1:], lmax)
			else:  # crossing-fiber case
				for l in range(len(theta)):
					DWI_1 = myresponse(b, ratio, theta_fib[i, j, 0], phi_fib[i, j, 0], theta[l], phi[l])
					DWI_2 = myresponse(b, ratio, theta_fib[i, j, 1], phi_fib[i, j, 1], theta[l], phi[l])
					DWI_noiseless[i, j, l] = (DWI_1+DWI_2)/2  # volumn fractions for two fibers are both 0.5
				SH_coef[i, j] = (spharmonic(theta_fib[i, j, :1], phi_fib[i, j, :1], lmax) 
					+ spharmonic(theta_fib[i, j, 1:], phi_fib[i, j, 1:], lmax))/2
	DWI = Rician_noise(DWI_noiseless, sigma, matlab_randn, seed)

	return DWI_noiseless, DWI, SH_coef, theta_fib, phi_fib, fib_indi

# generate DWI signals (noiseless and with Rician noise) and SH coefficients of dirac functions 
# on equal-angle grid specified by (theta, phi) in 3D simulation region
# n is the size of simulation region and r is the radius of smaller circle
# DWI_noiseless: region size * region size * region size * number of grid points, DWI signals without noise
# DWI: region size * region size * region size * number of grid points, DWI signals with Rician noise
# SH_coef: region size * region size * region size * number of symmetrized SH basis, SH coefficients of dirac functions
# theta_fib: region size * region size * 2, polar angle of fibers generated from schema 1 and schema 2
# phi_fib: region size * region size * 2, azimuthal angle of fibers generated from schema 1 and schema 2
# fib_indi: fib_indi[i,j]=0: voxel_ij contains no fibers
#			fib_indi[i,j]=1: voxel_ij contains fiber generated from schema 1
#			fib_indi[i,j]=2: voxel_ij contains fiber generated from schema 2
#			fib_indi[i,j]=3: voxel_ij contains fibers generated from schema 1 and schema 2 (crossing fibers)
def region3D(n, theta, phi, b, ratio, lmax, r = 0.6, sigma = 0.05, matlab_randn = False, seed = 0):

	r_mesh = np.linspace(1./(2*n), 1-1./(2*n), n)
	r_y, r_x, r_z = np.meshgrid(r_mesh, r_mesh, r_mesh)

	# fiber generation schema 1
	# find plain perpendicular to xy-plain and has pi/4 angle towards x-axis conter-clockwisely, 
	# and point (x,y,z) is on it (this plain has width sqrt(2)*(1-|x-y|) and height 1)
	# 2D coordinates of point (x,y,z) on this plain is (sqrt(2)*min(x,y), z)
	# extend this plain to have width 1 and height 1, then 2D coordinates become (min(x,y)/(1-|x-y|),z) = (tx,ty)
	# in this extended plain, set bottom right vertex as center, and draw 1/4 circle with radius 1 and r
	# points being selected in this extended plain should satisfy:
	# lower edge of its square should be lower than intersection between radius-1 circle and right edge of its square
	# i.e., ty-0.05<sqrt(1^2-(1-(tx+0.05))^2)
	# upper edge of its square should be higher than intersection between radius-r circle and left edge of its square
	# i.e., ty+0.05>sqrt(r^2-(1-(tx-0.05))^2)
	r_xy_1 = np.minimum(r_x, r_y)/(1-np.abs(r_x-r_y))
	fib_indi_1 = ((1-(r_xy_1+0.05))**2+(r_z-0.05)**2 < 1) & ((1-(r_xy_1-0.05))**2+(r_z+0.05)**2 > r**2)
	# fiber generation schema 2
	# find plain perpendicular to xy-plain and has pi*3/4 angle towards x-axis conter-clockwisely, 
	# and point (x,y,z) is on it (this plain has width sqrt(2)*min(2-x-y,x+y) and height 1)
	# 2D coordinates of point (x,y,z) on this plain is (sqrt(2)*min(x,1-y), z)
	# extend this plain to have width 1 and height 1, then 2D coordinates become (min(x,1-y)/min(2-x-y,x+y),z) = (tx,ty)
	# in this extended plain, set bottom left vertex as center, and draw 1/4 circle with radius 1 and r
	# points being selected in this extended plain should satisfy:
	# lower edge of its square should be lower than intersection between radius-1 circle and left edge of its square
	# i.e., ty-0.05<sqrt(1^2-(tx-0.05)^2)
	# upper edge of its square should be higher than intersection between radius-r circle and right edge of its square
	# i.e., ty+0.05>sqrt(r^2-(tx+0.05)^2)
	r_xy_2 = np.minimum(r_x, 1-r_y)/np.minimum(2-r_x-r_y, r_x+r_y)
	fib_indi_2 = ((r_xy_2-0.05)**2+(r_z-0.05)**2 < 1) & ((r_xy_2+0.05)**2+(r_z+0.05)**2 > r**2)

	# fib_indi[i,j]=0: voxel_ij contains no fibers
	# fib_indi[i,j]=1: voxel_ij contains fiber generated from schema 1
	# fib_indi[i,j]=2: voxel_ij contains fiber generated from schema 2
	# fib_indi[i,j]=3: voxel_ij contains fibers generated from schema 1 and schema 2 (crossing fibers)
	# simulation region plot can be obtained by rotating fib_indi 90 degree conter-clockwisely
	fib_indi = fib_indi_1 + 2*fib_indi_2

	# spherical coordinates of fiber directions in each voxel
	# coordinate system here is within each voxel, with x-axis perpendicular to the screen, 
	# and is different from coordinate system for generating fibers voxelwisely in simulation region
	theta_fib_1 = np.arctan2(r_xy_1, 1-r_z)  # polar angle of fiber 1
	theta_fib_2 = np.arctan2(r_xy_2, r_z)+np.pi/2 # polar angle of fiber 2
	theta_fib = np.stack((theta_fib_1, theta_fib_2), axis=-1)
	phi_fib_1 = np.ones((n, n, n))*0.75*np.pi  # azimuthal angle of fiber 1
	phi_fib_2 = np.ones((n, n, n))*0.25*np.pi  # azimuthal angle of fiber 2
	phi_fib = np.stack((phi_fib_1, phi_fib_2), axis=-1)

	# calculate noiseless DWI, DWI (with Rician noise) and SH coefficients of dirac functions
	DWI_noiseless = np.zeros((n, n, n, len(theta)))
	SH_coef = np.zeros((n, n, n, int((lmax+1)*(lmax+2)/2)))
	for i in range(n):
		for j in range(n):
			for k in range(n):
				if fib_indi[i, j, k] == 0:
					DWI_noiseless[i, j, k] = np.exp(-b)  # D becomes identity matrix in single tensor model
					SH_coef[i, j, k, 0] = spharmonic_eval(0, 0, 0, 0).real
				elif fib_indi[i, j, k] == 1:
					for l in range(len(theta)):
						DWI_1 = myresponse(b, ratio, theta_fib[i, j, k, 0], phi_fib[i, j, k, 0], theta[l], phi[l])
						DWI_noiseless[i, j, k, l] = DWI_1
					SH_coef[i, j, k] = spharmonic(theta_fib[i, j, k, :1], phi_fib[i, j, k, :1], lmax)
				elif fib_indi[i, j, k] == 2:
					for l in range(len(theta)):
						DWI_2 = myresponse(b, ratio, theta_fib[i, j, k, 1], phi_fib[i, j, k, 1], theta[l], phi[l])
						DWI_noiseless[i, j, k, l] = DWI_2
					SH_coef[i, j, k] = spharmonic(theta_fib[i, j, k, 1:], phi_fib[i, j, k, 1:], lmax)
				else:  # crossing-fiber case
					for l in range(len(theta)):
						DWI_1 = myresponse(b, ratio, theta_fib[i, j, k, 0], phi_fib[i, j, k, 0], theta[l], phi[l])
						DWI_2 = myresponse(b, ratio, theta_fib[i, j, k, 1], phi_fib[i, j, k, 1], theta[l], phi[l])
						DWI_noiseless[i, j, k, l] = (DWI_1+DWI_2)/2  # volumn fractions for two fibers are both 0.5
					SH_coef[i, j, k] = (spharmonic(theta_fib[i, j, k, :1], phi_fib[i, j, k, :1], lmax) 
						+ spharmonic(theta_fib[i, j, k, 1:], phi_fib[i, j, k, 1:], lmax))/2
	DWI = Rician_noise(DWI_noiseless, sigma, matlab_randn, seed)
	# take upper half of simulation region
	DWI_noiseless, DWI, SH_coef = DWI_noiseless[:, :, int(n/2):], DWI[:, :, int(n/2):], SH_coef[:, :, int(n/2):]
	theta_fib, phi_fib, fib_indi = theta_fib[:, :, int(n/2):], phi_fib[:, :, int(n/2):], fib_indi[:, :, int(n/2):]

	return DWI_noiseless, DWI, SH_coef, theta_fib, phi_fib, fib_indi

# generate identical two-fiber cases including DWI signals (noiseless and with Rician noise) 
# and SH coefficients of dirac functions on equal-angle grid specified by (theta, phi) in 2D simulation region
# n is the size of simulation region
# angle is the separation angle between two fibers
# DWI_noiseless: region size * region size * number of grid points, DWI signals without noise
# DWI: region size * region size * number of grid points, DWI signals with Rician noise
# SH_coef: region size * region size * number of symmetrized SH basis, SH coefficients of dirac functions
# theta_fib: region size * region size * 2, polar angle of fibers generated from schema 1 and schema 2
# phi_fib: region size * region size * 2, azimuthal angle of fibers generated from schema 1 and schema 2
# fib_indi: fib_indi[i,j]=3: voxel_ij contains fibers generated from schema 1 and schema 2 (crossing fibers)
def fiber_crossing(n, theta, phi, b, ratio, lmax, angle, sigma = 0.05, matlab_randn = False, seed = 0):

	# fib_indi[i,j]=3: voxel_ij contains fibers generated from schema 1 and schema 2 (crossing fibers)
	fib_indi = np.ones((n, n))*3

	# spherical coordinates of fiber directions in each voxel
	# coordinate system here is within each voxel, with x-axis perpendicular to the screen, 
	# and is different from coordinate system for generating fibers voxelwisely in simulation region
	theta_fib_1 = np.zeros((n, n))  # polar angle of fiber 1
	theta_fib_2 = np.ones((n, n))*angle # polar angle of fiber 2
	theta_fib = np.stack((theta_fib_1, theta_fib_2), axis=-1)
	phi_fib = np.ones((n, n, 2))*np.pi/2  # azimuthal angle of fiber 1 and fiber 2

	# calculate noiseless DWI, DWI (with Rician noise) and SH coefficients of dirac functions
	DWI_noiseless = np.zeros((n, n, len(theta)))
	SH_coef = np.zeros((n, n, int((lmax+1)*(lmax+2)/2)))
	for i in range(n):
		for j in range(n):
			for l in range(len(theta)):
				DWI_1 = myresponse(b, ratio, theta_fib[i, j, 0], phi_fib[i, j, 0], theta[l], phi[l])
				DWI_2 = myresponse(b, ratio, theta_fib[i, j, 1], phi_fib[i, j, 1], theta[l], phi[l])
				DWI_noiseless[i, j, l] = (DWI_1+DWI_2)/2  # volumn fractions for two fibers are both 0.5
			SH_coef[i, j] = (spharmonic(theta_fib[i, j, :1], phi_fib[i, j, :1], lmax) 
					+ spharmonic(theta_fib[i, j, 1:], phi_fib[i, j, 1:], lmax))/2
	DWI = Rician_noise(DWI_noiseless, sigma, matlab_randn, seed)

	return DWI_noiseless, DWI, SH_coef, theta_fib, phi_fib, fib_indi

# evaluate FOD estimation result by calculating Hellinger distance between estimated FOD and estimated FOD with 
# noiseless DWI (diff_noiseless), and Hellinger distance between estimated FOD and true FOD (diff_true)
def hellinger_dis_region(fod_all, fod_all_no, dirac_plot):

	n_index, S = fod_all.shape[:-2], fod_all.shape[-2]-1
	diff_no, diff_true = np.zeros(n_index + (S+1,)), np.zeros(n_index + (S+1,))

	for k in list(np.ndindex(n_index)):
		for s in range(S+1):
			diff_no[k][s] = hellinger_dis(fod_stand(fod_all[k][s]), fod_stand(fod_all_no[k]))
			diff_true[k][s] = hellinger_dis(fod_stand(fod_all[k][s]), fod_stand(dirac_plot[k]))

	diff_axis = tuple(range(len(n_index)))
	diff_no_mean, diff_no_std = diff_no.mean(axis=diff_axis), diff_no.std(axis=diff_axis, ddof=1)
	diff_true_mean, diff_true_std = diff_true.mean(axis=diff_axis), diff_true.std(axis=diff_axis, ddof=1)

	diff = {"smooth_round":range(S+1), "diff_noiseless(mean)":diff_no_mean, "diff_noiseless(std)":diff_no_std, 
	"diff_true(mean)":diff_true_mean, "diff_true(std)":diff_true_std}
	diff_table = pd.DataFrame(diff, columns=diff.keys())
	diff_table["smooth_round"] = diff_table["smooth_round"].apply(lambda x: "{:.0f}".format(x))
	diff_table.iloc[:, 1:] = diff_table.iloc[:, 1:].applymap(lambda x: "{:.3f}".format(x))
	print(diff_table)

# evaluate FOD estimation result by calcuating angular errors
def angular_error_region(fod_region, theta_fib, phi_fib, fib_indi, idx, nbhd, thresh, degree, pos, sampling_index):

	n_index = fod_region.shape[:-1]
	index_list = list(np.ndindex(n_index))

	# nfib_true: number of fibers in true FOD
	nfib_true = fib_indi.copy()
	nfib_true[nfib_true>1] -= 1
	n0fib, n1fib, n2fib = np.sum(nfib_true==0), np.sum(nfib_true==1), np.sum(nfib_true==2)

	# peak_pos_true: 3D coordinates of peaks in true FOD
	peak_pos_true = {}
	for k in index_list:
		if fib_indi[k] == 1:
			peak_pos_true[k] = np.array([np.sin(theta_fib[k][0])*np.cos(phi_fib[k][0]), 
				np.sin(theta_fib[k][0])*np.sin(phi_fib[k][0]), np.cos(theta_fib[k][0])])[..., np.newaxis]
		elif fib_indi[k] == 2:
			peak_pos_true[k] = np.array([np.sin(theta_fib[k][1])*np.cos(phi_fib[k][1]), 
				np.sin(theta_fib[k][1])*np.sin(phi_fib[k][1]), np.cos(theta_fib[k][1])])[..., np.newaxis]
		elif fib_indi[k] == 3:
			peak_pos_true[k] = np.array([[np.sin(theta_fib[k][0])*np.cos(phi_fib[k][0]), 
				np.sin(theta_fib[k][0])*np.sin(phi_fib[k][0]), np.cos(theta_fib[k][0])], 
				[np.sin(theta_fib[k][1])*np.cos(phi_fib[k][1]), np.sin(theta_fib[k][1])*np.sin(phi_fib[k][1]), 
				np.cos(theta_fib[k][1])]]).T

	# nfib: number of fibers in estimated FOD using peak detection function FOD_Peak
	# peak_pos: 3D coordinates of peaks in estimated FOD using peak detection function FOD_Peak
	nfib, peak_pos = np.zeros(n_index), {}
	for k in index_list:
		nfib[k], peak_pos[k] = FOD_Peak(fod_region[k], idx, nbhd, thresh, degree, pos, sampling_index, True)

	angle_error_1fib, angle_error_2fib, sep_error = [], [], []
	for k in index_list:
		if fib_indi[k] > 0 and peak_pos_true[k].shape[1] == peak_pos[k].shape[1]:
			if peak_pos_true[k].shape[1] == 1:
				angle_error_1fib.append(np.arccos(np.abs(peak_pos_true[k].T.dot(peak_pos[k])))/np.pi*180)
			if peak_pos_true[k].shape[1] == 2:
				angle_mat = np.arccos(np.abs(peak_pos_true[k].T.dot(peak_pos[k])))/np.pi*180
				angle_error_cand1, angle_error_cand2 = [angle_mat[0,0], angle_mat[1,1]], [angle_mat[0,1], angle_mat[1,0]]
				if np.sum(angle_error_cand1) <= np.sum(angle_error_cand2):
					angle_error_2fib += angle_error_cand1 
				else:
					angle_error_2fib += angle_error_cand2
				sep_error.append(np.abs(np.arccos(np.abs(peak_pos[k][:,0].T.dot(peak_pos[k][:,1]))) 
					- np.arccos(np.abs(peak_pos_true[k][:,0].T.dot(peak_pos_true[k][:,1]))))/np.pi*180)

	if n0fib>0:
		print("0-fiber: correct/under/over_rate {:.2f}/{:.2f}/{:.2f}".format(
			np.sum(nfib[nfib_true==0]==0)/n0fib, np.sum(nfib[nfib_true==0]<0)/n0fib, np.sum(nfib[nfib_true==0]>0)/n0fib))
	if n1fib>0:
		print("1-fiber: correct/under/over_rate {:.2f}/{:.2f}/{:.2f}, angle_error(mean/median) {:.2f}/{:.2f}".format(
			np.sum(nfib[nfib_true==1]==1)/n1fib, np.sum(nfib[nfib_true==1]<1)/n1fib, np.sum(nfib[nfib_true==1]>1)/n1fib, 
			np.mean(angle_error_1fib), np.median(angle_error_1fib)))
	if n2fib>0:
		print("2-fiber: correct/under/over_rate {:.2f}/{:.2f}/{:.2f}, angle_error(mean/median) {:.2f}/{:.2f}, separation_error(mean/median) {:.2f}/{:.2f}".format(
			np.sum(nfib[nfib_true==2]==2)/n2fib, np.sum(nfib[nfib_true==2]<2)/n2fib, np.sum(nfib[nfib_true==2]>2)/n2fib, 
			np.mean(angle_error_2fib), np.median(angle_error_2fib), np.mean(sep_error), np.median(sep_error)))