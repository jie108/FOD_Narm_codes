import numpy as np
cimport numpy as np

# external declaration of C function "sn_classo"
cdef extern from "classo.h":
	void sn_classo(double *y, double *XT, double *CT, double *inv_seq, double *gamma, double *Lambda_seq, double *Rho_seq, 
		int *P, int *N, int *L, int *Len, double *stop_percent, double *stop_thresh, double *Epi_abs, double *Epi_rel, 
		int *Max_step, int *Verbose, double *record);

# implementation of C function "sn_classo" (wrapper function)
# design matrix and constraint matrix are transposed
# gamma: estimated SN coefficient
# vertex_matrix.dot(gamma): estimated FOD on dense grid
# Len: index of selected lambda in lambda_seq
def SN_CLasso(np.ndarray[double, ndim=1, mode="c"] DWI, np.ndarray[double, ndim=2, mode="c"] design_matrix_T, 
	np.ndarray[double, ndim=2, mode="c"] constraint_T, np.ndarray[double, ndim=2, mode="c"] vertex_matrix, 
	np.ndarray[double, ndim=3, mode="c"] inv_seq, np.ndarray[double, ndim=1, mode="c"] lambda_seq, 
	np.ndarray[double, ndim=1, mode="c"] rho_seq, double stop_percent, double stop_thresh, double epi_abs=1e-4, 
	double epi_rel=1e-2, int max_step=5000, int verbose=0):

	cdef int p = design_matrix_T.shape[0]
	cdef int n = design_matrix_T.shape[1]
	cdef int l = constraint_T.shape[1]
	cdef int Len = lambda_seq.shape[0]
	cdef np.ndarray[double, ndim=1, mode="c"] gamma = np.ascontiguousarray(np.zeros(p, dtype=np.float64))
	cdef np.ndarray[double, ndim=1, mode="c"] record = np.ascontiguousarray(np.zeros(8, dtype=np.float64))

	sn_classo(&DWI[0], &design_matrix_T[0, 0], &constraint_T[0, 0], &inv_seq[0, 0, 0], &gamma[0], &lambda_seq[0], 
		&rho_seq[0], &p, &n, &l, &Len, &stop_percent, &stop_thresh, &epi_abs, &epi_rel, &max_step, &verbose, &record[0])

	return gamma, vertex_matrix.dot(gamma), Len, record