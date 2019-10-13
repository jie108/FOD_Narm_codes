//ADMM algorithm on constraint lasso with RSS-based lambda selection
void sn_classo(double *y, double *XT, double *CT, double *inv_seq, double *gamma, double *Lambda_seq, double *Rho_seq, 
	int *P, int *N, int *L, int *Len, double *stop_percent, double *stop_thresh, double *Epi_abs, double *Epi_rel, 
	int *Max_step, int *Verbose, double *record);

//ADMM algorithm on constraint lasso
void classo(double *Xy, double *Chol, double *CT, double *C, double *gamma, double *eta, double *u, double *t, int *P, 
	int *N, int *L, double *Lambda, double *Rho, double *Epi_abs, double *Epi_rel, int *Max_step, 
	int *Verbose, double *record);