#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "classo.h"
#include <sys/time.h>

extern void dgemv_(char*, int*, int*, double*, double*, int*, double*, int*, double*, double*, int*);

//ADMM algorithm on constraint lasso with RSS-based lambda selection
void sn_classo(double *y, double *XT, double *CT, double *inv_seq, double *gamma, double *Lambda_seq, double *Rho_seq, 
	int *P, int *N, int *L, int *Len, double *stop_percent, double *stop_thresh, double *Epi_abs, double *Epi_rel, 
	int *Max_step, int *Verbose, double *record){

	int p = *P, n = *N, l = *L, len = *Len;
	double *eta = (double *) malloc(l*sizeof(double));
	double *u = (double *) malloc(p*sizeof(double));
	double *t = (double *) malloc(l*sizeof(double));
	double *C = (double *) malloc(l*p*sizeof(double));
	double *Xy = (double *) malloc(p*sizeof(double));
	double *RSS = (double *) malloc(len*sizeof(double));
	int stop_len = floor(*stop_percent * len), iter, i, j, one = 1;
	double stop_space = log10(Lambda_seq[0]) - log10(Lambda_seq[1]), mean_diff_rel, a = 1, b = 0;
	double *diff_rel = (double *) malloc(len*sizeof(double));
	double *temp_n = (double *) malloc(n*sizeof(double));
	struct timeval t1, t2;

	gettimeofday(&t1, NULL);
	for(i=0; i<l; i++){
		eta[i] = 0;
		t[i] = 0;
	}
	for(i=0; i<p; i++){
		u[i] = 0;
	}
	for(i=0; i<l; i++){
		for(j=0; j<p; j++){
			C[p*i+j] = CT[l*j+i];
		}
	} //C=CT'
	dgemv_("T", N, P, &a, XT, N, y, &one, &b, Xy, &one); //Xy=XT*y
	gettimeofday(&t2, NULL);
	record[0] += (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec)/1000000.0;

	for(iter=0; iter<len; iter++){

		classo(Xy, inv_seq+p*p*iter, CT, C, gamma, eta, u, t, P, N, L, Lambda_seq+iter, Rho_seq+iter, Epi_abs, Epi_rel, 
			Max_step, Verbose, record);

		gettimeofday(&t1, NULL);
		RSS[iter] = 0;
		dgemv_("N", N, P, &a, XT, N, gamma, &one, &b, temp_n, &one); //temp_n=XT'*gamma
		for(i=0; i<n; i++){
			RSS[iter] += pow(temp_n[i]-y[i], 2);
		} //RSS[iter]=norm(XT'*gamma-y)^2

		if(*Verbose == 1){
			printf("iter = %d, RSS = %f\n", iter, RSS[iter]);
		}

		if(iter > 0){
			diff_rel[iter-1] = fabs(log10(RSS[iter]) - log10(RSS[iter-1])) / stop_space;
		}

		if(iter >= stop_len){
			mean_diff_rel = 0;
			for(j=iter-stop_len; j<iter; j++){
				mean_diff_rel += diff_rel[j];
			}
			mean_diff_rel /= stop_len;
			if(mean_diff_rel < *stop_thresh){
				*Len = iter;
				break;
			}
		}
		gettimeofday(&t2, NULL);
		record[1] += (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec)/1000000.0;
	}

	free(eta);
	free(u);
	free(t);
	free(C);
	free(Xy);
	free(RSS);
	free(diff_rel);
	free(temp_n);
}

//ADMM algorithm on constraint lasso
void classo(double *Xy, double *inv, double *CT, double *C, double *gamma, double *eta, double *u, double *t, int *P, 
	int *N, int *L, double *Lambda, double *Rho, double *Epi_abs, double *Epi_rel, int *Max_step, 
	int *Verbose, double *record){

	int p = *P, l = *L, max_step = *Max_step, i, j, one = 1;
	double lambda = *Lambda, rho = *Rho, epi_abs = *Epi_abs, epi_rel = *Epi_rel, alpha = 1.5;
	double prd = 1, drd = 1, r, s, epi_pri, epi_dual, coef, epi_pri_1, epi_pri_2, a = 1, b = 0;
	double *beta = (double *) malloc(p*sizeof(double));
	double *gamma_prev = (double *) malloc(p*sizeof(double));
	double *eta_prev = (double *) malloc(l*sizeof(double));
	double *temp_p1 = (double *) malloc(p*sizeof(double));
	double *temp_p2 = (double *) malloc(p*sizeof(double));
	double *temp_l = (double *) malloc(l*sizeof(double));
	struct timeval t1, t2;

	//ADMM iteration
	while((prd>0 || drd>0) && max_step>0){

		r = 0, s = 0, epi_dual = 0, epi_pri_1 = 0, epi_pri_2 = 0;

		gettimeofday(&t1, NULL);
		for(j=0; j<l; j++){
			temp_l[j] = eta[j] - t[j];
		}
		dgemv_("T", L, P, &a, CT, L, temp_l, &one, &b, temp_p1, &one); //temp_p1=CT*(eta-t)
		for(i=0; i<p; i++){
			temp_p2[i] = Xy[i] + rho * (gamma[i] - u[i] + temp_p1[i]);
		} //beta=Xy+rho*(gamma-u+CT*(eta-t))
		gettimeofday(&t2, NULL);
		record[2] += (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec)/1000000.0;
		gettimeofday(&t1, NULL);
		dgemv_("T", P, P, &a, inv, P, temp_p2, &one, &b, beta, &one); //beta=inv*(Xy+rho*(gamma-u+CT*(eta-t)))
		gettimeofday(&t2, NULL);
		record[3] += (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec)/1000000.0;

		gettimeofday(&t1, NULL);
		for(i=0; i<p; i++){
			gamma_prev[i] = gamma[i];
			u[i] += alpha * beta[i] + (1-alpha) * gamma[i]; //beta_alpha=alpha*beta+(1-alpha)*gamma
			coef = 1 - lambda / (rho * fabs(u[i]));
			if(coef <= 0){
				gamma[i] = 0;
				r += pow(beta[i], 2);
			}
			else{
				gamma[i] = coef * u[i]; //gamma=S_lambda/rho(beta_alpha+u)
				u[i] -= gamma[i]; //u=beta_alpha-gamma
				r += pow(beta[i]-gamma[i], 2); //r+=norm(beta-gamma)^2
				epi_pri_2 += pow(gamma[i], 2); //epi_pri_2+=norm(gamma)^2
			}
			epi_pri_1 += pow(beta[i], 2); //epi_pri_1+=norm(beta)^2
		}
		gettimeofday(&t2, NULL);
		record[4] += (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec)/1000000.0;

		gettimeofday(&t1, NULL);
		dgemv_("T", P, L, &a, C, P, beta, &one, &b, temp_l, &one); //temp_l=C*beta
		gettimeofday(&t2, NULL);
		record[5] += (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec)/1000000.0;
		gettimeofday(&t1, NULL);
		for(j=0; j<l; j++){
			eta_prev[j] = eta[j];
			t[j] += alpha * temp_l[j] + (1-alpha) * eta[j]; //temp_alpha=alpha*CT'*beta+(1-alpha)*eta
			if(t[j] <= 0){
				eta[j] = 0;
				r += pow(temp_l[j], 2);
			}
			else{
				eta[j] = t[j]; //eta=max(0,temp_alpha+t)
				t[j] = 0; //t=t+temp_alpha-eta
				r += pow(temp_l[j]-eta[j], 2); //r+=norm(CT'*beta-eta)^2
				epi_pri_2 += pow(eta[j], 2); //epi_pri_2+=norm(eta)^2
			}
			epi_pri_1 += pow(temp_l[j], 2); //epi_pri_1+=norm(CT'*beta)^2
		}
		gettimeofday(&t2, NULL);
		record[6] += (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec)/1000000.0;

		gettimeofday(&t1, NULL);
		r = sqrt(r);
		epi_pri = sqrt(p+l) * epi_abs + epi_rel * sqrt((epi_pri_1>epi_pri_2)?epi_pri_1:epi_pri_2);
		prd = (r>epi_pri);

		if(prd<=0){
			for(j=0; j<l; j++){
				temp_l[j] = eta[j] - eta_prev[j];
			}
			dgemv_("T", L, P, &a, CT, L, temp_l, &one, &b, temp_p1, &one); //temp_p1=CT*(eta-eta_prev)
			dgemv_("T", L, P, &a, CT, L, t, &one, &b, temp_p2, &one); //temp_p2=CT*t
			for(i=0; i<p; i++){
				s += pow(gamma[i]-gamma_prev[i]+temp_p1[i], 2); //s=norm(gamma-gamma_prev+CT*(eta-eta_prev))^2
				epi_dual += pow(u[i]+temp_p2[i], 2); //epi_dual=norm(u+CT*t)^2
			}
			s = rho * sqrt(s);
			epi_dual = sqrt(p) * epi_abs + epi_rel * rho * sqrt(epi_dual);
			drd = (s>epi_dual);
		}
		gettimeofday(&t2, NULL);
		record[7] += (t2.tv_sec - t1.tv_sec) + (t2.tv_usec - t1.tv_usec)/1000000.0;
		
		max_step--;
	}//end ADMM iteration

	if(*Verbose == 1){
		printf("total step = %d\n", *Max_step-max_step);
	}

	if(max_step == 0){
		printf("Warning: algorithm does not converge at max step = %d!\n", *Max_step);
	}

	free(beta);
	free(gamma_prev);
	free(eta_prev);
	free(temp_p1);
	free(temp_p2);
	free(temp_l);
}