load('ROI.mat');
%load('ROIs.mat');
%load('ROIss.mat');
gcp;

[n1, n2, n] = size(DWI);
[N, p] = size(Constraint);

%parameter setting for ROI
L = 50;
lambda_seq = 10.^(linspace(-2, -5, L));
stop_percent = 0.05; 
stop_thresh = 2e-4;

%parameter setting for ROIs and ROIss
%L = 100;
%lambda_seq = 10.^(linspace(0, -5, L));
%stop_percent = 0.05;
%stop_thresh = 1e-3;

ep_r = 1e-2;
ep_a = 1e-4;
maxit = 5000;

DWI_weighted = DWI_noiseless;

stop_index_all_vec = zeros(n1*n2, 1);
beta_all_vec = zeros(n1*n2, p);
fod_all_vec = zeros(n1*n2, N);
DWI_weighted_vec = reshape(DWI_weighted, n1*n2, n);

tic;
parfor k = 1:n1*n2
    
    [beta, sn_stop_index] = sn_classo(DWI_weighted_vec(k, :)', design_SN, ...
        Constraint, lambda_seq, stop_percent, stop_thresh, ep_a, ep_r, maxit, 0);
        
    stop_index_all_vec(k) = sn_stop_index;
    beta_all_vec(k, :) = beta;
    fod_all_vec(k, :) = Constraint * beta;
        
    sprintf('k1 = %d, k2 = %d', rem(k-1, n1)+1, fix((k-1)/n1)+1)
end
time = toc;

stop_index_all_no = reshape(stop_index_all_vec, n1, n2);
beta_all_no = reshape(beta_all_vec, n1, n2, p);
fod_all_no = reshape(fod_all_vec, n1, n2, N);

save('ROI_no_result.mat', 'stop_index_all_no', 'beta_all_no', 'fod_all_no')
%save('ROIs_no_result.mat', 'stop_index_all_no', 'beta_all_no', 'fod_all_no')
%save('ROIss_no_result.mat', 'stop_index_all_no', 'beta_all_no', 'fod_all_no')
