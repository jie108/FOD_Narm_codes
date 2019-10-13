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

ep_a = 1e-4;
ep_r = 1e-2;
maxit = 5000;

ch = 1.15;
S = 10;

stop_index_all = zeros(n1, n2, S+1);
beta_all = zeros(n1, n2, S+1, p);
fod_all = zeros(n1, n2, S+1, N);


tic;
for s = 0:S
    
    if s == 0
        DWI_weighted = DWI;
    else
        %reweight DWI with h=ch^s in Kloc
        %a=8, fod_prev=squeeze(fod_all(:, :, s, :)), scaling = true, q=0.9 in Kst
        %or a=8, fod_prev=squeeze(fod_all(:, :, s, :)), scaling = false in Kst
        DWI_weighted = DWI_weight(DWI, ch^s, 8, squeeze(fod_all(:, :, s, :)), true, 0.9);
    end
    
    stop_index_vec = zeros(n1*n2, 1);
    beta_vec = zeros(n1*n2, p);
    fod_vec = zeros(n1*n2, N);
    DWI_weighted_vec = reshape(DWI_weighted, n1*n2, n);
    
    parfor k = 1:n1*n2
        
        [beta, sn_stop_index] = sn_classo(DWI_weighted_vec(k, :)', design_SN, ...
            Constraint, lambda_seq, stop_percent, stop_thresh, ep_a, ep_r, maxit, 0);
        
        stop_index_vec(k) = sn_stop_index;
        beta_vec(k, :) = beta;
        fod_vec(k, :) = Constraint * beta;
        
        sprintf('s = %d, k1 = %d, k2 = %d', s, rem(k-1, n1)+1, fix((k-1)/n1)+1)
    end
    
    stop_index_all(:, :, s+1) = reshape(stop_index_vec, n1, n2);
    beta_all(:, :, s+1, :) = reshape(beta_vec, n1, n2, p);
    fod_all(:, :, s+1, :) = reshape(fod_vec, n1, n2, N);
end
time = toc;

save('ROI_result.mat', 'stop_index_all', 'beta_all', 'fod_all')
%save('ROIs_result.mat', 'stop_index_all', 'beta_all', 'fod_all')
%save('ROIss_result.mat', 'stop_index_all', 'beta_all', 'fod_all')
