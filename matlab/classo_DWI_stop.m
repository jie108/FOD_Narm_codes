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

ch = 1.15;
S = 10;

stop_index_all = zeros(n1, n2, S+1);
beta_all = zeros(n1, n2, S+1, p);
fod_all = zeros(n1, n2, S+1, N);
stop_s = zeros(n1, n2); %record stopping at which s for each voxel (0 means no stopping in the end)

tic;
for s = 0:S
    
    if s == 0
        DWI_weighted = DWI;
    else
        %fix scaling=false
        DWI_weighted = DWI_weight(DWI, ch^s, 8, squeeze(fod_all(:, :, s, :)), false, 0.9);
    end
    
    stop_index_vec = zeros(n1*n2, 1);
    beta_vec = zeros(n1*n2, p);
    fod_vec = zeros(n1*n2, N);
    stop_s_vec = reshape(stop_s, n1*n2, 1);
    DWI_weighted_vec = reshape(DWI_weighted, n1*n2, n);
    
    parfor k = 1:n1*n2
        
        if stop_s_vec(k) == 0 %not stopped yet
            [beta, sn_stop_index] = sn_classo(DWI_weighted_vec(k, :)', design_SN, ...
                Constraint, lambda_seq, stop_percent, stop_thresh, ep_a, ep_r, maxit, 0);
        
            stop_index_vec(k) = sn_stop_index;
            beta_vec(k, :) = beta;
            fod_vec(k, :) = Constraint * beta;
        end
        
        sprintf('s = %d, k1 = %d, k2 = %d', s, rem(k-1, n1)+1, fix((k-1)/n1)+1)
    end
    
    if s == 0
        stop_index_all(:, :, s+1) = reshape(stop_index_vec, n1, n2);
        beta_all(:, :, s+1, :) = reshape(beta_vec, n1, n2, p);
        fod_all(:, :, s+1, :) = reshape(fod_vec, n1, n2, N);
    else
        stop_index_temp = reshape(stop_index_vec, n1, n2);
        beta_temp = reshape(beta_vec, n1, n2, p);
        fod_temp = reshape(fod_vec, n1, n2, N);
        
        for k1 = 1:n1
            for k2 = 1:n2
                if stop_s(k1, k2) == 0 %not stopped yet
                    if hellinger_dis(fod_stand(squeeze(fod_temp(k1, k2, :))), fod_stand(squeeze(fod_all(k1, k2, s, :)))) > chi2inv(0.6/s, 1)*4
                        %meet the stopping criterion, record s and use previous result as current result
                        stop_s(k1, k2) = s;
                        stop_index_all(k1, k2, s+1) = stop_index_all(k1, k2, s);
                        beta_all(k1, k2, s+1, :) = beta_all(k1, k2, s, :);
                        fod_all(k1, k2, s+1, :) = fod_all(k1, k2, s, :);
                    else
                        %not meet the stopping criterion, record current result
                        stop_index_all(k1, k2, s+1) = stop_index_temp(k1, k2);
                        beta_all(k1, k2, s+1, :) = beta_temp(k1, k2, :);
                        fod_all(k1, k2, s+1, :) = fod_temp(k1, k2, :);
                    end
                else
                    %already stopped, use previous result as current result
                    stop_index_all(k1, k2, s+1) = stop_index_all(k1, k2, s);
                    beta_all(k1, k2, s+1, :) = beta_all(k1, k2, s, :);
                    fod_all(k1, k2, s+1, :) = fod_all(k1, k2, s, :);
                end
            end
        end
    end
end
time = toc;

save('ROI_result.mat', 'stop_index_all', 'beta_all', 'fod_all', 'stop_s')
%save('ROIs_result.mat', 'stop_index_all', 'beta_all', 'fod_all', 'stop_s')
%save('ROIss_result.mat', 'stop_index_all', 'beta_all', 'fod_all', 'stop_s')
