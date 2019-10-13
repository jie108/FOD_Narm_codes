clear all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% set path 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
path.cur='/Users/jlyang/Documents/FOD/DMRI_code/';

%%% add path of the wavelet package and wavelet_meshes to generate equi-angle grid 
addpath(path.cur);
addpath(strcat(path.cur,'toolbox_wavelet_meshes'));   
addpath(strcat(path.cur,'toolbox_wavelet_meshes/toolbox'));

%%% add path of ADMM functions and helper functions
addpath(strcat(path.cur,'ADMM'));
addpath(strcat(path.cur,'help_functions'));
addpath(strcat(path.cur,'construction_functions'));

%%% needlets package 
addpath(strcat(path.cur,'MEALPix'));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% simulation set up
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

lmax = 8;  % SH levels
jmax = 3; % SN levels corresponding to lmax
b = [1, 1]; % back ground magnetic field strength (1 is same as b=1000)
ratio = [10, 10]; % ratio of the leading eigenvalue to the second eigenvalue in the signal simulation
weight = [0.5, 0.5]; % weight of each fiber in the voxel
J = 2.5; % vertex level (decides number of observation of this voxel)
J_r = 5; % vertex level used for graph and representation purpose (dense)
b_response = b(1);  % b value for the response matrix Rmatrix that we use
ratio_response = ratio(1); % shape of the response function, could be misspecified
weight_response = 1;
%% noise level 
sigma = 0.05;  %%middle  noise: note S0=1, so SNR=20 

J_sample = 2; % 1->21 directions; 2->41; 3->81; 4->319
if(J_sample == 3)
    n_sample = 81;
elseif(J_sample == 4)
    n_sample =321;
elseif(J_sample == 2)
    n_sample = 41;
elseif(J_sample == 1)
    n_sample = 21;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% using equal-angle grid with level J (3->81 direction on half sphere, 4->321; 2->21)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

options.base_mesh = 'ico';
options.relaxation = 1;
options.keep_subdivision = 1;

if(J == 2.5)
    [vertex,~] = compute_semiregular_sphere(3,options); %%vertex and face of the grid
else
    [vertex,~] = compute_semiregular_sphere(J,options); %%vertex and face of the grid
end
pos = vertex{end};  %% x-y-z coordinates of the vertex 

% spherical coordinates of the vertex
phi = atan2(pos(2,:),pos(1,:))/(2*pi);   %%phi: azimuthal  angle, [0,1)
phi = phi+(phi<0);
theta = acos(pos(3,:))/(pi);             %% theta: polar angle, [0,1)

pos_corr = pos'*pos;
pos_pair = zeros(size(pos_corr,1),1);

% find the paired cabature points
for i = 1:size(pos_pair,1)
    pos_pair(i) = find(pos_corr(i,:)<-0.9999999);
end

sampling_index = zeros(size(pos_corr,1),1); % only uses half of the symmetrized needlets
for i = 1:size(pos_pair,1)
    if(pos(3,i)>-1e-15&&pos(3,i)<1e-15)
        if(pos(2,i)>-1e-15&&pos(2,i)<1e-15)
            if(pos(1,i)>0)
                sampling_index(i) = i;
            else
                sampling_index(i) = pos_pair(i);
            end
        elseif(pos(2,i)>1e-15)
            sampling_index(i) = i;
        elseif(pos(2,i)<-1e-15)
            sampling_index(i) = pos_pair(i);
        end
    elseif(pos(3,i)>1e-15)
        sampling_index(i) = i;
    elseif(pos(3,i)<-1e-15)
        sampling_index(i) = pos_pair(i);
    end
end
sampling_index = unique(sampling_index);

if(n_sample == 41)
    pos_sampling_h = pos(:,sampling_index); %% position of the half-sphere grid points 
    phi_h=phi(:,sampling_index)*180; 
    theta_h=theta(:,sampling_index)*180;

    %%% take 40 out of these 81 directions: at each level of theta, take about
    %%% half phi 
    index_1=find(theta_h<10); %%only 1
    n_1=size(index_1,2);

    index_t=find(theta_h>10&theta_h<20);
    n_2=size(index_t,2); %%6 
    [~, I]=sort(phi_h(index_t));
    index_2=index_t(1, I([1 3 5]));

    index_t=find(theta_h>20&theta_h<40);
    n_3=size(index_t,2); %%12
    [~, I]=sort(phi_h(index_t));
    index_3=index_t(1, I([1 3 5 7 9 11]));

    index_t=find(theta_h>40&theta_h<50);
    n_4=size(index_t,2); %%12
    [~, I]=sort(phi_h(index_t));
    index_4=index_t(1, I([1 3 5 7 9 11]));

    index_t=find(theta_h>50&theta_h<70);
    n_5=size(index_t,2); %%20
    [~, I]=sort(phi_h(index_t));
    index_5=index_t(1, I([1 3 5 7 9 11 13 15 17 19]));

    index_t=find(theta_h>70&theta_h<85);
    n_6=size(index_t,2); %%22
    [~, I]=sort(phi_h(index_t));
    index_6=index_t(1, I([1 3 5 7 9 11 13 15 17 19 21]));

    index_t=find(theta_h>85);
    n_7=size(index_t,2); %%8
    [~, I]=sort(phi_h(index_t));
    index_7=index_t(1, I([1 3 5 7]));

    index_s=unique([index_1 index_2 index_3 index_4 index_5 index_6 index_7]);
    sampling_grid_index=sampling_index(index_s); 
else
    sampling_grid_index=sampling_index;
end
clearvars index_1 index_2 index_3 index_4 index_5 index_6 index_7 index_t I i n_1 n_2 n_3 n_4 n_5 n_6 n_7 pos_corr pos_pair sampling_index phi_h theta_h;

pos_sampling = pos(:,sampling_grid_index); %% The x , y , z coordinates of the sampling grid.
phi_sampling = phi(:,sampling_grid_index); %% The sampled phi.
theta_sampling = theta(:,sampling_grid_index); %% The sampled theta.

% denser grid for interpolation and plots 
[v_plot,f_plot] = compute_semiregular_sphere(J_r,options);
pos_plot = v_plot{end};
phi_p = atan2(pos_plot(2,:),pos_plot(1,:))/(2*pi);   %%phi: azimuthal  angle, [0,1)
phi_p = phi_p+(phi_p<0);
theta_p = acos(pos_plot(3,:))/(pi);             %% theta: polar angle, [0,1)

%%%plotting options 
options.spherical = 1;
% options for the display
options.use_color = 1;
options.color = 'wavelets';
options.use_elevation = 2;
options.rho = 0.5;
options.scaling = 1.5;
% for draw_fiber
plot_rho = options.rho;
plot_scale = options.scaling;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Set up design matrices and penalty parameters 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% symmetric needlets design matrix
% generate SH matrix on sample grids, load pre-stored ones will be much faster
SH_matrix = SH_vertex(J, lmax, 1);
SH_matrix_plot = SH_vertex(J_r, lmax, 0);
% generate response matrix, load pre-stored ones will be much much faster
R_matrix = Response_Rmatrix_construction(b_response,ratio_response,J_r,lmax);
design_SH = SH_matrix*R_matrix;     % design matrix

Constraint = SN_vertex_symm(J_r,jmax,2,0);  %% constraint matrix: Constraint*beta>=0;
SN_vertex_symm = Constraint;
C_trans_symm = C_trans_symm_construction(lmax);  %% beta = C_trans_symm'*f
C_trans=(C_trans_symm*C_trans_symm')\C_trans_symm; %% f = C_trans*beta
design_SN = design_SH*C_trans;  %% design_SH = Phi*R

%% location of cabature points
% location of cabature points for each level
pix_all = zeros(3, 12*(1-4^(jmax+1))/(1-4));
for i = 1:(jmax+1)
    pix_temp = cell2mat(pix2vec(2^(i-1), 'nest', false));  % Nside = 2^(i-1)
    pix_all(:, (12*(1-4^(i-1))/(1-4)+1):(12*(1-4^(i))/(1-4))) = pix_temp;
end
        
% find the paired cabature points
cabature_corr = pix_all' * pix_all;
cabature_pair = zeros(size(cabature_corr, 1), 1);
for i = 1:size(cabature_pair,1)
    cabature_pair(i) = find(cabature_corr(i,:) < -0.9999999);
end

% only uses half of the symmetrized needlets
cabature_use = zeros(size(cabature_corr,1)/2, 1);
count = 1;
for i = 1:size(cabature_pair, 1)
    if cabature_pair(i) > i
        cabature_use(count) = i;
        count = count+1;
    end
end
pix_all_use = pix_all(:, cabature_use);

%% sequences of penalty parameters 
 
%%% for classo
lambda_min_la = 1e-5;
lambda_max_la = 1e-2;
lambda_length_la = 50;
lambda_seq_la=10.^(linspace(log10(lambda_max_la), log10(lambda_min_la), lambda_length_la));
stop_percent = 0.05;
stop_length = floor(stop_percent*lambda_length_la);   
stop_thresh = 2e-4;
stop_spacing = log10(lambda_seq_la(2))-log10(lambda_seq_la(1));
%%% admm stoping criterion
ep_r=1e-2;
ep_a=1e-4;

maxit=5000;
print=1;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% set simulation region for ROI
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n1=10;
n2=10;

%% functions in the region
x = linspace(0, 1, n1+1); 
y = linspace(0, 1, n2+1);
f1 = sqrt(1-(1-x).^2);
f2 = real(sqrt(0.36-(1-x).^2));
f3 = sqrt(1-x.^2);
f4 = real(sqrt(0.36-x.^2));

fib_indi = zeros(n1, n2);
for i = 1:n1
    for j = 1:n2
        lb = y(j);
        up = y(j+1);
        fib1 = ~((lb>=f1(i+1))|(up<=(f2(i))));
        fib2 = 2*(~(((lb>=f3(i))|(up<=f4(i+1)))));
        fib_indi(i,j) = fib1+fib2;  % rotate fib_indi 90 degree counterclockwise to obtain matrix in plot
    end
end

r_x = linspace(1/(2*n1), 1-1/(2*n1), n1); 
r_y = linspace(1/(2*n2), 1-1/(2*n2), n2);
[r_y, r_x] = meshgrid(r_y, r_x); 

slope_f1 = (1-r_y)./r_x.*(fib_indi==1|fib_indi==3);
l2_norm1 = sqrt((slope_f1.^2+(ones(n1, n2)).^2));
slope_f2 = -r_y./r_x.*(fib_indi==2|fib_indi==3);
l2_norm2 = sqrt((slope_f2.^2+(ones(n1, n2)).^2));

%% weights in each voxel
w1_matrix = (fib_indi==1)+0.5.*(fib_indi==3);
w2_matrix = (fib_indi==2)+0.5.*(fib_indi==3);

%% angles in each voxel
theta_fib = zeros(n1,n2,2);
phi_fib = zeros(n1,n2,2);

for i=1:n1
    for j=1:n2   
        theta_fib(i,j,1) = atan(1/slope_f1(i,j));
        theta_fib(i,j,2) = atan(-1/slope_f2(i,j))+pi/2;
        phi_fib(i,j,1) = pi/2;
        phi_fib(i,j,2) = pi/2;
    end
end

dirac_sh_all = zeros(n1,n2,2562);
dirac_sh1_all = zeros(n1,n2,2562);
dirac_sh2_all = zeros(n1,n2,2562);

coe_sn_all = zeros(n1, n2, 511);
coe_sn1_all = zeros(n1, n2, 511);
coe_sn2_all = zeros(n1, n2, 511);

angle1_DWI_all = zeros(n1, n2, 41);
angle2_DWI_all = zeros(n1, n2, 41);

angle1_all = zeros(n1, n2, 511);
angle2_all = zeros(n1, n2, 511);

angle1_plot_all = zeros(n1, n2, 2562);
angle2_plot_all = zeros(n1, n2, 2562);

for i=1:n1
    for j=1:n2
        coe_sh1 = Dirac_SH_coe(lmax,theta_fib(i,j,1),phi_fib(i,j,1)); %% SH coefficients 
        coe_sh2 = Dirac_SH_coe(lmax,theta_fib(i,j,2),phi_fib(i,j,2));
        coe_sh = w1_matrix(i,j).*coe_sh1 + w2_matrix(i,j).*coe_sh2;
        
        coe_sn1_all(i, j, :) = C_trans_symm' * coe_sh1;
        coe_sn2_all(i, j, :) = C_trans_symm' * coe_sh2;
        coe_sn_all(i, j, :) = C_trans_symm' * coe_sh;
        
        angle1_DWI_all(i, j, :) = acos(abs(pos_sampling'*[0; sin(theta_fib(i,j,1)); cos(theta_fib(i,j,1))]))/pi*180;
        angle2_DWI_all(i, j, :) = acos(abs(pos_sampling'*[0; sin(theta_fib(i,j,2)); cos(theta_fib(i,j,2))]))/pi*180;

        angle1_all(i, j, :) = [0; acos(abs(pix_all_use'*[0; sin(theta_fib(i,j,1)); cos(theta_fib(i,j,1))]))/pi*180];
        angle2_all(i, j, :) = [0; acos(abs(pix_all_use'*[0; sin(theta_fib(i,j,2)); cos(theta_fib(i,j,2))]))/pi*180];
        
        angle1_plot_all(i, j, :) = acos(abs(pos_plot'*[0; sin(theta_fib(i,j,1)); cos(theta_fib(i,j,1))]))/pi*180;
        angle2_plot_all(i, j, :) = acos(abs(pos_plot'*[0; sin(theta_fib(i,j,2)); cos(theta_fib(i,j,2))]))/pi*180;

        dirac_sh1 = SH_matrix_plot*coe_sh1; 
        dirac_sh2 = SH_matrix_plot*coe_sh2;
        dirac_sh = w1_matrix(i,j).*dirac_sh1 + w2_matrix(i,j).*dirac_sh2;  %%SH representation
        dirac_sh_all(i,j,:) = dirac_sh; 
        dirac_sh1_all(i,j,:) = dirac_sh1; 
        dirac_sh2_all(i,j,:) = dirac_sh2; 
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% generate dwi signals on the equal-angle grid/gradient-direction grid (J=3, 81*2 points) 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% noiseless DWI
DWI_noiseless_all = zeros(n1,n2,size(theta,2)); 
DWI_all = zeros(n1,n2,size(theta,2)); 

rng(0)
for i = 1:n1
    for j = 1:n2
        for at = 1:size(theta, 2)
            w = [w1_matrix(i,j),w2_matrix(i,j)];
            DWI_noiseless_all(i,j,at) = myresponse_crossing([b_response b_response],[ratio_response ratio_response],w,theta_fib(i,j,:),phi_fib(i,j,:),theta(at)*pi,phi(at)*2*pi); %%162 by 1 
        end
        DWI_all(i,j,:) = add_Rician_noise(DWI_noiseless_all(i,j,:),sigma);
    end
end


%%% get observed DWI on the sampling grid
DWI_noiseless=DWI_noiseless_all(:,:,sampling_grid_index);
DWI=DWI_all(:,:,sampling_grid_index);

save('ROI.mat', 'Constraint', 'design_SN', 'DWI', 'DWI_noiseless')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% set simulation region for ROIs and ROIss
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

n1=10;
n2=10;

%% functions in the region
x = linspace(0, 1, n1+1); 
y = linspace(0, 1, n2+1);
f1 = sqrt(1-(1-x).^2);
f2 = real(sqrt(0.64-(1-x).^2)); %ROIs
%f2 = real(sqrt(0.81-(1-x).^2)); %ROIss
f3 = sqrt(1-x.^2);
f4 = real(sqrt(0.64-x.^2)); %ROIs
%f4 = real(sqrt(0.81-x.^2)); %ROIss

fib_indi = zeros(n1, n2);
for i = 1:n1
    for j = 1:n2
        lb = y(j);
        up = y(j+1);
        fib1 = ~((lb>=f1(i+1))|(up<=(f2(i))));
        fib2 = 2*(~(((lb>=f3(i))|(up<=f4(i+1)))));
        fib_indi(i,j) = fib1+fib2;  % rotate fib_indi 90 degree counterclockwise to obtain matrix in plot
    end
end

r_x = linspace(1/(2*n1), 1-1/(2*n1), n1); 
r_y = linspace(1/(2*n2), 1-1/(2*n2), n2);
[r_y, r_x] = meshgrid(r_y, r_x); 

slope_f1 = (1-r_y)./r_x.*(fib_indi==1|fib_indi==3);
l2_norm1 = sqrt((slope_f1.^2+(ones(n1, n2)).^2));
slope_f2 = -r_y./r_x.*(fib_indi==2|fib_indi==3);
l2_norm2 = sqrt((slope_f2.^2+(ones(n1, n2)).^2));

%% weights in each voxel
w1_matrix = (fib_indi==1)+0.5.*(fib_indi==3);
w2_matrix = (fib_indi==2)+0.5.*(fib_indi==3);

%% angles in each voxel
theta_fib = zeros(n1,n2,2);
phi_fib = zeros(n1,n2,2);

for i=1:n1
    for j=1:n2   
        theta_fib(i,j,1) = atan(1/slope_f1(i,j));
        theta_fib(i,j,2) = atan(-1/slope_f2(i,j))+pi/2;
        phi_fib(i,j,1) = pi/2;
        phi_fib(i,j,2) = pi/2;
    end
end

dirac_sh_all = zeros(n1,n2,2562);
dirac_sh1_all = zeros(n1,n2,2562);
dirac_sh2_all = zeros(n1,n2,2562);

for i=1:n1
    for j=1:n2
        coe_sh1 = Dirac_SH_coe(lmax,theta_fib(i,j,1),phi_fib(i,j,1)); %% SH coefficients 
        coe_sh2 = Dirac_SH_coe(lmax,theta_fib(i,j,2),phi_fib(i,j,2));
        coe_sh = w1_matrix(i,j).*coe_sh1 + w2_matrix(i,j).*coe_sh2;
 
        dirac_sh1 = SH_matrix_plot*coe_sh1; 
        dirac_sh2 = SH_matrix_plot*coe_sh2;
        dirac_sh = w1_matrix(i,j).*dirac_sh1 + w2_matrix(i,j).*dirac_sh2;  %%SH representation
        dirac_sh_all(i,j,:) = dirac_sh; 
        dirac_sh1_all(i,j,:) = dirac_sh1; 
        dirac_sh2_all(i,j,:) = dirac_sh2; 
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%% generate dwi signals on the equal-angle grid/gradient-direction grid (J=3, 81*2 points) 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% noiseless DWI
DWI_noiseless_all = zeros(n1,n2,size(theta,2)); 
DWI_all = zeros(n1,n2,size(theta,2)); 

rng(0)
for i = 1:n1
    for j = 1:n2
        for at = 1:size(theta, 2)
            w = [w1_matrix(i,j),w2_matrix(i,j)];
            if sum(w) == 0
                DWI_noiseless_all(i,j,at) = exp(-b_response);
            else
                DWI_noiseless_all(i,j,at) = myresponse_crossing([b_response b_response],[ratio_response ratio_response],w,theta_fib(i,j,:),phi_fib(i,j,:),theta(at)*pi,phi(at)*2*pi); %%162 by 1
            end
        end
        DWI_all(i,j,:) = add_Rician_noise(DWI_noiseless_all(i,j,:),sigma);
    end
end


%%% get observed DWI on the sampling grid
DWI_noiseless=DWI_noiseless_all(:,:,sampling_grid_index);
DWI=DWI_all(:,:,sampling_grid_index);

save('ROIs.mat', 'Constraint', 'design_SN', 'DWI', 'DWI_noiseless')
%save('ROIss.mat', 'Constraint', 'design_SN', 'DWI', 'DWI_noiseless')


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Method III: classo+ADMM
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
beta_classo_all=zeros(n1,n2,size(design_SN,2), size(lambda_seq_la,2));
fod_classo_all=zeros(n1,n2,size(SH_matrix_plot,1), size(lambda_seq_la,2));
dwi_classo_all=zeros(n1,n2, n_sample,size(lambda_seq_la,2));

df_classo_all=zeros(n1,n2,size(lambda_seq_la,2));
df_rank_classo_all=zeros(n1,n2,size(lambda_seq_la,2));
RSS_classo_all=zeros(n1,n2,size(lambda_seq_la,2));
BIC_rank_classo_all=zeros(n1,n2,size(lambda_seq_la,2));
AIC_rank_classo_all=zeros(n1,n2,size(lambda_seq_la,2));

index_df_classo_BIC_rank = zeros(n1,n2,1);
index_df_classo_AIC_rank = zeros(n1,n2,1);
index_df_classo_RSSdiff = zeros(n1,n2,1);

df_classo_BIC_rank = zeros(n1,n2,1);
df_classo_AIC_rank = zeros(n1,n2,1);
df_classo_RSSdiff = zeros(n1,n2,1);

beta_classo_s_BIC_rank=zeros(n1,n2,size(design_SN,2));
beta_classo_s_AIC_rank=zeros(n1,n2,size(design_SN,2));
beta_classo_s_RSSdiff=zeros(n1,n2,size(design_SN,2));

fod_classo_s_BIC_rank=zeros(n1,n2,size(SH_matrix_plot,1));
fod_classo_s_AIC_rank=zeros(n1,n2,size(SH_matrix_plot,1));
fod_classo_s_RSSdiff=zeros(n1,n2,size(SH_matrix_plot,1));


for k1 = 1:n1
    for k2 = 1:n2
    
        z_all_C = zeros(size(design_SN,2),size(lambda_seq_la,2));
        w_all_C = zeros(size(Constraint,1),size(lambda_seq_la,2));
        beta_admm_all_C=zeros(size(design_SN,2), size(lambda_seq_la,2));
            
        dwi_admm_all_C=zeros(size(design_SN,1), size(lambda_seq_la,2));
        FOD_admm_all_C=zeros(size(SN_vertex_symm,1), size(lambda_seq_la,2));
        df_admm_C=zeros(size(lambda_seq_la,2),1);
        df_admm_rank_C=zeros(size(lambda_seq_la,2),1);

        RSS_admm_C=zeros(size(lambda_seq_la,2),1);
        BIC_admm_rank_C=zeros(size(lambda_seq_la,2),1);
        AIC_admm_rank_C=zeros(size(lambda_seq_la,2),1);

        DWI_simulated_h = reshape(DWI(k1,k2,:),numel(DWI(k1,k2,:)),1);

        SN_stop_index = lambda_length_la;   
        %% admm
        print=0;  %%print iteration number of not
        for i =1:size(lambda_seq_la,2) %% use a decreasing lambda fitting scheme to speed up: start use results from previous lamabda as initial
            if i==1 
                z=z_all_C(:,1);
                w=w_all_C(:,1);
            else
                z=z_all_C(:,i-1);
                w=w_all_C(:,i-1);
            end
            lambda_c=lambda_seq_la(1,i); %%current lambda
            X = design_SN'*design_SN+lambda_c*eye(size(design_SN,2))+lambda_c*((-Constraint)'*((-Constraint)));
            Y = design_SN'*DWI_simulated_h;
            X_inv = inv(X);
            [beta_admm_all_C(:,i), z_all_C(:,i), w_all_C(:,i)] = ADMM_classo(Y, X_inv, -Constraint, (-Constraint)', z, w,lambda_c, ep_r,ep_a, lambda_c, maxit, print);
            
            idx_temp = find(abs(z_all_C(:,i))>0); %% set cutoff manually
            df_admm_rank_C(i,1)=rank(design_SN(:,idx_temp));
            df_admm_C(i,1)=size(idx_temp,1);
            FOD_admm_all_C(:,i) = SN_vertex_symm*z_all_C(:,i);
            dwi_admm_all_C(:,i) = design_SN*z_all_C(:,i);  %%fitted dwi
            RSS_admm_C(i,1)=sum((dwi_admm_all_C(:,i)-DWI_simulated_h).^2);
            dwi_temp = dwi_admm_all_C(:,i);
            BIC_admm_rank_C(i,1)=n_sample.*log(RSS_admm_C(i,1))+df_admm_rank_C(i,1).*log(n_sample);
            AIC_admm_rank_C(i,1)=n_sample.*log(RSS_admm_C(i,1))+df_admm_C(i,1).*2;
             
            beta_classo_all(k1,k2,:,i) = z_all_C(:,i);
            df_classo_all(k1,k2,i) = df_admm_C(i,1);
            df_rank_classo_all(k1,k2,i) = df_admm_rank_C(i,1);

            RSS_classo_all(k1,k2,i) = RSS_admm_C(i,1);
            BIC_rank_classo_all(k1,k2,i) = BIC_admm_rank_C(i,1);
            AIC_rank_classo_all(k1,k2,i) = AIC_admm_rank_C(i,1);
            fod_classo_all(k1,k2,:,i) = FOD_admm_all_C(:,i);
            dwi_classo_all(k1,k2,:,i) = dwi_admm_all_C(:,i);  
            
             if(i>stop_length)
                 rela_diff_temp = diff(log10(RSS_admm_C((i-stop_length):i,1)),1)./(log10(RSS_admm_C((i-stop_length):(i-1),1))*stop_spacing);
                 
                 if(sum(abs(rela_diff_temp))<stop_percent)
                     SN_stop_index = i;
                     break;
                 end
                 indi_temp = abs(rela_diff_temp)<stop_thresh;
                 if(sum(indi_temp)==stop_length)
                     SN_stop_index = i;
                     break;
                 end
             end
             
%            if(i>stop_length)
%                rela_diff_temp = diff(log10(RSS_admm_C((i-stop_length):i,1)),1)./stop_spacing;
%                if(mean(abs(rela_diff_temp))<stop_thresh)
%                    SN_stop_index = i;
%                    break;
%                end
%            end
             display(i);
        end
        
        idx_admm_BIC_rank_C = find(BIC_admm_rank_C==min(BIC_admm_rank_C(1:SN_stop_index)));
        idx_admm_AIC_rank_C = find(AIC_admm_rank_C==min(AIC_admm_rank_C(1:SN_stop_index)));

        FOD_admm_BIC_rank_C=FOD_admm_all_C(:,idx_admm_BIC_rank_C);
        FOD_admm_BIC_rank_C_st = fod_stand(FOD_admm_BIC_rank_C);

        FOD_admm_AIC_rank_C=FOD_admm_all_C(:,idx_admm_AIC_rank_C);
        FOD_admm_AIC_rank_C_st = fod_stand(FOD_admm_AIC_rank_C);

        FOD_admm_RSSdiff_C=FOD_admm_all_C(:,SN_stop_index);
        FOD_admm_RSSdiff_C_st = fod_stand(FOD_admm_RSSdiff_C);
        
        index_df_classo_BIC_rank(k1,k2,1) = idx_admm_BIC_rank_C;
        index_df_classo_AIC_rank(k1,k2,1) = idx_admm_AIC_rank_C;
        index_df_classo_RSSdiff(k1,k2,1) = SN_stop_index;

        df_classo_BIC_rank(k1,k2,:) = df_rank_classo_all(k1,k2,idx_admm_BIC_rank_C);
        df_classo_AIC_rank(k1,k2,:) = df_rank_classo_all(k1,k2,idx_admm_AIC_rank_C);
        df_classo_RSSdiff(k1,k2,:) = df_rank_classo_all(k1,k2,SN_stop_index);

        beta_classo_s_BIC_rank(k1,k2,:) = beta_classo_all(k1,k2,:,idx_admm_BIC_rank_C);
        beta_classo_s_AIC_rank(k1,k2,:) = beta_classo_all(k1,k2,:,idx_admm_AIC_rank_C);
        beta_classo_s_RSSdiff(k1,k2,:) = beta_classo_all(k1,k2,:,SN_stop_index);

        fod_classo_s_BIC_rank(k1,k2,:) = FOD_admm_BIC_rank_C;
        fod_classo_s_AIC_rank(k1,k2,:) = FOD_admm_AIC_rank_C;
        fod_classo_s_RSSdiff(k1,k2,:) = FOD_admm_RSSdiff_C;

            
        display(k2);
    end
    display(k1);
end


figure
plot(x,f1)
hold on
plot(x,f2)
hold on 
plot(x,f3)
hold on 
plot(x,f4)
hold on
grid on
% quiver(r_y, r_x, w1_matrix.*ones(size(r_x)).*(fib_indi==1|fib_indi==3)./l2_norm1, w1_matrix.*slope_f1./l2_norm1,0.3)
% quiver(r_x, r_y, w2_matrix.*ones(size(r_x)).*(fib_indi==2|fib_indi==3)./l2_norm2, w2_matrix.*slope_f2./l2_norm2,0.3)
quiver(r_y, r_x, ones(size(r_x)).*(fib_indi==1|fib_indi==3)./l2_norm1, slope_f1./l2_norm1,0.3)
quiver(r_x, r_y, ones(size(r_x)).*(fib_indi==2|fib_indi==3)./l2_norm2, slope_f2./l2_norm2,0.3)
title('Fiber flow')
savefig(strcat(save_path,'directions.fig'));

figure
for k1=1:n1
    for k2=1:n2
        subplot(n1,n2,(n2-k2)*n1+k1)
        plot_spherical_function(v_plot,f_plot,squeeze(dirac_sh_all(k1,k2,:)),options)
        %draw_direction(theta_fib(k1,k2,:),phi_fib(k1,k2,:),0.001);
        view([1,0,0])
    end
    display(k1);
end
text(0.5, 1,'SH representation of true fiber','HorizontalAlignment','center','VerticalAlignment', 'top');
savefig(strcat(save_path,'SH_lmax',num2str(lmax8),'_b',num2str(b(1)),'_rep.fig'));

figure
for k2=1:n2
    for k1=1:n1
        subplot(n1,n2,(n2-k2)*n1+k1)
        plot_spherical_function(v_plot,f_plot,(reshape(fod_classo_s_BIC_rank(k1,k2,:),numel(fod_classo_s_BIC_rank(k1,k2,:)),1)),options)
        %draw_direction(theta_fib(k1,k2,:),phi_fib(k1,k2,:),0.001);
        view([1,0,0])
    end
    display(k2);
end
text(0.5, 1,'SN lasso BIC rank','HorizontalAlignment','center','VerticalAlignment', 'top');
savefig(strcat(save_path,'SN_lmax',num2str(lmax),'_b',num2str(b(1)),'_est_BICrank.fig'));

figure
for k2=1:n2
    for k1=1:n1
        subplot(n1,n2,(n2-k2)*n1+k1)
        plot_spherical_function(v_plot,f_plot,(reshape(fod_classo_s_AIC_rank(k1,k2,:),numel(fod_classo_s_AIC_rank(k1,k2,:)),1)),options)
        %draw_direction(theta_fib(k1,k2,:),phi_fib(k1,k2,:),0.001);
        view([1,0,0])
    end
    display(k2);
end
text(0.5, 1,'SN lasso AIC rank','HorizontalAlignment','center','VerticalAlignment', 'top');
savefig(strcat(save_path,'SN_lmax',num2str(lmax),'_b',num2str(b(1)),'_est_AICrank.fig'));

figure
for k1=1:n1
    for k2=1:n2
        subplot(n1,n2,(n2-k2)*n1+k1)
        plot_spherical_function(v_plot,f_plot,(reshape(fod_classo_s_RSSdiff(k1,k2,:),numel(fod_classo_s_RSSdiff(k1,k2,:)),1)),options)
        %draw_direction(theta_fib(k1,k2,:),phi_fib(k1,k2,:),0.001);
        view([1,0,0])
    end
    display(k1);
end
text(0.5, 1,'SN lasso RSSdiff','HorizontalAlignment','center','VerticalAlignment', 'top');
savefig(strcat(save_path,'SN_lmax',num2str(lmax),'_b',num2str(b(1)),'_est_RSSdiff.fig'));


nfib_SN_BIC_rank = zeros(n1,n2);
nfib_SN_AIC_rank = zeros(n1,n2);
nfib_SN_RSSdiff = zeros(n1,n2);

peak_thresh = 0.15;
Dis = squareform(pdist(pos_plot','cosine'));

for i=1:n1
    for j=1:n2
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%
        kmin = 40;
        cut_thresh = peak_thresh;
        [~, ~, ~, ~, ~, peak_pos_SN_final_BIC_rank] = FOD_peak(fod_classo_s_BIC_rank(i,j,:), Dis, kmin, cut_thresh, pos_plot, theta_p, phi_p);
        [~, ~, ~, ~, ~, peak_pos_SN_final_AIC_rank] = FOD_peak(fod_classo_s_AIC_rank(i,j,:), Dis, kmin, cut_thresh, pos_plot, theta_p, phi_p);
        [~, ~, ~, ~, ~, peak_pos_SN_final_RSSdiff] = FOD_peak(fod_classo_s_RSSdiff(i,j,:), Dis, kmin, cut_thresh, pos_plot, theta_p, phi_p);

        nfib_SN_BIC_rank(i,j) = size(peak_pos_SN_final_BIC_rank,2);
        nfib_SN_AIC_rank(i,j) = size(peak_pos_SN_final_AIC_rank,2);
        nfib_SN_RSSdiff(i,j) = size(peak_pos_SN_final_RSSdiff,2);
   
    end
    display(i);
end

nfib = (fib_indi>2);
nfib = nfib+1;

n_1fib = sum(sum(nfib==1))
n_2fib = sum(sum(nfib==2))

SN_BIC_rank_1fib_over = sum(sum(nfib_SN_BIC_rank(nfib==1)>1))
SN_BIC_rank_1fib_correct = sum(sum(nfib_SN_BIC_rank(nfib==1)==1))
SN_BIC_rank_1fib_under = sum(sum(nfib_SN_BIC_rank(nfib==1)<1))

SN_BIC_rank_2fib_over = sum(sum(nfib_SN_BIC_rank(nfib==2)>2))
SN_BIC_rank_2fib_correct = sum(sum(nfib_SN_BIC_rank(nfib==2)==2))
SN_BIC_rank_2fib_under = sum(sum(nfib_SN_BIC_rank(nfib==2)<2))

SN_AIC_rank_1fib_over = sum(sum(nfib_SN_AIC_rank(nfib==1)>1))
SN_AIC_rank_1fib_correct = sum(sum(nfib_SN_AIC_rank(nfib==1)==1))
SN_AIC_rank_1fib_under = sum(sum(nfib_SN_AIC_rank(nfib==1)<1))

SN_AIC_rank_2fib_over = sum(sum(nfib_SN_AIC_rank(nfib==2)>2))
SN_AIC_rank_2fib_correct = sum(sum(nfib_SN_AIC_rank(nfib==2)==2))
SN_AIC_rank_2fib_under = sum(sum(nfib_SN_AIC_rank(nfib==2)<2))

SN_RSSdiff_1fib_over = sum(sum(nfib_SN_RSSdiff(nfib==1)>1))
SN_RSSdiff_1fib_correct = sum(sum(nfib_SN_RSSdiff(nfib==1)==1))
SN_RSSdiff_1fib_under = sum(sum(nfib_SN_RSSdiff(nfib==1)<1))

SN_RSSdiff_2fib_over = sum(sum(nfib_SN_RSSdiff(nfib==2)>2))
SN_RSSdiff_2fib_correct = sum(sum(nfib_SN_RSSdiff(nfib==2)==2))
SN_RSSdiff_2fib_under = sum(sum(nfib_SN_RSSdiff(nfib==2)<2))


SN_BIC_rank_1fib_over_rate = sum(sum(nfib_SN_BIC_rank(nfib==1)>1))/n_1fib
SN_BIC_rank_1fib_correct_rate = sum(sum(nfib_SN_BIC_rank(nfib==1)==1))/n_1fib
SN_BIC_rank_1fib_under_rate = sum(sum(nfib_SN_BIC_rank(nfib==1)<1))/n_1fib

SN_BIC_rank_2fib_over_rate = sum(sum(nfib_SN_BIC_rank(nfib==2)>2))/n_2fib
SN_BIC_rank_2fib_correct_rate = sum(sum(nfib_SN_BIC_rank(nfib==2)==2))/n_2fib
SN_BIC_rank_2fib_under_rate = sum(sum(nfib_SN_BIC_rank(nfib==2)<2))/n_2fib

SN_AIC_rank_1fib_over_rate = sum(sum(nfib_SN_AIC_rank(nfib==1)>1))/n_1fib
SN_AIC_rank_1fib_correct_rate = sum(sum(nfib_SN_AIC_rank(nfib==1)==1))/n_1fib
SN_AIC_rank_1fib_under_rate = sum(sum(nfib_SN_AIC_rank(nfib==1)<1))/n_1fib

SN_AIC_rank_2fib_over_rate = sum(sum(nfib_SN_AIC_rank(nfib==2)>2))/n_2fib
SN_AIC_rank_2fib_correct_rate = sum(sum(nfib_SN_AIC_rank(nfib==2)==2))/n_2fib
SN_AIC_rank_2fib_under_rate = sum(sum(nfib_SN_AIC_rank(nfib==2)<2))/n_2fib

SN_RSSdiff_1fib_over_rate = sum(sum(nfib_SN_RSSdiff(nfib==1)>1))/n_1fib
SN_RSSdiff_1fib_correct_rate = sum(sum(nfib_SN_RSSdiff(nfib==1)==1))/n_1fib
SN_RSSdiff_1fib_under_rate = sum(sum(nfib_SN_RSSdiff(nfib==1)<1))/n_1fib

SN_RSSdiff_2fib_over_rate = sum(sum(nfib_SN_RSSdiff(nfib==2)>2))/n_2fib
SN_RSSdiff_2fib_correct_rate = sum(sum(nfib_SN_RSSdiff(nfib==2)==2))/n_2fib
SN_RSSdiff_2fib_under_rate = sum(sum(nfib_SN_RSSdiff(nfib==2)<2))/n_2fib