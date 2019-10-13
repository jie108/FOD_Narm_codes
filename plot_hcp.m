%% Example real data
%% First you need to add the pathes that contains all the needed functions

clear
path = '/Users/jlyang/Documents/FOD/DMRI_code/';

addpath(path);
addpath(strcat(path,'real_data/103818/T1w/Diffusion_3K'));
addpath(strcat(path,'toolbox_wavelet_meshes'));
addpath(strcat(path,'toolbox_wavelet_meshes/toolbox/'));
addpath(strcat(path,'NIfTI/'));

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% real data range
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% ROI1
x_range = 56:75;
y_range = 86:105;
z_range = 66:85;
% ROI2
%x_range = 54:68;
%y_range = 118:132;
%z_range = 61:75;
% ROI3
%x_range = 60:74;
%y_range = 91:105;
%z_range = 71:85;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% load .nii data
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% load nii data (raw)
bvec_raw = load('bvecs'); 
bval_raw = load('bvals');
nii_data = load_nii('data.nii.gz');

%% load fsl processed data
nii_FA = load_nii('dti_FA.nii.gz');
nii_MD = load_nii('dti_MD.nii.gz');
nii_S0 = load_nii('dti_S0.nii.gz');

nii_V1 = load_nii('dti_V1.nii.gz');
nii_V2 = load_nii('dti_V2.nii.gz');
nii_V3 = load_nii('dti_V3.nii.gz');

nii_L1 = load_nii('dti_L1.nii.gz');
nii_L2 = load_nii('dti_L2.nii.gz');
nii_L3 = load_nii('dti_L3.nii.gz');

%% transfer to matrix
img_data_raw = nii_data.img;
size(img_data_raw)

img_FA_fsl_all = nii_FA.img;
img_MD_fsl_all = nii_MD.img;
img_S0_fsl_all = nii_S0.img;

img_V1_fsl_all = nii_V1.img;
img_V2_fsl_all = nii_V2.img;
img_V3_fsl_all = nii_V3.img;

img_L1_fsl_all = nii_L1.img;
img_L2_fsl_all = nii_L2.img;
img_L3_fsl_all = nii_L3.img;

bval_raw(abs(bval_raw)<100) = 0;
bval_raw(abs(bval_raw-1000)<100) = 1000;
bval_raw(abs(bval_raw-2000)<100) = 2000;
bval_raw(abs(bval_raw-3000)<100) = 3000;

b_sample = find(abs(bval_raw)~=0);
img_data_all = img_data_raw(:,:,:,b_sample);
b0_sample = find(abs(bval_raw)==0);
img_b0_all = img_data_raw(:,:,:,b0_sample);

bvec = bvec_raw(:,b_sample);
bval = bval_raw(:,b_sample);

clearvars nii_data nii_FA nii_MD nii_MO nii_S0 nii_V1 nii_V2 nii_V3 nii_L1 nii_L2 nii_L3;


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% real data range
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

img_data = img_data_all(x_range,y_range,z_range,:);
img_b0 = img_b0_all(x_range,y_range,z_range,:);

img_S0_fsl = img_S0_fsl_all(x_range,y_range,z_range);
img_MD_fsl = img_MD_fsl_all(x_range,y_range,z_range);
img_FA_fsl = img_FA_fsl_all(x_range,y_range,z_range);
img_V1_fsl = img_V1_fsl_all(x_range,y_range,z_range,:);
img_L1_fsl = img_L1_fsl_all(x_range,y_range,z_range,:);
img_L2_fsl = img_L2_fsl_all(x_range,y_range,z_range,:);
img_L3_fsl = img_L3_fsl_all(x_range,y_range,z_range,:);

[n1, n2, n3, n4] = size(img_data);


%% MLE estimation of S0 and sigma in each voxel

Sigma_mle = zeros(n1,n2,n3);
S0_mle = zeros(n1,n2,n3);
for k1=1:n1
    for k2=1:n2
        for k3=1:n3
            y = squeeze(img_b0(k1,k2,k3,:));
            if(min(y)>0&&max(abs(y))~=Inf)
                Sigma_mle(k1,k2,k3) = sqrt(var(y));
                S0_mle(k1,k2,k3) = mean(y);
            end
        end
    end
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Single tensor model
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
N = n1*n2*n3;
S_min = min(min(img_data(img_data>0)));
S_max = max(max(img_data(img_data<Inf)));

S_data = zeros(n4,N);
for i = 1:n4
    S_data(i,:) = reshape(img_data(:,:,:,i),1,N);
    S_data(i,:) = S_data(i,:)./reshape(S0_mle,1,N);  %% also can use img_S0_fsl
end

index_negative = [];
index_infinity = [];
for i=1:N
    if(min(S_data(:,i))<0)
        index_negative = [index_negative i];
    end
    if(sum(abs(S_data(:,i)==Inf))>0)
        index_infinity = [index_infinity i];
    end
end
index_temp = setdiff(1:N,union(index_negative,index_infinity));
S_data(S_data<=0) = min(min(min(min(S_data(S_data>0)))));
S_data(S_data==Inf) = max(max(max(max(S_data(S_data<Inf)))));

X = zeros(n4,6);
X(:,1) = bvec(1,:).^2;
X(:,2) = bvec(2,:).^2;
X(:,3) = bvec(3,:).^2;
X(:,4) = 2*bvec(1,:).*bvec(2,:);
X(:,5) = 2*bvec(1,:).*bvec(3,:);
X(:,6) = 2*bvec(2,:).*bvec(3,:);
X = diag(bval)*X;

FA_temp = zeros(1,N);
eval_temp = zeros(3,N);
evec_temp = zeros(3,3,N);

%% single tensor model
for i=1:N
    l_S = log(S_data(:,i));
    D_est_temp = -inv((X'*X))*X'*l_S;
    [D_nl, iter, DWI_est] = LM_dti(X,S_data(:,i),D_est_temp,1e-10);
    D_matrix_nl = [D_nl(1) D_nl(4) D_nl(5);D_nl(4) D_nl(2) D_nl(6);D_nl(5) D_nl(6) D_nl(3)];
    if(sum(isnan(D_matrix_nl))==0)
      [egvec_nl, egval_nl] = eig(D_matrix_nl);
    end
    eval_temp(:,i) = diag(egval_nl);
    evec_temp(:,:,i) = egvec_nl;
    FA_temp(i) = sqrt(1/2)*sqrt(((eval_temp(1,i)-eval_temp(2,i))^2+(eval_temp(1,i)-eval_temp(3,i))^2+(eval_temp(3,i)-eval_temp(2,i))^2))/sqrt((eval_temp(1,i)^2+eval_temp(2,i)^2+eval_temp(3,i)^2)); 
end


MD_temp = sum(eval_temp,1)./3;
eval_ratio23_temp = eval_temp(2,:)./eval_temp(1,:);
eval_ratio_temp = eval_temp(3,:)*2./(eval_temp(1,:)+eval_temp(2,:));
index_ttemp = find(FA_temp<1&MD_temp>=0);
data_FA = reshape(FA_temp,n1,n2,n3);
data_MD = reshape(MD_temp,n1,n2,n3);

figure % compare single tensor model estimation vs fsl estimation
subplot(2,1,1)
scatter(FA_temp, reshape(img_FA_fsl,1,numel(img_FA_fsl)))
subplot(2,1,2)
scatter(MD_temp, reshape(img_MD_fsl,1,numel(img_MD_fsl)))


%% set uniform color range for heat map later
FA_temp_restrict = min(1,FA_temp);
MD_temp_restrict = max(0,MD_temp);
FA_top = max(FA_temp_restrict);
FA_bottom = min(FA_temp_restrict);
MD_top = max(MD_temp_restrict);
MD_bottom = min(MD_temp_restrict);

% for colormap
Eig1_FAcorrected = zeros(n1,n2,n3,3);
Eig1_fsl_FAcorrected = zeros(n1,n2,n3,3);

for i = 1:N
    [i1,i2,i3] = ind2sub([n1,n2,n3],i);
    Eig1_FAcorrected(i1,i2,i3,:) = abs(evec_temp(:,3,i)).*FA_temp(i);
    Eig1_fsl_FAcorrected(i1,i2,i3,:) = abs(img_V1_fsl(i1,i2,i3,:)).*img_FA_fsl(i1,i2,i3);
end

%%%%%%%%%%%%%%%%%%% FA, MD and color maps
%% plot est.FA, MD color map

fod_path = 'Results_new/ROI_HCP/';

for i3 = 1:n3

    %% FA map
    FA_temp_map = dti_data_reindex(data_FA(:,:,i3));
    imagesc(FA_temp_map);
    colormap('gray');
    caxis manual 
    caxis([FA_bottom FA_top]);
    colorbar
    set(gcf,'units','points','position',[0,0,500,500])
    set(gcf,'Units','Inches');
    pos = get(gcf,'Position');
    set(gcf,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)]);
    saveas(gcf, sprintf('%s%sFA_est3K_%d.pdf', path, fod_path, i3))
    close(gcf)

    %% MD map
    MD_temp_map = dti_data_reindex(data_MD(:,:,i3)*1e3);
    imagesc(MD_temp_map);
    colormap('gray');
    caxis manual 
    caxis([MD_bottom*1e3 MD_top*1e3]);
    colorbar
    set(gcf,'units','points','position',[0,0,500,500])
    set(gcf,'Units','Inches');
    pos = get(gcf,'Position');
    set(gcf,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)]);
    saveas(gcf, sprintf('%s%sMD_est3K_%d.pdf', path, fod_path, i3))
    close(gcf)

    %% Color leading eigenvector map
    Eig1_FAcorrected_temp = dti_data_reindex(squeeze(Eig1_FAcorrected(:,:,i3,:)));
    imagesc(Eig1_FAcorrected_temp);
    set(gcf,'units','points','position',[0,0,500,500])
    set(gcf,'Units','Inches');
    pos = get(gcf,'Position');
    set(gcf,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)]);
    saveas(gcf, sprintf('%s%scolor_est3K_%d.pdf', path, fod_path, i3))
    close(gcf)

end

clear options

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% plot setting
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
options.base_mesh = 'ico';
options.relaxation = 1;
options.keep_subdivision = 1;

[v_p,f_p] = compute_semiregular_sphere(5,options);
pos_p = v_p{end};

%%%plotting options
options.spherical = 1;
% options for the display
options.use_color = 1;
options.color = 'wavelets';
options.use_elevation = 2;
options.rho = 0.5;
options.scaling = 1.5;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%% SN-lasso
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fod_path = 'Results_new/ROI_HCP/';
fod_name = '123K_3e3_b4';
fod_smooth = 'sm6';
fod_all = readNPY(strcat(path,fod_path,'fod_all_', fod_name, '.npy'));
if strcmp(fod_smooth, 'voxel')
    fod_sele_SN_all = squeeze(fod_all(:,:,:,1,:));
elseif strcmp(fod_smooth, 'sm')
    fod_sele_SN_all = squeeze(fod_all(:,:,:,11,:));
elseif strcmp(fod_smooth, 'sm8')
    fod_sele_SN_all = squeeze(fod_all(:,:,:,9,:));
elseif strcmp(fod_smooth, 'sm6')
    fod_sele_SN_all = squeeze(fod_all(:,:,:,7,:));
end

clearvars fod_all img_data img_data_raw img_data_all img_b0 img_b0_all 
clearvars img_L1_fsl_all img_L2_fsl_all img_L3_fsl_all img_V1_fsl_all img_V2_fsl_all img_V3_fsl_all 
clearvars img_S0_fsl img_FA_fsl img_MD_fsl S_data

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Plot estimated FOD (takes around 10 min for 15 by 15 slice), 
%% 4 methods (5 if plot relaxed stopping SN result), n3 (or n1) slices
%% usually SN and SN relaxed stopping resutls are similar
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

oax_left = 0;
oax_bottom = 0;
oax_width = 1;
oax_height = 1;

ax_left = 0.05;
ax_bottom = 0.05;
ax_width = 0.9;
ax_height = 0.9;

width_space = ax_width/n1;
height_space = ax_height/n2;

            
%% plot est. FOD on top of FA color map

for i3 = 1:n3

    FA_temp_map = dti_data_reindex(squeeze(data_FA(:,:,i3)));
    %% plot est. FOD on FA background inversely scaled with MD value
    FA_fig = figure('units','normalized','position',[0 0 0.6 1]);
    axis_FA_fig = axes;
    colormap(axis_FA_fig, gray);
    imagesc(FA_temp_map);
    caxis(axis_FA_fig,[FA_bottom FA_top]);
    ax = gca;  
    ax.Units = 'normalized';
    ax.OuterPosition = [oax_left, oax_bottom, oax_width, oax_height];
    ax.Position = [ax_left, ax_bottom, ax_width, ax_height];
    axis(axis_FA_fig,'off')

    hold on;
    for i1 = 1:n1
        for i2 = 1:n2
            if(data_MD(i1,i2,i3)<1.5)
                fig_factor = 0.2;
            else
                fig_factor = 0.2+(data_MD(i1,i2,i3)-1.5)/MD_top;
            end

            ax_temp_coordinate = [ax_left+(i1-1)*width_space+width_space*(fig_factor)/2, ax_bottom+(i2-1)*height_space+height_space*(fig_factor)/2, width_space*(1-fig_factor), height_space*(1-fig_factor)];
            ax_temp = axes('Position', ax_temp_coordinate);
            axis(ax_temp,'off');
            colormap(ax_temp,jet);
            options.use_axis = ax_temp_coordinate;
            plot_spherical_function(v_p,f_p,squeeze(fod_sele_SN_all(i1,i2,i3,:)),options);
            lightangle(pi,0)
            view([0 0 -1])
        end
        disp(i1)
    end
    hold off;
    set(FA_fig,'Units','Inches');
    %pos = get(FA_fig,'Position');
    %set(FA_fig,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)]);
    set(FA_fig,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[11.5, 10]);
    saveas(gcf, sprintf('%s%sROI_HCP_%s_%s_%d.pdf', path, fod_path, fod_name, fod_smooth, i3))     
    close(gcf);
    display(i3);
end


%% plot subregion of est. FOD on top of FA color map
% ROI1
%i3 = 17;
%x_sub = 7:11;
%y_sub = 9:13;
% ROI2
i3 = 11;
x_sub = 10:15;
y_sub = 1:6;

width_space_sub = ax_width/length(x_sub);
height_space_sub = ax_height/length(y_sub);

FA_temp_map = dti_data_reindex(squeeze(data_FA(x_sub,y_sub,i3)));
%% plot est. FOD on FA background inversely scaled with MD value
FA_fig = figure('units','normalized','position',[0 0 0.6 1]);
axis_FA_fig = axes;
colormap(axis_FA_fig, gray);
imagesc(FA_temp_map);
caxis(axis_FA_fig,[FA_bottom FA_top]);
ax = gca;  
ax.Units = 'normalized';
ax.OuterPosition = [oax_left, oax_bottom, oax_width, oax_height];
ax.Position = [ax_left, ax_bottom, ax_width, ax_height];
axis(axis_FA_fig,'off')

hold on;
for i1 = 1:length(x_sub)
    for i2 = 1:length(y_sub)
        if(data_MD(x_sub(i1),y_sub(i2),i3)<1.5)
            fig_factor = 0.2;
        else
            fig_factor = 0.2+(data_MD(x_sub(i1),y_sub(i2),i3)-1.5)/MD_top;
        end

        ax_temp_coordinate = [ax_left+(i1-1)*width_space_sub+width_space_sub*(fig_factor)/2, ax_bottom+(i2-1)*height_space_sub+height_space_sub*(fig_factor)/2, width_space_sub*(1-fig_factor), height_space_sub*(1-fig_factor)];
        ax_temp = axes('Position', ax_temp_coordinate);
        axis(ax_temp,'off');
        colormap(ax_temp,jet);
        options.use_axis = ax_temp_coordinate;
        plot_spherical_function(v_p,f_p,squeeze(fod_sele_SN_all(x_sub(i1),y_sub(i2),i3,:)),options);
        lightangle(pi,0)
        view([0 0 -1])
    end
    disp(i1)
end
hold off;
set(FA_fig,'Units','Inches');
%pos = get(FA_fig,'Position');
%set(FA_fig,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)]);
set(FA_fig,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[11.5, 10]);
saveas(gcf, sprintf('%s%sROI_HCP_%s_%s_%d_sub1.pdf', path, fod_path, fod_name, fod_smooth, i3))     
close(gcf);