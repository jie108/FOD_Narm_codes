clear;

%% set path
path = '/Users/jlyang/Documents/FOD/DMRI_code/';
% add path of the wavelet package and wavelet_meshes to generate equi-angle grid 
addpath(strcat(path, 'toolbox_wavelet_meshes'));   
addpath(strcat(path, 'toolbox_wavelet_meshes/toolbox'));
% add path of npy-matlab to read python npy file into matlab
addpath(strcat(path, 'npy-matlab'));

% specify and add path of FOD estimation results
path_fod = 'Results_new/ROI/';
addpath(strcat(path, path_fod));

%% load FOD estimation results
unzip_npz = unzip('ROI.npz');
dirac_plot = readNPY(cell2mat(unzip_npz(7)));
% load FOD estimation result (using noiseless DWI)
fod_all_no = readNPY('fod_all_no.npy');
% load FOD estimation result (including voxel-wise and spacially-smoothed)
fod_all = readNPY('fod_all_a15_b2_stop.npy');
% specify save name
save_name = 'ROI_voxel';

%% specify parameters for plotting
J_plot = 5;
options.base_mesh = 'ico';
options.relaxation = 1;
options.keep_subdivision = 1;
[v_plot, f_plot] = compute_semiregular_sphere(J_plot, options);

% plotting options
options.spherical = 1;
options.use_color = 1;
options.color = 'wavelets';
options.use_elevation = 2;
options.rho = 0.5;
options.scaling = 1.5;

%% plot of estimated FOD on 2D simulation region
[n1, n2, ~] = size(dirac_plot);
h = figure;
for k1 = 1:n1
    for k2 = 1:n2
        subplot(n1, n2, (n2-k2)*n1+k1)
        %plot_spherical_function(v_plot, f_plot, squeeze(dirac_sh_all(k1,k2,:)+1e-16), options) %True FODs
        %plot_spherical_function(v_plot, f_plot, squeeze(fod_all_no(k1,k2,:)), options) %Noiseless estimates
        plot_spherical_function(v_plot, f_plot, squeeze(fod_all(k1,k2,1,:)), options) %Voxel-wise estimates
        %plot_spherical_function(v_plot, f_plot, squeeze(fod_all(k1,k2,11,:)), options) %Spacially-smoothed estimates
        view([1, 0, 0])
    end
    disp(k1);
end
set(h,'Units','Inches');
pos = get(h,'Position');
set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)]);
saveas(gcf, strcat(path, path_fod, sprintf('%s.pdf', save_name)))
close(gcf);

%% plot of estimated FOD on 3D simulation region
[n1, n2, n3, ~] = size(dirac_plot);
for k3 = 1:n3
    h = figure;
    for k1 = 1:n1
        for k2 = 1:n2
            subplot(n1, n2, (n2-k2)*n1+k1)
            %plot_spherical_function(v_plot, f_plot, squeeze(dirac_sh_all(k1,k2,k3,:)+1e-16), options) %True FODs
            %plot_spherical_function(v_plot, f_plot, squeeze(fod_all_no(k1,k2,k3,:)), options) %Noiseless estimates
            plot_spherical_function(v_plot, f_plot, squeeze(fod_all(k1,k2,k3,1,:)), options) %Voxel-wise estimates
            %plot_spherical_function(v_plot, f_plot, squeeze(fod_all(k1,k2,k3,7,:)), options) %Spacially-smoothed estimates
            lightangle(30,-30)
            view([0,0,-1])
        end
        disp(k1);
    end
    set(h,'Units','Inches');
    pos = get(h,'Position');
    set(h,'PaperPositionMode','Auto','PaperUnits','Inches','PaperSize',[pos(3), pos(4)]);
    saveas(gcf, strcat(path, path_fod, sprintf('%s_ud_%d.pdf', save_name, k3)))
    close(gcf);
end