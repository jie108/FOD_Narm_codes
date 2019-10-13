function [gamma, sn_stop_index] = sn_classo(DWI, design_matrix, constraint, ...
    lambda_seq, stop_percent, stop_thresh, epi_abs, epi_rel, max_iter, print)

[N, p] = size(constraint);
L = length(lambda_seq);

gamma = zeros(p, 1);
eta = zeros(N, 1);
u = zeros(p, 1);
t = zeros(N, 1);
RSS = zeros(L, 1);

stop_length = floor(stop_percent * length(lambda_seq));   
stop_spacing = log10(lambda_seq(2)) - log10(lambda_seq(1));
        
for i = 1:L %% use a decreasing lambda fitting scheme to speed up
            
    lambda_c = lambda_seq(i); %%current lambda
    [gamma, eta, u, t] = classo(DWI, design_matrix, ...
        constraint, gamma, eta, u, t, lambda_c, lambda_c, epi_abs, epi_rel, max_iter);
    RSS(i) = sum((design_matrix * gamma - DWI).^2);
            
    if(i > stop_length)
        rela_diff_temp = diff(log10(RSS((i-stop_length):i)))./stop_spacing;
        sn_stop_index = i;
        if(mean(abs(rela_diff_temp)) < stop_thresh)
            break;
        end
    end
    
    if(print)
        sprintf('i = %d, RSS = %f', i, RSS(i))
    end
end