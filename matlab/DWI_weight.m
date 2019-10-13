function DWI_weighted = DWI_weight(DWI, h, a, fod_prev, scaling, q)

[n1, n2, n] = size(DWI);
DWI_weighted = zeros(n1, n2, n);

for k1 = 1:n1
    for k2 = 1:n2
        fod_prev(k1, k2, :) = fod_stand(squeeze(fod_prev(k1, k2, :)));
    end
end

if scaling
    weight_min = zeros(n1, n2);
    for k1 = 1:n1
        for k2 = 1:n2
            weight_cand = [];
            if k2 < n2
                weight_cand = [weight_cand, hellinger_dis(squeeze(fod_prev(k1, k2, :)), squeeze(fod_prev(k1, k2+1, :)))];
                %weight_cand = [weight_cand, js_dis(squeeze(fod_prev(k1, k2, :)), squeeze(fod_prev(k1, k2+1, :)))];
            end
            if k1 < n1
                weight_cand = [weight_cand, hellinger_dis(squeeze(fod_prev(k1, k2, :)), squeeze(fod_prev(k1+1, k2, :)))];
                %weight_cand = [weight_cand, js_dis(squeeze(fod_prev(k1, k2, :)), squeeze(fod_prev(k1+1, k2, :)))];
            end
            if k2 > 1
                weight_cand = [weight_cand, hellinger_dis(squeeze(fod_prev(k1, k2, :)), squeeze(fod_prev(k1, k2-1, :)))];
               %weight_cand = [weight_cand, js_dis(squeeze(fod_prev(k1, k2, :)), squeeze(fod_prev(k1, k2-1, :)))];
            end
            if k1 > 1
               weight_cand = [weight_cand, hellinger_dis(squeeze(fod_prev(k1, k2, :)), squeeze(fod_prev(k1-1, k2, :)))];
               %weight_cand = [weight_cand, js_dis(squeeze(fod_prev(k1, k2, :)), squeeze(fod_prev(k1-1, k2, :)))];
            end
            weight_min(k1, k2) = min(weight_cand); %minimum nearest-neighbor hellinger distance
        end
    end
    %smoothness assumption: MNNHD is small for each voxel
    %for those 100(1-q) percent voxels with large MNNHD, they are not well
    %smoothed; when calculating Kst, the hellinger distance corresponding to
    %MNNHD is decreased to 100q percentile of MNNHDs, and all the other
    %hellinger distances are decreased in proportion
    %for those 100q percent voxels with small MNNHD, they are smoothed enough; 
    %when calculating Kst, the hellinger distance corresponding to
    %MNNHD is increased to 100q percentile of MNNHDs, and all the other
    %hellinger distances are increased in proportion
    %weight_ratio_1 = max(weight_min / quantile(reshape(weight_min, n1*n2, 1), q), 1);
    %weight_ratio_2 = min(weight_min / quantile(reshape(weight_min, n1*n2, 1), q), 1);
    %weight_ratio = weight_ratio_1 .* weight_ratio_2; %same as the next line but is more flexible
    weight_ratio = weight_min / quantile(reshape(weight_min, n1*n2, 1), q);
else
    weight_ratio = ones(n1, n2); %no scaling on hellinger distance
end


for k1 = 1:n1
    for k2 = 1:n2
        weight_loc = zeros(n1, n2);
        weight_st = zeros(n1, n2);
        for l1 = max(1, ceil(k1-h)):min(n1, floor(k1+h))
            for l2 = max(1, ceil(k2-h)):min(n2, floor(k2+h))
                weight_loc(l1, l2) = max(0, 1 - ((k1-l1)^2+(k2-l2)^2) / h^2); %quadratic kernel for location
                %weight_loc(l1, l2) = max(0, 1 - sqrt((k1-l1)^2+(k2-l2)^2) / h); %linear kernel for location
                weight_st(l1, l2) = exp(-a / weight_ratio(k1, k2)^2 * hellinger_dis(squeeze(fod_prev(k1, k2, :)), squeeze(fod_prev(l1, l2, :)))^2);
                %weight_st(l1, l2) = exp(-a / weight_ratio(k1, k2)^2 * js_dis(squeeze(fod_prev(k1, k2, :)), squeeze(fod_prev(l1, l2, :)))^2);
            end
        end
        weight_st(k1, k2) = 1;
        weight = weight_loc .* weight_st;
        weight = weight / sum(sum(weight));
        for l1 = max(1, ceil(k1-h)):min(n1, floor(k1+h))
            for l2 = max(1, ceil(k2-h)):min(n2, floor(k2+h))
                DWI_weighted(k1, k2, :) = DWI_weighted(k1, k2, :) + weight(l1, l2) * DWI(l1, l2, :);
            end
        end
    end
end