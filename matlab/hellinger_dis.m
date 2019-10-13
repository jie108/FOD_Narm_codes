%% function to compute Hellinger distance between two density functions

function hellinger_dis = hellinger_dis(pdf1, pdf2)
    
hellinger_dis = sqrt(0.5 * sum((sqrt(pdf1) - sqrt(pdf2)).^2));
hellinger_dis = max(hellinger_dis, 1e-4);