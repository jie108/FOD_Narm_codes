function fod_st = fod_stand(fod)
%% parameter: fod: an estimated fod 
%% return: standardized fod such that it is nonnegative and the summation
%% over the grid points equals to 1

fod_st = fod;
fod_st(fod < 0) = 0;
fod_st = fod_st./sum(fod_st);
