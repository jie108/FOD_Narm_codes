function [gamma, eta, u, t] = classo(y, X, C, gamma, eta, u, t, rho, lambda, epi_abs, epi_rel, max_iter)

% y is n by 1
% X is n by p
% C is l by p
% gamma is p by 1
% eta is l by 1
% u is p by 1
% t is l by 1

[l, p] = size(C);
alpha = 1.5;

U = chol(X' * X + rho * (eye(p) + C' * C));
Xy = X'*y;

for k = 1:max_iter
    
    beta = U \ (U' \ (Xy + rho * (gamma - u + C' * (eta - t))));
    
    gamma_prev = gamma;
    eta_prev = eta;
    temp = alpha * beta + (1-alpha) * gamma + u;
    gamma = max(0, 1 - lambda ./ (rho * abs(temp))) .* temp;
    u = temp - gamma;
    
    temp = C * beta;
    temp_1 = alpha * temp + (1-alpha) * eta;
    eta = max(0, temp_1 + t);
    t = t + temp_1 - eta;
    
    r = sqrt(norm(beta - gamma)^2 + norm(temp - eta)^2);
    s = rho * norm(gamma - gamma_prev + C' * (eta - eta_prev));
    epi_pri_1 = sqrt(norm(beta)^2 + norm(temp)^2);
    epi_pri_2 = sqrt(norm(gamma)^2 + norm(eta)^2);
    epi_pri = sqrt(p+l) * epi_abs + epi_rel * max(epi_pri_1, epi_pri_2);
    epi_dual = sqrt(p) * epi_abs + epi_rel * rho * norm(u + C' * t);
    
    if r < epi_pri && s < epi_dual
        break;
    end
end