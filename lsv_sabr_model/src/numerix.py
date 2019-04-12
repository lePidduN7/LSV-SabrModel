import math
import numba as nb
from numba import cfunc, carray, types, vectorize, guvectorize, njit, float64
import numpy as np
from scipy import LowLevelCallable
from scipy.integrate import quad

@njit
def lv_sigma(u, beta_low, beta_high, eps, eps_s, f_low, f_high, lambda_, Psi, K_h, F0):
    """
    Argument of the Lamperti's transform that return the local displacement
    of the stochastic volatility function.
    """
    sigma_argument = min(max(u, f_low), f_high)
    heta_1 = eps * math.log(1 + math.exp(sigma_argument/eps))
    heta_2 = eps_s * math.log(1 + math.exp((sigma_argument - K_h)/eps_s))
    beta_factor = math.tanh(lambda_*(sigma_argument - F0))
    beta = 0.5*(beta_high + beta_low) - 0.5*(beta_high - beta_low)*beta_factor
    return 1.0/(math.pow(heta_1, beta) + Psi*heta_2)

@njit
def lv_sigma_vectorized(u_array, beta_low, beta_high, eps, eps_s, f_low, f_high, lambda_, Psi, K_h, F0):
    sigma_output = np.zeros(len(u_array))
    for i, u in enumerate(u_array):
        sigma_output[i] = lv_sigma(u, beta_low, beta_high, eps, eps_s, f_low, f_high, lambda_, Psi, K_h, F0)
    return sigma_output


lv_sigma_csignature = types.double(types.intc, types.CPointer(types.double))
@cfunc(lv_sigma_csignature, nopython=True)
def lv_sigma_cfunc(n, xx):
    in_array = carray(xx, (n, ))
    u, beta_low, beta_high, eps, eps_s, f_low, f_high, lambda_, Psi, K_h, F0 = in_array
    sigma_argument = min(max(u, f_low), f_high)
    heta_1 = eps * math.log(1 + math.exp(sigma_argument/eps))
    heta_2 = eps_s * math.log(1 + math.exp((sigma_argument - K_h)/eps_s))
    beta_factor = math.tanh(lambda_*(sigma_argument - F0))
    beta = 0.5*(beta_high + beta_low) - 0.5*(beta_high - beta_low)*beta_factor
    return 1.0/(math.pow(heta_1, beta) + Psi*heta_2)

lv_sigma_integrand_scipy = LowLevelCallable(lv_sigma_cfunc.ctypes)


@njit
def lamperti_transform_lv_sigma(upper_bound, lower_bound, beta_low, beta_high, eps, eps_s, f_low, f_high, lambda_, Psi, K_h, F0):
    u_points = np.array(
        [-0.99930504, -0.99634012, -0.99101337, -0.98333625, -0.97332683,
       -0.9610088 , -0.94641137, -0.92956917, -0.91052214, -0.88931545,
       -0.8659994 , -0.8406293 , -0.81326532, -0.78397236, -0.75281991,
       -0.71988185, -0.68523631, -0.64896547, -0.61115536, -0.57189565,
       -0.53127946, -0.48940315, -0.44636602, -0.40227016, -0.35722016,
       -0.31132287, -0.26468716, -0.21742364, -0.16964442, -0.12146282,
       -0.07299312, -0.02435029,  0.02435029,  0.07299312,  0.12146282,
        0.16964442,  0.21742364,  0.26468716,  0.31132287,  0.35722016,
        0.40227016,  0.44636602,  0.48940315,  0.53127946,  0.57189565,
        0.61115536,  0.64896547,  0.68523631,  0.71988185,  0.75281991,
        0.78397236,  0.81326532,  0.8406293 ,  0.8659994 ,  0.88931545,
        0.91052214,  0.92956917,  0.94641137,  0.9610088 ,  0.97332683,
        0.98333625,  0.99101337,  0.99634012,  0.99930504])
    w_i = np.array(
        [0.00178328, 0.00414703, 0.00650446, 0.00884676, 0.01116814,
       0.01346305, 0.01572603, 0.01795172, 0.02013482, 0.02227017,
       0.0243527 , 0.02637747, 0.02833967, 0.03023466, 0.03205793,
       0.03380516, 0.03547221, 0.03705513, 0.03855015, 0.03995374,
       0.04126256, 0.04247352, 0.04358372, 0.04459056, 0.04549163,
       0.0462848 , 0.04696818, 0.04754017, 0.04799939, 0.04834476,
       0.04857547, 0.04869096, 0.04869096, 0.04857547, 0.04834476,
       0.04799939, 0.04754017, 0.04696818, 0.0462848 , 0.04549163,
       0.04459056, 0.04358372, 0.04247352, 0.04126256, 0.03995374,
       0.03855015, 0.03705513, 0.03547221, 0.03380516, 0.03205793,
       0.03023466, 0.02833967, 0.02637747, 0.0243527 , 0.02227017,
       0.02013482, 0.01795172, 0.01572603, 0.01346305, 0.01116814,
       0.00884676, 0.00650446, 0.00414703, 0.00178328])
    
    u_i = 0.5*((upper_bound - lower_bound)*u_points + (lower_bound + upper_bound))
    f_xi = lv_sigma_vectorized(u_i, beta_low, beta_high, eps, eps_s, f_low, f_high, lambda_, Psi, K_h, F0)
    return (upper_bound - lower_bound)*0.5*np.dot(f_xi, w_i)


@njit
def normal_cdf(x):
    return 0.5*(1 + math.erf(x/math.sqrt(2)))

@njit
def gamma_star(lambda_, h_cut, time_to_maturity):
    h_cut_over_T = h_cut/math.sqrt(time_to_maturity)
    lambda_sqrT_halves = 0.5*lambda_*math.sqrt(time_to_maturity)
    lambda_hcut_halves = 0.5*lambda_*h_cut

    N_d1 = normal_cdf(h_cut_over_T - lambda_sqrT_halves)
    N_d1 *= math.exp(-lambda_hcut_halves)

    N_d2 = normal_cdf(-h_cut_over_T - lambda_sqrT_halves)
    N_d2 *= math.exp(lambda_hcut_halves)

    expo_factor = math.exp(lambda_*lambda_*time_to_maturity*0.125)

    return expo_factor*(N_d1 + N_d2)

@njit
def gamma_big(lambda_, k, h_cut, time_to_maturity):
    if k >= h_cut:
        exponential = 0.125*lambda_*lambda_*time_to_maturity - 0.5*lambda_*h_cut
        sqrt_T = math.sqrt(time_to_maturity)
        d1 = -k/sqrt_T - 0.5*lambda_*sqrt_T
        return math.exp(exponential)*normal_cdf(d1)
    else:
        exponential = 0.125*lambda_*lambda_*time_to_maturity + 0.5*lambda_*h_cut
        sqrt_T = math.sqrt(time_to_maturity)
        d1 = k/sqrt_T - 0.5*lambda_*sqrt_T
        return gamma_star(lambda_, h_cut, time_to_maturity) - math.exp(exponential)*normal_cdf(d1)


@njit
def one_minus_density(strike, forward_rate, time_to_maturity,
                        alpha, nu, rho):
    z_k = (strike - forward_rate)*(nu/alpha)
    xhi_k = math.sqrt(1 - rho*rho + (z_k + rho)*(z_k + rho)) + z_k + rho 
    xhi_k /= 1 + rho

    h_cut = math.log((1 + rho)/(1 - rho))/(2 * nu)

    gamma = 1.5*gamma_star(nu, h_cut, time_to_maturity) - 0.5*gamma_star(3*nu, h_cut, time_to_maturity)

    E_factor = 0.5*math.exp(nu*nu*time_to_maturity*0.5)/gamma

    E_plus = 3*gamma_star(nu, h_cut + nu*time_to_maturity, time_to_maturity)
    E_plus -= gamma_star(3*nu, h_cut + nu*time_to_maturity, time_to_maturity)
    E_plus *= E_factor

    E_minus = 3*gamma_star(nu, h_cut - nu*time_to_maturity, time_to_maturity)
    E_minus -= gamma_star(3*nu, h_cut - nu*time_to_maturity, time_to_maturity)
    E_minus *= E_factor

    Gamma = rho + math.sqrt(rho*rho + (1 - rho)*(1 - rho)*E_plus*E_minus)
    Gamma *= (1 + rho)*E_plus

    k = math.log(xhi_k/Gamma)/nu

    out = 1.25*gamma_big(nu, k, h_cut, time_to_maturity)
    out -= 0.5*gamma_big(3*nu, k, h_cut, time_to_maturity)
    return out/gamma

@njit
def call_density(strike, forward_rate, time_to_maturity,
                alpha, nu, rho, mu,
                beta_low, beta_high, eps, eps_s, f_low, f_high, lambda_, Psi, K_h):
    displaced_strike = lamperti_transform_lv_sigma(strike, forward_rate, 
                                                    beta_low, beta_high, eps, eps_s, 
                                                    f_low, f_high, lambda_, Psi, K_h, forward_rate)
    displaced_strike += mu
    return one_minus_density(displaced_strike, forward_rate, time_to_maturity, alpha, nu, rho)

@njit
def put_density(strike, forward_rate, time_to_maturity,
                alpha, nu, rho, mu,
                beta_low, beta_high, eps, eps_s, f_low, f_high, lambda_, Psi, K_h):
    displaced_strike = lamperti_transform_lv_sigma(strike, forward_rate, 
                                                    beta_low, beta_high, eps, eps_s, 
                                                    f_low, f_high, lambda_, Psi, K_h, forward_rate)
    displaced_strike += mu
    return 1 - one_minus_density(displaced_strike, forward_rate, time_to_maturity, alpha, nu, rho)
    
def call_terminal_value(strike, forward_rate, time_to_maturity,
                        alpha, nu, rho, mu,
                        beta_low, beta_high, eps, eps_s, f_low, f_high, lambda_, Psi, K_h):
    extra_args = (forward_rate, time_to_maturity, 
                    alpha, nu, rho, mu, 
                    beta_low, beta_high, eps, eps_s, f_low, f_high, lambda_, Psi, K_h)
    return quad(call_density, a=strike, b=np.inf, args=extra_args)

def put_terminal_value(strike, forward_rate, time_to_maturity,
                        alpha, nu, rho, mu,
                        beta_low, beta_high, eps, eps_s, f_low, f_high, lambda_, Psi, K_h):
    extra_args = (forward_rate, time_to_maturity, 
                    alpha, nu, rho, mu, 
                    beta_low, beta_high, eps, eps_s, f_low, f_high, lambda_, Psi, K_h)
    return quad(put_density, a=-np.inf, b=strike, args=extra_args)




