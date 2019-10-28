import math
import numba as nb
from numba import cfunc, carray, types, vectorize, guvectorize, njit, float64
import numpy as np
from scipy import LowLevelCallable
from scipy.integrate import quad, quadrature

@njit
def lv_sigma(u, beta_low, beta_high, eps, eps_s, f_low, f_high, lambda_, Psi, K_h, F0):
    """
    Argument of the Lamperti's transform that returns the local displacement
    of the stochastic volatility function. Inverse of Def (21)
    """
    sigma_argument = min(max(u, f_low), f_high)
    heta_1 = eps * math.log(1 + math.exp(sigma_argument/eps))
    heta_2 = eps_s * math.log(1 + math.exp((sigma_argument - K_h)/eps_s))
    beta_factor = math.tanh(lambda_*(sigma_argument - F0))
    beta = 0.5*(beta_high + beta_low) + 0.5*(beta_high - beta_low)*beta_factor
    return 1.0/(math.pow(heta_1, beta) + Psi*heta_2)

@njit
def lv_sigma_vectorized(u_array, beta_low, beta_high, eps, eps_s, f_low, f_high, lambda_, Psi, K_h, F0):
    """
    Argument of the Lamperti's transform that returns the local displacement
    of the stochastic volatility function. Vectorized version
    """
    sigma_output = np.zeros(len(u_array))
    for i, u in enumerate(u_array):
        sigma_output[i] = lv_sigma(u, beta_low, beta_high, eps, eps_s, f_low, f_high, lambda_, Psi, K_h, F0)
    return sigma_output

@njit
def lamperti_transform_lv_sigma(upper_bound, lower_bound, beta_low, beta_high, eps, eps_s, f_low, f_high, lambda_, Psi, K_h, F0):
    """
    Local volatility function obtained as Lamperti's transform of the
    underlying displacement function. Def Int_F0^f(1/(sigma(u)))du 
    pag 2 in Bang(2018). This function performs the integral numerically
    with a Gaussian Quadrature over 128 points. Points and weights in the
    (1, 1) interval are cached. The integral is then computed by changing 
    the integration interval from (-1, 1) to (lower_bound, upper_bound).
    """
    u_points = np.array(
                [-0.99982489, -0.99907746, -0.99773325, -0.99579276, -0.99325711,
                -0.99012782, -0.98640674, -0.98209611, -0.97719849, -0.97171682,
                -0.96565437, -0.95901476, -0.95180196, -0.94402029, -0.93567439,
                -0.92676925, -0.9173102 , -0.90730288, -0.89675329, -0.88566772,
                -0.8740528 , -0.86191547, -0.84926299, -0.83610292, -0.82244312,
                -0.80829176, -0.79365729, -0.77854848, -0.76297433, -0.74694417,
                -0.73046757, -0.71355438, -0.69621471, -0.67845892, -0.66029763,
                -0.64174169, -0.62280219, -0.60349046, -0.58381802, -0.56379665,
                -0.5434383 , -0.52275515, -0.50175956, -0.48046407, -0.45888142,
                -0.4370245 , -0.41490638, -0.39254028, -0.36993956, -0.34711773,
                -0.32408844, -0.30086544, -0.27746262, -0.25389397, -0.23017356,
                -0.20631559, -0.18233431, -0.15824404, -0.1340592 , -0.10979423,
                -0.08546364, -0.06108197, -0.03666379, -0.0122237 ,  0.0122237,
                0.03666379,  0.06108197,  0.08546364,  0.10979423,  0.1340592,
                0.15824404,  0.18233431,  0.20631559,  0.23017356,  0.25389397,
                0.27746262,  0.30086544,  0.32408844,  0.34711773,  0.36993956,
                0.39254028,  0.41490638,  0.4370245 ,  0.45888142,  0.48046407,
                0.50175956,  0.52275515,  0.5434383 ,  0.56379665,  0.58381802,
                0.60349046,  0.62280219,  0.64174169,  0.66029763,  0.67845892,
                0.69621471,  0.71355438,  0.73046757,  0.74694417,  0.76297433,
                0.77854848,  0.79365729,  0.80829176,  0.82244312,  0.83610292,
                0.84926299,  0.86191547,  0.8740528 ,  0.88566772,  0.89675329,
                0.90730288,  0.9173102 ,  0.92676925,  0.93567439,  0.94402029,
                0.95180196,  0.95901476,  0.96565437,  0.97171682,  0.97719849,
                0.98209611,  0.98640674,  0.99012782,  0.99325711,  0.99579276,
                0.99773325,  0.99907746,  0.99982489])
    w_i = np.array(
                [0.00044938, 0.00104581, 0.0016425 , 0.00223829, 0.00283275,
                0.00342553, 0.00401625, 0.00460458, 0.00519016, 0.00577264,
                0.00635166, 0.00692689, 0.00749798, 0.00806459, 0.00862638,
                0.00918301, 0.00973415, 0.01027948, 0.01081866, 0.01135138,
                0.01187731, 0.01239614, 0.01290756, 0.01341127, 0.01390696,
                0.01439435, 0.01487312, 0.01534301, 0.01580373, 0.016255,
                0.01669656, 0.01712814, 0.01754948, 0.01796033, 0.01836044,
                0.01874959, 0.01912752, 0.01949403, 0.01984888, 0.02019187,
                0.02052279, 0.02084145, 0.02114765, 0.02144121, 0.02172195,
                0.02198971, 0.02224433, 0.02248565, 0.02271354, 0.02292784,
                0.02312845, 0.02331523, 0.02348808, 0.02364688, 0.02379156,
                0.02392201, 0.02403817, 0.02413996, 0.02422732, 0.0243002 ,
                0.02435856, 0.02440236, 0.02443157, 0.02444618, 0.02444618,
                0.02443157, 0.02440236, 0.02435856, 0.0243002 , 0.02422732,
                0.02413996, 0.02403817, 0.02392201, 0.02379156, 0.02364688,
                0.02348808, 0.02331523, 0.02312845, 0.02292784, 0.02271354,
                0.02248565, 0.02224433, 0.02198971, 0.02172195, 0.02144121,
                0.02114765, 0.02084145, 0.02052279, 0.02019187, 0.01984888,
                0.01949403, 0.01912752, 0.01874959, 0.01836044, 0.01796033,
                0.01754948, 0.01712814, 0.01669656, 0.016255, 0.01580373,
                0.01534301, 0.01487312, 0.01439435, 0.01390696, 0.01341127,
                0.01290756, 0.01239614, 0.01187731, 0.01135138, 0.01081866,
                0.01027948, 0.00973415, 0.00918301, 0.00862638, 0.00806459,
                0.00749798, 0.00692689, 0.00635166, 0.00577264, 0.00519016,
                0.00460458, 0.00401625, 0.00342553, 0.00283275, 0.00223829,
                0.0016425 , 0.00104581, 0.00044938])
    
    u_i = 0.5*((upper_bound - lower_bound)*u_points + (lower_bound + upper_bound))
    f_xi = lv_sigma_vectorized(u_i, beta_low, beta_high, eps, eps_s, f_low, f_high, lambda_, Psi, K_h, F0)
    return (upper_bound - lower_bound)*0.5*np.dot(f_xi, w_i)

@njit
def normal_cdf(x):
    """
    Cumulative distribution function of a standard Normal random variable.
    """
    return 0.5*(1 + math.erf(x/math.sqrt(2)))

@njit
def gamma_star(lambda_, h_cut, time_to_maturity):
    """
    Auxiliary function for the computation of the CDF of the underlying 
    under displaced-stochastic volatility model. Refer to (18) in Bang (2018).
    """
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
    """
    Auxiliary function for the computation of the CDF of the underlying 
    under displaced-stochastic volatility model. Refer to (19) in Bang (2018).
    """
    if k >= - h_cut:
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
def one_minus_CDF(strike, forward_rate, time_to_maturity,
                    alpha, nu, rho):
    """
    CDF of the process underlying the Sabr normal approximation
    by Bang (2018).
    """
    z_k = (strike - forward_rate)*(nu/alpha) # def Lemma2
    z_k_plus_rho = z_k + rho
    xhi_k = math.sqrt(1 - rho*rho + z_k_plus_rho*z_k_plus_rho) + z_k_plus_rho
    xhi_k /= 1 + rho

    h_cut = math.log((1 + rho)/(1 - rho))/(2 * nu) # def (10)

    gamma = 1.5*gamma_star(nu, h_cut, time_to_maturity) # def (15)
    gamma -= 0.5*gamma_star(3*nu, h_cut, time_to_maturity) 

    E_factor = 0.5*math.exp(nu*nu*time_to_maturity*0.5)/gamma # def (17)

    E_plus = 3*gamma_star(nu, h_cut + nu*time_to_maturity, time_to_maturity)
    E_plus -= gamma_star(3*nu, h_cut + nu*time_to_maturity, time_to_maturity)
    E_plus *= E_factor # def (17), positive

    E_minus = 3*gamma_star(nu, h_cut - nu*time_to_maturity, time_to_maturity)
    E_minus -= gamma_star(3*nu, h_cut - nu*time_to_maturity, time_to_maturity)
    E_minus *= E_factor # def (17), negative

    Gamma = rho + math.sqrt(rho*rho + (1 - rho*rho)*E_plus*E_minus) # def (16)
    Gamma /= (1 + rho)*E_plus

    k = math.log(xhi_k/Gamma)/nu # def (14)

    out = 1.5*gamma_big(nu, k, h_cut, time_to_maturity)
    out -= 0.5*gamma_big(3*nu, k, h_cut, time_to_maturity)
    return out/gamma

@njit(cache=True)
def one_minus_CDF_underlying(strike, forward_rate, time_to_maturity,
                                alpha, nu, rho, mu,
                                beta_low, beta_high, eps, eps_s, f_low, f_high, lambda_, Psi, K_h):
    """
    One minus CDF of the locally displaced underlyng process in the LV-Sabr model
    by Bang et al. Argument of Def (5) for Call prices.
    """
    G_k = lamperti_transform_lv_sigma(strike, forward_rate, 
                                        beta_low, beta_high, eps, eps_s, 
                                        f_low, f_high, lambda_, Psi, K_h, forward_rate)
    displaced_strike = G_k + mu
    return one_minus_CDF(displaced_strike, forward_rate, time_to_maturity, alpha, nu, rho)

@njit(cache=True)
def CDF_underlying(strike, forward_rate, time_to_maturity,
                    alpha, nu, rho, mu,
                    beta_low, beta_high, eps, eps_s, f_low, f_high, lambda_, Psi, K_h):
    """
    CDF of the locally displaced underlyng process in the LV-Sabr model
    by Bang et al. Argument of Def (5) for Put prices.
    """
    G_k = lamperti_transform_lv_sigma(strike, forward_rate, 
                                        beta_low, beta_high, eps, eps_s,
                                        f_low, f_high, lambda_, Psi, K_h, forward_rate)
    displaced_strike = G_k + mu
    return 1.0 - one_minus_CDF(displaced_strike, forward_rate, time_to_maturity, alpha, nu, rho)

def call_terminal_value(strike, forward_rate, time_to_maturity,
                        alpha, nu, rho, mu,
                        beta_low, beta_high, eps, eps_s, f_low, f_high, lambda_, Psi, K_h):
    """
    Undiscounted value of the Call price under the LV-Sabr model 
    by Bang (2018). Def (5)
    """
    extra_args = (forward_rate, time_to_maturity,
                    alpha, nu, rho, mu, 
                    beta_low, beta_high, eps, eps_s, f_low, f_high, lambda_, Psi, K_h)
    return quad(one_minus_CDF_underlying, a=strike, b=np.inf, epsabs=1e-6, epsrel=1e-6, args=extra_args, limit=100)

def put_terminal_value(strike, forward_rate, time_to_maturity,
                        alpha, nu, rho, mu,
                        beta_low, beta_high, eps, eps_s, f_low, f_high, lambda_, Psi, K_h):
    """
    Undiscounted value of the Put price under the LV-Sabr model
    by Bang (2018). Def (5)
    """
    extra_args = (forward_rate, time_to_maturity, 
                    alpha, nu, rho, mu, 
                    beta_low, beta_high, eps, eps_s, f_low, f_high, lambda_, Psi, K_h)
    return quad(CDF_underlying, a=-np.inf, b=strike, epsabs=1e-6, epsrel=1e-6, args=extra_args, limit=100)

#### Extra function for checking integration is correct
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

