import math
import numba as nb
from numba.extending import overload
from scipy import LowLevelCallable
from scipy.integrate import quad

lv_signature = nb.types.double(nb.types.intc, nb.types.CPointer(nb.types.double))
@nb.cfunc(lv_signature)
def lv_sigma(n, xx):
    """
    Argument of the Lamperti's transform that return the local displacement
    of the stochastic volatility function. The function has to be wrapped in a
    scipy.LowLevelCallable in order to be integrated efficiently.
    """
    in_array = nb.carray(xx, (n, ))
    f, beta_low, beta_high, eps, eps_s, f_low, f_high, lambda_, Psi, K_h, F0 = in_array

    sigma_argument = min(max(f, f_low), f_high)
    heta_1 = eps * math.log(1 + math.exp(sigma_argument/eps))
    heta_2 = eps_s * math.log(1 + math.exp((sigma_argument - K_h)/eps_s))
    beta_factor = math.tanh(lambda_*(sigma_argument - F0))
    beta = 0.5*(beta_high + beta_low) - 0.5*(beta_high - beta_low)*beta_factor
    return 1.0/(math.pow(heta_1, beta) + Psi*heta_2)

lv_sigma_integrand = LowLevelCallable(lv_sigma.ctypes)

