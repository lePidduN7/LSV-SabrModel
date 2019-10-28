import attr
from lmfit import Parameters, minimize
from typing import List
from QuantLib import Date, Actual365Fixed, TARGET 
from QuantLib import bachelierBlackFormulaImpliedVol, blackFormulaImpliedStdDev
from numerix import call_terminal_value, put_terminal_value
from numerix import CDF_underlying

@attr.s(auto_attribs=True)
class SmileCurve:
    maturity: Date
    forward_rate: float
    local_volatility_paramters: Parameters
    stochastic_volatility_parameters: Parameters
    lognormal_shift: float = 0.0
