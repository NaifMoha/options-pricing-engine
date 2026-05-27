"""
Black-Scholes pricing and Greeks for European options.

The model assumes lognormal underlying, constant volatility, and no dividends.
Both pricing and Greeks have closed-form solutions under Black-Scholes; this
module implements them and exposes a small, well-typed API.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from scipy.stats import norm


OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class BSInputs:
    """Inputs to a Black-Scholes valuation."""
    S: float        # spot price
    K: float        # strike
    T: float        # time to maturity (years)
    r: float        # risk-free rate (cont. comp.)
    sigma: float    # volatility (annualized)
    q: float = 0.0  # continuous dividend yield

    def __post_init__(self) -> None:
        if self.T < 0:
            raise ValueError("Time to maturity must be non-negative.")
        if self.sigma <= 0:
            raise ValueError("Volatility must be strictly positive.")
        if self.S <= 0 or self.K <= 0:
            raise ValueError("Spot and strike must be strictly positive.")


def _d1_d2(inp: BSInputs) -> tuple[float, float]:
    S, K, T, r, sigma, q = inp.S, inp.K, inp.T, inp.r, inp.sigma, inp.q
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    return d1, d2


def price(inp: BSInputs, option: OptionType) -> float:
    """Black-Scholes price for a European call or put."""
    if inp.T == 0:
        intrinsic = max(inp.S - inp.K, 0.0) if option == "call" else max(inp.K - inp.S, 0.0)
        return intrinsic
    d1, d2 = _d1_d2(inp)
    discounted_strike = inp.K * math.exp(-inp.r * inp.T)
    discounted_spot = inp.S * math.exp(-inp.q * inp.T)
    if option == "call":
        return discounted_spot * norm.cdf(d1) - discounted_strike * norm.cdf(d2)
    return discounted_strike * norm.cdf(-d2) - discounted_spot * norm.cdf(-d1)


def delta(inp: BSInputs, option: OptionType) -> float:
    """dPrice/dSpot."""
    d1, _ = _d1_d2(inp)
    sign = 1.0 if option == "call" else -1.0
    return sign * math.exp(-inp.q * inp.T) * norm.cdf(sign * d1)


def gamma(inp: BSInputs) -> float:
    """d^2 Price / dSpot^2 (same for calls and puts)."""
    d1, _ = _d1_d2(inp)
    return math.exp(-inp.q * inp.T) * norm.pdf(d1) / (inp.S * inp.sigma * math.sqrt(inp.T))


def vega(inp: BSInputs) -> float:
    """dPrice/dSigma. Returned per unit of volatility (multiply by 0.01 for %)."""
    d1, _ = _d1_d2(inp)
    return inp.S * math.exp(-inp.q * inp.T) * norm.pdf(d1) * math.sqrt(inp.T)


def theta(inp: BSInputs, option: OptionType) -> float:
    """dPrice/dT. Per year."""
    d1, d2 = _d1_d2(inp)
    first = -(inp.S * math.exp(-inp.q * inp.T) * norm.pdf(d1) * inp.sigma) / (2.0 * math.sqrt(inp.T))
    if option == "call":
        second = inp.q * inp.S * math.exp(-inp.q * inp.T) * norm.cdf(d1)
        third = -inp.r * inp.K * math.exp(-inp.r * inp.T) * norm.cdf(d2)
        return first + second + third
    second = -inp.q * inp.S * math.exp(-inp.q * inp.T) * norm.cdf(-d1)
    third = inp.r * inp.K * math.exp(-inp.r * inp.T) * norm.cdf(-d2)
    return first + second + third


def rho(inp: BSInputs, option: OptionType) -> float:
    """dPrice/dr. Per unit of rate."""
    _, d2 = _d1_d2(inp)
    if option == "call":
        return inp.K * inp.T * math.exp(-inp.r * inp.T) * norm.cdf(d2)
    return -inp.K * inp.T * math.exp(-inp.r * inp.T) * norm.cdf(-d2)


def greeks(inp: BSInputs, option: OptionType) -> dict:
    """Return all Greeks as a dict, in addition to the price."""
    return {
        "price": price(inp, option),
        "delta": delta(inp, option),
        "gamma": gamma(inp),
        "vega": vega(inp),
        "theta": theta(inp, option),
        "rho": rho(inp, option),
    }


def finite_difference_delta(inp: BSInputs, option: OptionType, h: float = 0.01) -> float:
    """Delta computed via central finite differences. Sanity check for analytical delta."""
    up = BSInputs(inp.S + h, inp.K, inp.T, inp.r, inp.sigma, inp.q)
    dn = BSInputs(inp.S - h, inp.K, inp.T, inp.r, inp.sigma, inp.q)
    return (price(up, option) - price(dn, option)) / (2 * h)


def finite_difference_gamma(inp: BSInputs, option: OptionType, h: float = 0.5) -> float:
    """Gamma via central differences."""
    up = BSInputs(inp.S + h, inp.K, inp.T, inp.r, inp.sigma, inp.q)
    dn = BSInputs(inp.S - h, inp.K, inp.T, inp.r, inp.sigma, inp.q)
    return (price(up, option) - 2 * price(inp, option) + price(dn, option)) / (h * h)
