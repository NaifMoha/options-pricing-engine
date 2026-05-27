"""Tests for closed-form Black-Scholes and Greeks."""

import math
import pytest

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from src.black_scholes import (
    BSInputs, price, delta, gamma, vega, theta, rho,
    finite_difference_delta, finite_difference_gamma,
)
from src.monte_carlo import price_european_mc


def test_put_call_parity():
    """C - P = S e^-qT - K e^-rT under Black-Scholes."""
    inp = BSInputs(S=100, K=100, T=1.0, r=0.05, sigma=0.2, q=0.0)
    c = price(inp, "call")
    p = price(inp, "put")
    lhs = c - p
    rhs = inp.S * math.exp(-inp.q * inp.T) - inp.K * math.exp(-inp.r * inp.T)
    assert abs(lhs - rhs) < 1e-10


def test_atm_call_known_value():
    """At-the-money call with standard params has a textbook value near 10.45."""
    inp = BSInputs(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
    c = price(inp, "call")
    assert abs(c - 10.4506) < 1e-3


def test_call_delta_bounds():
    """Call delta lives in (0, 1) for finite T."""
    inp = BSInputs(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
    d = delta(inp, "call")
    assert 0 < d < 1


def test_finite_difference_delta_matches_analytical():
    inp = BSInputs(S=100, K=105, T=0.5, r=0.03, sigma=0.25)
    analytical = delta(inp, "call")
    numerical = finite_difference_delta(inp, "call", h=0.01)
    assert abs(analytical - numerical) < 1e-4


def test_finite_difference_gamma_matches_analytical():
    inp = BSInputs(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
    analytical = gamma(inp)
    numerical = finite_difference_gamma(inp, "call", h=0.5)
    assert abs(analytical - numerical) < 1e-4


def test_gamma_positive():
    inp = BSInputs(S=100, K=100, T=0.5, r=0.05, sigma=0.2)
    assert gamma(inp) > 0


def test_vega_positive():
    inp = BSInputs(S=100, K=100, T=0.5, r=0.05, sigma=0.2)
    assert vega(inp) > 0


def test_call_theta_negative():
    """A call's theta is generally negative when not deep ITM with high q."""
    inp = BSInputs(S=100, K=100, T=0.5, r=0.05, sigma=0.2)
    assert theta(inp, "call") < 0


def test_monte_carlo_converges_to_black_scholes():
    inp = BSInputs(S=100, K=100, T=1.0, r=0.05, sigma=0.2)
    analytical = price(inp, "call")
    mc = price_european_mc(inp, "call", n_paths=200_000, n_steps=1, seed=42)
    # 3 standard-error window
    assert abs(mc.price - analytical) < 3 * mc.std_error
