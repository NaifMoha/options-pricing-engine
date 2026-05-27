"""Options pricing and Greeks under Black-Scholes and Merton jump-diffusion."""

from .black_scholes import (
    BSInputs,
    price,
    delta,
    gamma,
    vega,
    theta,
    rho,
    greeks,
    finite_difference_delta,
    finite_difference_gamma,
)

from .monte_carlo import (
    MCResult,
    simulate_gbm_paths,
    simulate_jump_diffusion_paths,
    european_payoff,
    asian_payoff,
    price_european_mc,
    price_path_dependent_mc,
)

__all__ = [
    "BSInputs",
    "price",
    "delta",
    "gamma",
    "vega",
    "theta",
    "rho",
    "greeks",
    "finite_difference_delta",
    "finite_difference_gamma",
    "MCResult",
    "simulate_gbm_paths",
    "simulate_jump_diffusion_paths",
    "european_payoff",
    "asian_payoff",
    "price_european_mc",
    "price_path_dependent_mc",
]
