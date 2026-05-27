"""
Monte Carlo simulation for European and path-dependent options.

Two underlying processes:
  - Geometric Brownian motion (for benchmarking against Black-Scholes)
  - Merton jump-diffusion (captures tail risk that pure diffusion misses)

The path simulator returns full paths so we can price path-dependent payoffs
(Asian options, barrier options) with the same engine.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

from .black_scholes import BSInputs


OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class MCResult:
    """Holds the simulated price plus a standard-error estimate."""
    price: float
    std_error: float
    n_paths: int


def simulate_gbm_paths(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    n_paths: int,
    n_steps: int,
    q: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate geometric Brownian motion paths.

    Returns array of shape (n_paths, n_steps + 1) with path[:, 0] == S0.
    """
    rng = rng if rng is not None else np.random.default_rng()
    dt = T / n_steps
    drift = (r - q - 0.5 * sigma * sigma) * dt
    diffusion = sigma * math.sqrt(dt)

    # Standard normal draws
    z = rng.standard_normal((n_paths, n_steps))
    log_returns = drift + diffusion * z

    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = S0
    paths[:, 1:] = S0 * np.exp(np.cumsum(log_returns, axis=1))
    return paths


def simulate_jump_diffusion_paths(
    S0: float,
    r: float,
    sigma: float,
    T: float,
    n_paths: int,
    n_steps: int,
    jump_intensity: float,       # lambda: expected jumps per year
    jump_mean: float,            # mean of log-jump size
    jump_std: float,             # std of log-jump size
    q: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Simulate Merton jump-diffusion paths.

    log S_{t+dt} - log S_t = (r - q - 0.5 sigma^2 - lambda * kappa) dt
                           + sigma sqrt(dt) Z + sum_{i=1..N_t} log(J_i)
    where N_t ~ Poisson(lambda * dt), log(J_i) ~ Normal(jump_mean, jump_std^2),
    and kappa = E[J - 1] = exp(jump_mean + 0.5 jump_std^2) - 1 (jump compensator).
    """
    rng = rng if rng is not None else np.random.default_rng()
    dt = T / n_steps
    kappa = math.exp(jump_mean + 0.5 * jump_std * jump_std) - 1.0
    drift = (r - q - 0.5 * sigma * sigma - jump_intensity * kappa) * dt
    diffusion = sigma * math.sqrt(dt)

    z = rng.standard_normal((n_paths, n_steps))
    n_jumps = rng.poisson(jump_intensity * dt, size=(n_paths, n_steps))

    # Aggregate jump log-returns per step
    # Sum of n_jumps i.i.d. N(jump_mean, jump_std^2) ~ N(n*jump_mean, n*jump_std^2)
    # We sample directly using the closed form.
    jump_means = n_jumps * jump_mean
    jump_vars = n_jumps * (jump_std * jump_std)
    jump_log_returns = jump_means + np.sqrt(jump_vars) * rng.standard_normal((n_paths, n_steps))

    log_returns = drift + diffusion * z + jump_log_returns
    paths = np.empty((n_paths, n_steps + 1))
    paths[:, 0] = S0
    paths[:, 1:] = S0 * np.exp(np.cumsum(log_returns, axis=1))
    return paths


def european_payoff(S_T: np.ndarray, K: float, option: OptionType) -> np.ndarray:
    if option == "call":
        return np.maximum(S_T - K, 0.0)
    return np.maximum(K - S_T, 0.0)


def asian_payoff(paths: np.ndarray, K: float, option: OptionType) -> np.ndarray:
    """Arithmetic-average Asian option payoff."""
    avg = paths[:, 1:].mean(axis=1)
    return european_payoff(avg, K, option)


def price_european_mc(
    inp: BSInputs,
    option: OptionType,
    n_paths: int = 100_000,
    n_steps: int = 1,
    process: Literal["gbm", "jump"] = "gbm",
    jump_params: dict | None = None,
    seed: int | None = None,
) -> MCResult:
    """Price a European option via Monte Carlo."""
    rng = np.random.default_rng(seed)
    if process == "gbm":
        paths = simulate_gbm_paths(
            inp.S, inp.r, inp.sigma, inp.T, n_paths, n_steps, q=inp.q, rng=rng
        )
    elif process == "jump":
        jp = jump_params or {"jump_intensity": 0.5, "jump_mean": -0.05, "jump_std": 0.1}
        paths = simulate_jump_diffusion_paths(
            inp.S, inp.r, inp.sigma, inp.T, n_paths, n_steps,
            q=inp.q, rng=rng, **jp
        )
    else:
        raise ValueError(f"Unknown process: {process}")

    payoff = european_payoff(paths[:, -1], inp.K, option)
    discount = math.exp(-inp.r * inp.T)
    discounted = discount * payoff
    mean = discounted.mean()
    se = discounted.std(ddof=1) / math.sqrt(n_paths)
    return MCResult(price=mean, std_error=se, n_paths=n_paths)


def price_path_dependent_mc(
    inp: BSInputs,
    option: OptionType,
    payoff_fn: Callable[[np.ndarray, float, OptionType], np.ndarray],
    n_paths: int = 100_000,
    n_steps: int = 252,
    seed: int | None = None,
) -> MCResult:
    """Price a path-dependent option given a payoff function on full paths."""
    rng = np.random.default_rng(seed)
    paths = simulate_gbm_paths(inp.S, inp.r, inp.sigma, inp.T, n_paths, n_steps, q=inp.q, rng=rng)
    payoff = payoff_fn(paths, inp.K, option)
    discount = math.exp(-inp.r * inp.T)
    discounted = discount * payoff
    return MCResult(price=discounted.mean(), std_error=discounted.std(ddof=1) / math.sqrt(n_paths), n_paths=n_paths)
