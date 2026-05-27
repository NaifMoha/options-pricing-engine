# Options Pricing & Greeks Engine

A small Python options-pricing library covering Black-Scholes closed-form pricing, the full set of Greeks (analytical and finite-difference), Monte Carlo simulation, and a Merton jump-diffusion extension for tail-risk scenarios.

## What's here

- `src/black_scholes.py` — Closed-form European call/put prices and Greeks (delta, gamma, vega, theta, rho). Each Greek is implemented analytically with a finite-difference reference for sanity-checking.
- `src/monte_carlo.py` — Path simulation under geometric Brownian motion and Merton jump-diffusion. Pricers for European and path-dependent payoffs (Asian and a hook for arbitrary payoff functions).
- `tests/test_black_scholes.py` — Tests covering put-call parity, textbook benchmark values, sign conventions on Greeks, and Monte Carlo convergence to the analytical price within standard-error bounds.
- `notebooks/demo.ipynb` — Walkthrough: price an ATM call, plot Greeks across spot and time, compare GBM and jump-diffusion price surfaces.

## Quick start

```bash
pip install -r requirements.txt
python -m pytest tests/
```

```python
from src.black_scholes import BSInputs, greeks

inp = BSInputs(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
print(greeks(inp, "call"))
# {'price': 10.45, 'delta': 0.64, 'gamma': 0.019, 'vega': 37.5, 'theta': -6.41, 'rho': 53.2}
```

```python
from src.black_scholes import BSInputs
from src.monte_carlo import price_european_mc

inp = BSInputs(S=100, K=100, T=1.0, r=0.05, sigma=0.20)
mc = price_european_mc(inp, "call", n_paths=200_000, seed=42)
print(f"MC: {mc.price:.4f}  ±{1.96*mc.std_error:.4f}  (95% CI)")
```

## Modeling notes

- Black-Scholes assumes a lognormal underlying with constant volatility, no dividends (or a continuous yield), and frictionless trading. The library exposes the dividend yield `q` as an input for completeness.
- The Greeks are derived directly from the closed-form price function. The finite-difference variants in `black_scholes.py` exist for sanity-checking and are not the primary computation path.
- The Merton jump-diffusion extension adds a compound Poisson jump component to the log-return, with the standard jump compensator term applied to the drift so the risk-neutral expectation of the underlying remains correct. It produces fatter tails than pure GBM, which matters for far-OTM option prices.

## Roadmap

- American-style options via least-squares Monte Carlo (Longstaff-Schwartz)
- Local volatility calibration to a quoted vol surface
- Heston stochastic-volatility paths
