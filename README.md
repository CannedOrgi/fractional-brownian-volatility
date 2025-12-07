# Implementation of Rough Volatility Models

This repository implements several classical and rough stochastic volatility models used for option pricing, together with fast neural-network approximations of the pricing maps.

The basic setup is as follows. Let \(S_t\) be the asset price, with deterministic short rate \(r(t)\) and dividend yield \(q(t)\). Under the risk-neutral measure we assume

$$
dS_t = S_t \bigl(r(t) - q(t)\bigr)\,dt + S_t \sqrt{V_t}\, dW_{2,t},
$$

where $(V_t)$ is the instantaneous variance process and $(W_1, W_2)$ are Brownian motions with correlation

$$
dW_{1,t} dW_{2,t} = \rho\, dt, \qquad \rho \in [-1,1].
$$

Different choices for the dynamics of \(V_t\) give the models described below.

---

## Heston Model

In the (classical) Heston model the variance follows a square-root diffusion,

$$
dV_t = \kappa \bigl(v_{\infty} - V_t\bigr)\, dt + \eta \sqrt{V_t}\, dW_{1,t},
$$

where
- \(\kappa \ge 0\) is the speed of mean reversion,
- \(v_{\infty} \ge 0\) is the long-run variance level,
- \(\eta \ge 0\) controls the volatility of variance.

---

## Rough Heston Model

The rough Heston model replaces the Markovian variance dynamics with a process that has fractional-Brownian-motion–type memory. The variance is given by

$$
V_t = \xi_0(t)
+ \frac{\nu}{\Gamma\!\left(H + \tfrac12\right)}
\int_0^t (t-s)^{H - \tfrac12} \sqrt{V_s}\, dW_{1,s},
\qquad t \ge 0,
$$

where
- \(H \in (0, \tfrac12)\) is the Hurst parameter controlling roughness,
- \(\nu \ge 0\) is a volatility-of-volatility parameter.

The forward variance curve \(\xi_0(t)\) is modeled as

$$
\xi_0(t)
= V_0
+ \frac{1}{\Gamma\!\left(H + \tfrac12\right)}
\int_0^t (t-s)^{H - \tfrac12}\, \theta(s)\, ds,
\qquad t \ge 0,
$$

and we may write the associated (non-negative) measure as

$$
\theta(t)\,dt + V_0 L(dt), \qquad
L(dt) = \Gamma(1/2 - H)^{-1} t^{-H - 1/2}\, dt.
$$

---

## Rough Bergomi Model

In the rough Bergomi model the variance is specified directly via a lognormal–type representation,

$$
V_t
= \xi_0(t)\,
\exp\!\left(
\eta \sqrt{2H}
\int_0^t (t-s)^{H - 1/2}\, dW_{1,s}
- \frac{\eta^2}{2} t^{2H}
\right), \qquad t \ge 0,
$$

where
- \(\xi_0(t) > 0\) is the forward variance curve,
- \(H \in (0, \tfrac12)\) is the Hurst parameter,
- \(\eta \ge 0\) again controls volatility of volatility.

---

## Extended Rough Bergomi Model

An extended version uses two independent factors \(V_{1,t}\) and \(V_{2,t}\) to give additional flexibility:

$$
V_t = \xi_0(t)\, V_{1,t} V_{2,t}.
$$

The factors are defined by

$$
V_{1,t}
= \exp\!\left(
\zeta \sqrt{2\alpha + 1}
\int_0^t (t-s)^{\alpha}\, dW_{1,s}
- \frac{\zeta^2}{2} t^{2\alpha + 1}
\right),
$$

$$
V_{2,t}
= \exp\!\left(
\lambda \sqrt{2\beta + 1}
\int_0^t (t-s)^{\beta}\, dW_{2,s}
- \frac{\lambda^2}{2} t^{2\beta + 1}
\right),
$$

where
- \(\alpha, \beta \in (-\tfrac12, \tfrac12)\),
- \(W_1\) and \(W_2\) are independent Brownian motions,
- \(\zeta, \lambda \in \mathbb{R}\).

A convenient re-parameterisation is

$$
\rho = \frac{\lambda}{\sqrt{\zeta^2 + \lambda^2}},
\qquad
\eta = \sqrt{\zeta^2 + \lambda^2},
$$

relating \(\zeta,\lambda\) to the correlation \(\rho\) and an effective volatility-of-volatility level \(\eta\).

---

## What the Code Provides

The repository is organised around three main components:

1. **Pricing routines**  
   Algorithms for pricing European calls and puts under the models above. The focus is on Monte-Carlo methods and their numerical details.

2. **Dataset generation**  
   Scripts that generate large collections of option prices across different strikes, maturities and parameter sets. These datasets are used to train neural networks that learn the pricing map.

3. **Neural-network evaluators**  
   Interfaces for evaluating pre-trained neural networks that approximate the option prices of the underlying models. Interfaces can be provided in multiple languages (for example MATLAB, Python and R).

The trained network weights can be stored separately (the corresponding training data can be very large and is not shipped directly in this repository).

---

## Speed and Accuracy

Once a neural network has been trained, it can approximate the full volatility surface extremely quickly. On a standard laptop, evaluating a surface typically takes on the order of a millisecond, and a calibration to SPX options can be performed in less than a second.

As a representative example under the rough Bergomi model, one may consider

$$
H = 0.1, \qquad
\eta = 2.1, \qquad
\rho = -0.9, \qquad
\xi_0(t) = 0.15^2.
$$

For such parameter choices the neural network reproduces Monte-Carlo prices very accurately; detailed error statistics are discussed in the associated research papers.

---

## Getting Started

The original project includes “getting started” scripts in several languages, for example:

- MATLAB examples for using the neural-network pricers,
- Python examples for neural networks,
- R examples for neural networks,
- MATLAB examples for directly working with the stochastic-volatility models.

You can mirror this structure here or adapt it to your own workflow.

---

## References

The theoretical background and numerical methods are inspired by the following works:

1. S. E. Römer, *Empirical analysis of rough and classical stochastic volatility models applied to SPX and VIX*, Quantitative Finance (2022).  
2. S. L. Heston, *A closed-form solution for options with stochastic volatility with applications to bond and currency options*, Review of Financial Studies (1993).  
3. O. El Euch, J. Gatheral, M. Rosenbaum, *Roughening Heston*, Risk Magazine (2019).  
4. C. Bayer, P. Friz, J. Gatheral, *Pricing under rough volatility*, Quantitative Finance (2016).  
5. B. Horvath, A. Muguruza, M. Tomas, *Deep learning volatility: a deep neural-network perspective on pricing and calibration in (rough) volatility models*, Quantitative Finance (2021).

---

## External Packages

One external MATLAB utility that is commonly used with this type of code is:

- Adi Navve, **Pack & Unpack variables to/from structures with enhanced functionality**, MATLAB File Exchange  
  <https://www.mathworks.com/matlabcentral/fileexchange/31532-pack-unpack-variables-to-from-structures-with-enhanced-functionality>

You can of course adapt or replace this with your own data-handling utilities.
