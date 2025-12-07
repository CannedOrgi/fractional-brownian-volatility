# Implementation of Rough Volatility Models

This repository implements several classical and rough stochastic volatility models for option pricing, together with fast neural-network approximations of the pricing maps.

We work under a risk-neutral measure. Let $S_t$ be the asset price, $r(t)$ the risk-free rate, and $q(t)$ the dividend yield (all deterministic). The price dynamics are

$dS_t = S_t ( r(t) - q(t) ) dt + S_t \sqrt{V_t} \, dW_{2,t},$

where $V_t$ is the instantaneous variance process. The Brownian motions $W_1$ and $W_2$ satisfy

$dW_{1,t} dW_{2,t} = \rho \, dt,$ with $\rho \in [-1,1]$.

Different choices for the dynamics of $V_t$ give the models described below.

---

## Heston Model

In the classical Heston model, the variance follows a square-root process:

$dV_t = \kappa ( v_{\infty} - V_t ) dt + \eta \sqrt{V_t} \, dW_{1,t},$

where $\kappa \ge 0$ is the mean-reversion speed, $v_{\infty} \ge 0$ is the long-run variance level, and $\eta \ge 0$ controls the volatility of variance.

---

## Rough Heston Model

The rough Heston model introduces fractional behavior in the variance process:

$V_t = \xi_0(t) + \frac{\nu}{\Gamma(H + 1/2)} \int_0^t (t-s)^{H - 1/2} \sqrt{V_s} \, dW_{1,s}, \quad t \ge 0,$

with Hurst parameter $H \in (0, 1/2)$ and roughness scale $\nu \ge 0$.

The forward variance curve $\xi_0(t)$ is given by

$\xi_0(t) = V_0 + \frac{1}{\Gamma(H + 1/2)} \int_0^t (t-s)^{H - 1/2} \theta(s) \, ds.$

The measure associated with the kernel can be written as $\theta(t) dt + V_0 L(dt)$, with

$L(dt) = \Gamma(1/2 - H)^{-1} t^{-H - 1/2} dt.$

---

## Rough Bergomi Model

In the rough Bergomi model, the variance is modeled as a lognormal-type process:

$V_t = \xi_0(t) \exp \Big( \eta \sqrt{2H} \int_0^t (t-s)^{H - 1/2} dW_{1,s} - \frac{\eta^2}{2} t^{2H} \Big), \quad t \ge 0,$

where $\xi_0(t) > 0$ is the forward variance curve, $H \in (0, 1/2)$, and $\eta \ge 0$ controls volatility of volatility.

---

## Extended Rough Bergomi Model

An extended version of the rough Bergomi model uses two independent factors $V_{1,t}$ and $V_{2,t}$:

$V_t = \xi_0(t) V_{1,t} V_{2,t}.$

The factors are

$V_{1,t} = \exp \Big( \zeta \sqrt{2\alpha + 1} \int_0^t (t-s)^{\alpha} dW_{1,s} - \frac{\zeta^2}{2} t^{2\alpha + 1} \Big),$

$V_{2,t} = \exp \Big( \lambda \sqrt{2\beta + 1} \int_0^t (t-s)^{\beta} dW_{2,s} - \frac{\lambda^2}{2} t^{2\beta + 1} \Big),$

where $\alpha, \beta \in (-1/2, 1/2)$, $W_1$ and $W_2$ are independent Brownian motions, and $\zeta, \lambda \in \mathbb{R}$.

A useful reparameterisation links these to the correlation $\rho$ and an effective volatility-of-volatility level $\eta$ via

$\rho = \frac{\lambda}{\sqrt{\zeta^2 + \lambda^2}}, \quad \eta = \sqrt{\zeta^2 + \lambda^2}.$

---

## What the Code Provides

The repository is organised around three main components:

1. **Pricing routines**  
   Implementations of Monte Carlo pricing algorithms for European calls and puts under the models above.

2. **Dataset generation**  
   Scripts that generate large datasets of option prices for different strikes, maturities and parameter choices. These datasets are used for training neural networks.

3. **Neural-network evaluators**  
   Interfaces for evaluating neural networks that approximate the pricing map of each model. The interfaces can be written in multiple languages (for example MATLAB, Python or R). Trained network weights can be stored separately to keep the repository size manageable.

---

## Speed and Accuracy

Once a neural network has been trained, it can evaluate an entire implied-volatility surface extremely quickly. On a standard laptop, computing a full surface typically takes on the order of a millisecond, and a calibration to SPX options can be run in well under a second.

As a representative parameter set for the rough Bergomi model, one may consider

$H = 0.1, \quad \eta = 2.1, \quad \rho = -0.9, \quad \xi_0(t) = 0.15^2.$

For such values, the neural network reproduces Monte Carlo prices with high accuracy; more detailed error analysis is given in the underlying research papers.

---

## References

The models and methods here are based on the following references:

1. S. E. Römer, “Empirical analysis of rough and classical stochastic volatility models applied to SPX and VIX”, Quantitative Finance, 2022.  
2. S. L. Heston, “A closed-form solution for options with stochastic volatility with applications to bond and currency options”, Review of Financial Studies, 1993.  
3. O. El Euch, J. Gatheral, M. Rosenbaum, “Roughening Heston”, Risk, 2019.  
4. C. Bayer, P. Friz, J. Gatheral, “Pricing under rough volatility”, Quantitative Finance, 2016.  
5. B. Horvath, A. Muguruza, M. Tomas, “Deep learning volatility: a deep neural-network perspective on pricing and calibration in (rough) volatility models”, Quantitative Finance, 2021.

---

## External Packages

One useful external MATLAB utility for working with structured data is:

- Adi Navve (2020), “Pack & Unpack variables to & from structures with enhanced functionality”, MATLAB File Exchange:  
  https://www.mathworks.com/matlabcentral/fileexchange/31532-pack-unpack-variables-to-from-structures-with-enhanced-functionality
