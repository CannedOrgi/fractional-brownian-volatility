Implementation of Rough Volatility Models

This repository provides a clean Python-oriented implementation of several classical and rough stochastic volatility models used in modern derivatives pricing. It also includes fast neural-network approximations for option valuation under these models.

The goal of this project is to give an accessible but technically solid reference for simulating volatility dynamics, pricing European options, and exploring machine-learning-based surrogates for speed and accuracy.

Model Setup

Let 
𝑆
𝑡
S
t
	​

 be the asset price, with risk-free rate 
𝑟
(
𝑡
)
r(t) and dividend yield 
𝑞
(
𝑡
)
q(t) (both deterministic). Under standard no-arbitrage assumptions, the risk-neutral dynamics of 
𝑆
𝑡
S
t
	​

 take the form:

𝑑
𝑆
𝑡
=
𝑆
𝑡
(
𝑟
(
𝑡
)
−
𝑞
(
𝑡
)
)
 
𝑑
𝑡
+
𝑆
𝑡
𝑉
𝑡
 
𝑑
𝑊
2
,
𝑡
,
dS
t
	​

=S
t
	​

(r(t)−q(t))dt+S
t
	​

V
t
	​

	​

dW
2,t
	​

,

where 
𝑉
𝑡
V
t
	​

 is the instantaneous variance process, and 
𝑊
1
,
𝑊
2
W
1
	​

,W
2
	​

 are Brownian motions with correlation

𝑑
𝑊
1
,
𝑡
𝑑
𝑊
2
,
𝑡
=
𝜌
𝑑
𝑡
,
𝜌
∈
[
−
1
,
1
]
.
dW
1,t
	​

dW
2,t
	​

=ρdt,ρ∈[−1,1].

Different models specify different dynamics for 
𝑉
𝑡
V
t
	​

.
Below we outline the models implemented in this project.

Classical Stochastic Volatility Model
Heston Model

The Heston volatility process is given by:

𝑑
𝑉
𝑡
=
𝜅
(
𝑣
∞
−
𝑉
𝑡
)
 
𝑑
𝑡
+
𝜂
𝑉
𝑡
 
𝑑
𝑊
1
,
𝑡
,
dV
t
	​

=κ(v
∞
	​

−V
t
	​

)dt+η
V
t
	​

	​

dW
1,t
	​

,

where

𝜅
≥
0
κ≥0 is the mean-reversion speed,

𝑣
∞
≥
0
v
∞
	​

≥0 is the long-run variance,

𝜂
≥
0
η≥0 is the vol-of-vol parameter.

Rough Volatility Models
Rough Heston

The rough Heston model modifies the Heston dynamics by incorporating fractional behavior. The variance process is defined as:

𝑉
𝑡
=
𝜉
0
(
𝑡
)
+
𝜈
Γ
 ⁣
(
𝐻
+
1
2
)
∫
0
𝑡
(
𝑡
−
𝑠
)
𝐻
−
1
2
𝑉
𝑠
 
𝑑
𝑊
1
,
𝑠
,
𝑡
≥
0
,
V
t
	​

=ξ
0
	​

(t)+
Γ(H+
2
1
	​

)
ν
	​

∫
0
t
	​

(t−s)
H−
2
1
	​

V
s
	​

	​

dW
1,s
	​

,t≥0,

where

𝐻
∈
(
0
,
1
/
2
)
H∈(0,1/2) is the Hurst parameter driving the roughness,

𝜈
≥
0
ν≥0 controls volatility-of-volatility.

The forward variance curve 
𝜉
0
(
𝑡
)
ξ
0
	​

(t) is:

𝜉
0
(
𝑡
)
=
𝑉
0
+
1
Γ
 ⁣
(
𝐻
+
1
2
)
∫
0
𝑡
(
𝑡
−
𝑠
)
𝐻
−
1
2
𝜃
(
𝑠
)
 
𝑑
𝑠
,
ξ
0
	​

(t)=V
0
	​

+
Γ(H+
2
1
	​

)
1
	​

∫
0
t
	​

(t−s)
H−
2
1
	​

θ(s)ds,

with 
𝜃
(
𝑡
)
 
𝑑
𝑡
+
𝑉
0
𝐿
(
𝑑
𝑡
)
θ(t)dt+V
0
	​

L(dt) defining a non-negative measure
and

𝐿
(
𝑑
𝑡
)
=
Γ
(
1
/
2
−
𝐻
)
−
1
𝑡
−
𝐻
−
1
2
𝑑
𝑡
.
L(dt)=Γ(1/2−H)
−1
t
−H−
2
1
	​

dt.
Rough Bergomi

The rough Bergomi model takes the form:

𝑉
𝑡
=
𝜉
0
(
𝑡
)
 
exp
⁡
(
𝜂
2
𝐻
 ⁣
∫
0
𝑡
(
𝑡
−
𝑠
)
𝐻
−
1
2
𝑑
𝑊
1
,
𝑠
  
−
  
𝜂
2
2
𝑡
2
𝐻
)
,
𝑡
≥
0
,
V
t
	​

=ξ
0
	​

(t)exp(η
2H
	​

∫
0
t
	​

(t−s)
H−
2
1
	​

dW
1,s
	​

−
2
η
2
	​

t
2H
),t≥0,

with 
𝐻
∈
(
0
,
1
/
2
)
H∈(0,1/2), 
𝜂
≥
0
η≥0, and a positive forward variance curve 
𝜉
0
(
𝑡
)
ξ
0
	​

(t).

Extended Rough Bergomi

A more flexible extension expresses 
𝑉
𝑡
V
t
	​

 as the product of two independent factors:

𝑉
𝑡
=
𝜉
0
(
𝑡
)
𝑉
1
,
𝑡
𝑉
2
,
𝑡
,
V
t
	​

=ξ
0
	​

(t)V
1,t
	​

V
2,t
	​

,

where the factors follow:

𝑉
1
,
𝑡
=
exp
⁡
 ⁣
(
𝜁
2
𝛼
+
1
∫
0
𝑡
(
𝑡
−
𝑠
)
𝛼
𝑑
𝑊
1
,
𝑠
  
−
  
𝜁
2
2
𝑡
2
𝛼
+
1
)
,
V
1,t
	​

=exp(ζ
2α+1
	​

∫
0
t
	​

(t−s)
α
dW
1,s
	​

−
2
ζ
2
	​

t
2α+1
),
𝑉
2
,
𝑡
=
exp
⁡
 ⁣
(
𝜆
2
𝛽
+
1
∫
0
𝑡
(
𝑡
−
𝑠
)
𝛽
𝑑
𝑊
2
,
𝑠
  
−
  
𝜆
2
2
𝑡
2
𝛽
+
1
)
,
V
2,t
	​

=exp(λ
2β+1
	​

∫
0
t
	​

(t−s)
β
dW
2,s
	​

−
2
λ
2
	​

t
2β+1
),

with

𝛼
,
𝛽
∈
(
−
1
/
2
,
1
/
2
)
α,β∈(−1/2,1/2),

𝑊
1
W
1
	​

 and 
𝑊
2
W
2
	​

 independent,

𝜁
,
𝜆
∈
𝑅
ζ,λ∈R.

To simplify correlations in simulations, we use the re-parameterization:

𝜌
=
𝜆
𝜁
2
+
𝜆
2
,
𝜂
=
𝜁
2
+
𝜆
2
.
ρ=
ζ
2
+λ
2
	​

λ
	​

,η=
ζ
2
+λ
2
	​

.
What the Code Provides
✔ Pricing Algorithms

The repository contains implementations for pricing European calls and puts using Monte-Carlo simulation under all volatility models listed above.

✔ Dataset Generation

Scripts are provided for generating large datasets of option prices across model parameters. These datasets can be used to train deep neural-network surrogates.

✔ Neural-Network Approximations

Neural networks can be trained to learn the mapping

(
model parameters
,
𝑆
𝑡
,
𝐾
,
𝑇
)
  
↦
  
option price
,
(model parameters,S
t
	​

,K,T)↦option price,

allowing extremely fast inference—volatility surfaces can be generated in milliseconds.

✔ Interfaces for Multiple Languages

Code examples are included for working with the models or neural networks in:

Python

MATLAB

R

✔ Network Weights

Pre-trained neural-network weights can be stored externally (large datasets are not included in the repo to keep size manageable).

Speed and Accuracy

Neural-network surrogates can compute an entire implied-volatility surface in roughly 1 ms on a standard laptop.
Typical SPX option calibrations can be completed in under 1 second, enabling real-time model calibration.

For example, under the rough Bergomi specification with

𝐻
=
0.1
,
𝜂
=
2.1
,
𝜌
=
−
0.9
,
𝜉
0
(
𝑡
)
=
0.15
2
,
H=0.1,η=2.1,ρ=−0.9,ξ
0
	​

(t)=0.15
2
,

the neural network reproduces prices with high fidelity.
A detailed error analysis can be found in the cited research papers.

Getting Started

Several example scripts demonstrate how to run simulations, evaluate the models, and run neural-network approximations.

Language	Folder	Description
MATLAB	get_started/neural_networks_in_matlab	Examples using neural-network pricers in MATLAB
Python	get_started/neural_networks_in_python	Python examples for neural-network pricing
R	get_started/neural_networks_in_R	R examples for network-based pricing
MATLAB	get_started/models_in_matlab	MATLAB examples for the core stochastic-volatility models

Notes:

MATLAB code was developed on version 2019a; Python on 3.7.1; R on 3.4.3.

Neural-network implementations are optimized for speed in MATLAB, but all implementations are fast.

Main References

Römer, S.E., Empirical analysis of rough and classical stochastic volatility models applied to SPX and VIX, Quantitative Finance, 2022.

Heston, S.L., A closed-form solution for options with stochastic volatility, Review of Financial Studies, 1993.

El Euch, O., Gatheral, J., Rosenbaum, M., Roughening Heston, Risk Magazine, 2019.

Bayer, C., Friz, P., Gatheral, J., Pricing under rough volatility, Quantitative Finance, 2016.

Horvath, B., Muguruza, A., Tomas, M., Deep learning volatility: option pricing and calibration in rough models, Quantitative Finance, 2021.
