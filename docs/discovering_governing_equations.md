Recovering model from RD-data. TOC:

[[toc]]


# Ideas in the context of reaction diffusion:
- Discretize space into (small) boxes
- thus find small fluctuations in concentration
- we also have to take into account the ($$\tau$$ lagged) time series, yielding blocks of $$N_\text{boxes} \times \tau$$ i think...
- fit the data onto a model with all possible reactions, penalizing small values (rates etc)
- Trajektorien (dh #Partikel pro Box) wie MSMs behandeln $$X_t \rightarrow S_t \rightarrow P(\tau)$$
- Bei ReaDDy Trajektorien: Alle Species nehmen und mögliche Reaktionen dazu aufstellen:
  - Bspw: Species $$A,B,C$$ yield
    - $$\{ A,B,C\}\rightarrow \{ A,B,C \}$$ (without self-reactions)
    - $$\{A,B,C\}+\{A,B,C\}\rightarrow\{A,B,C\}$$


Propensity functions:
- $$A+A\rightarrow_{k}C$$ has $$\alpha(t)=A(t)(A(t)-1)k$$
- $$A+B\rightarrow_kD$$ has $$\alpha(t)=A(t)B(t)k$$

Goals
- Which reactions?
- Which rates corresponding to the reactions?
- Which diffusion constants?

# Papers

## Discovering governing equations from data by sparse identification of nonlinear dynamical systems 
- _goal_: discover governing equations from noisy measurement data
- combine sparsity-promoting techniques and machine learning with nonlinear dynamical systems
- use sparse regression

_SINDy_: Sparse identification of nonlinear dynamics
$$$
\frac{d\mathbf{x(t)}}{dt} = \mathbf{f}(\mathbf{x(t)})
$$$
- the function $$\mathbf{f}$$ usually only consists of a few terms, i.e., sparse in the space of possible functions
- to determine $$\mathbf{f}$$, collect a time history of the state $$\mathbf{x}(t)$$ and either measure $$\dot{\mathbf{x}}(t)$$ or approximate it numerically - the sample $$t_1, t_2, \ldots, t_m$$ and arrange it into matrices

$$$
\mathbf{X} = \begin{pmatrix}\mathbf{x}^T(t_1) \\ \vdots \\ \mathbf{x}^T(t_m)\end{pmatrix} = \begin{pmatrix} x_1(t_1) & \cdots & x_n(t_1)\\ \vdots & \vdots & \vdots \\ x_1(t_m) & \cdots & x_n(t_m) \end{pmatrix},\quad \dot{\mathbf{X}} = \begin{pmatrix}\dot\mathbf{x}^T(t_1) \\ \vdots \\ \dot\mathbf{x}^T(t_m)\end{pmatrix}
$$$

- construct a library $$\Theta(\mathbf{X})$$ of candidate functions of the columns of $$\mathbf{X}$$ - example:

$$$
\Theta(\mathbf{X}) = \begin{pmatrix} 1 & \mathbf{X} & \mathbf{X}^{P_2} & \cdots & \sin(\mathbf{X}) & \cos(\mathbf{X}) & \cdots \end{pmatrix},
$$$
where each function acts on the columns of $$\mathbf{X}$$.

# The method
## Single Box
- collect particle counts per species, write down reactions in CME style, e.g.,

$$$
\dot x = \begin{pmatrix}\dot x_A \\ \dot x_B \\ \dot x_C \end{pmatrix} = \begin{pmatrix} ... \end{pmatrix}
$$$

Use these functions as $$\theta_i$$ in $$\dot X = \Theta(X)\Xi$$.

# Examples
The space is discretized into $$m$$ pairwise disjoint boxes $$\Omega = \bigcup_{i=1}^m Q_i$$. The boxes should be chosen such that there is usually only one reaction at a time (so rather small). 

## For one box containing three species with occasional reactions
We have $$m=1$$, $$N=3$$ species and a corresponding ("continuous") trajectory
$$$
\mathbf{x}(t) = \left(A_i(t)\right)_{i=1}^N\in\mathbb{R}^N,
$$$
where $$A_i(t)$$ is the number of particles of species $$i$$ at time $$t$$ and a simulation time step of $$\Delta t$$.

Discretizing the trajectory: We have for the present species a set of possible reactions (unary and binary). By comparing $$\mathbf{x}(t)$$ and $$\mathbf{x}(t-\Delta t)$$ one can figure out the reaction (and thus the state) that has happened in the particular box.

This yields a discrete trajectory $$t\mapsto S(t)$$. In the case of multiple boxes we get multiple of these discrete trajectories plus diffusive reactions, as in one particle disappears in one box and appears out of nowhere in the other box.

- (i dont know about this) These discrete trajs can now be aggregated (n reactions of type xyz in time step t plus the diffusive stuff).
- check up on $$\tau$$ lagged data, linear regression, count matrix (but no changes in state)

# Known issues and limitations
## CME case
- only works when well mixed
- one has to take care of the time scales which are fitted: If one educt relevant species is depleted, further fitting will decrease the rate
  - perhaps remove respective ansatz functions from LASSO loop

## RDME case  

# Todos
- Use second derivative w.r.t time to obtain only the change in particle numbers. I.e. find $$\xi$$ according to 

$$$
\ddot{x} = \theta(x, \dot{x}) \xi
$$$