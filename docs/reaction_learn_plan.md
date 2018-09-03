# Frank's changes
Title: Maybe "Learning Reaction Mechanisms from Data" or so

### Method:
- [x] rename section to be more specific, such as sparse learning of reaction kinetics
- [ ] I would set up the section differently. Start with a general formulation of mass-action reaction kinetics. That leads to a system of ODEs where the change of the concentrations are given by a sum of terms that contain constants (order-0 reactions), the concentrations (order-1 reactions), the products of concentrations (order-2 reactions) etc. Then introduce basis functions for reactions by substituting concentrations and products of concentrations with basis functions. Then turn into a sparse linear regression problem and refer to SINDy.
The current formulation is purely mathematical and it's not obvious that this formulation is physically meaningful.

### Example System:
- [x] move to results section
- [x] regulation should be part of the set of equations defining the system (p. 4, top)

### Results:
- [x] don't give table-of-content like previews of what's going to come. A paper is not a book, the abstract/intro are already giving an overview of the contents. You don't need to do that chapter-wise. If you wanna do a reminder, restrict to 1 sentence.
- [ ] 4.1: noiseless but there is still a certain amount of data used and starting from a certain initial point, using a certain number of time steps etc. This needs to be specified, because the results will depend on it.
- [ ] 4.1: You say for certain settings you get the right result. But can we find the right result without knowing it, i.e. by cross-validation or another systematic approach?
- [ ] 4.2/Fig. 3:
    - [x] Use panel labeling (a/b) in all Figures.
    - [x] I don't like "failure rate". May just "number of spurious reactions"?
    - [x] Fig 3 panel 2: Use Estimation error in the y-Axis (you can use "estimation error |xi-^xi|"), then the legend becomes simpler.
    - [x] Fig 3: put legend into Figure space.
    - [ ] What does number of measurements mean? I guess you are measuring at different time points. So is this the number of complete trajectories at a fixed noise level? This should correspond to a certain effective noise level at each time point (noise variance divided by number of measurements at that point), which may be the more interesting quantity because it's transferable.
- [ ] 4.3/Fig. 4:
    - [ ] need to describe how the initial conditions were generated
    - [x] Fig. 4a is ugly.
    - [x] Fig. 4a will be more clear if you also show the noise-free trajectory from that initial condition, otherwise it just looks messy.
    - [x] Fig. 4: design as one-column figure.



### SCHEME
- [x] plot exemplaratory concentration curve for case 1, 2 [chris]
- [x] make mRNA dashed

### Case 1
- [x] LSQ fits well but doesnt necessarily recover the rates (varying $\alpha$)
  - [x] LSQ fits well (qualitatively, see notebook)
  - [x] LSQ doesnt necessarily recover the rates
  - [x] LSQ + L1 can (almost) recover the rates
  - [x] describe this in manuscript
- [x] use more basis functions (perhaps not needed)
- [x] plot
  - [x] bubbles

### Case 2
- [x] Same initial conditions, multiple measurements, noise is a variable.  -> in the limit of small noise you get a good fit, 
- [x] obtain cutoff in noise free case (or case 1) -> compute failure rate, i.e., false positives and negatives
- [x] $\| \Xi - \Xi_\mathrm{est} \|_1$ with cutoff and include $0$ noise as case in the plot
- [x] plot
  - [x] convergence
  - [x] failure rate
  - [x] PANELIZE [the hoff]
- [x] add CV parameters to case_config.py [chris]
- [x] würger Chris: LSQ durchlösen für Realisierungen, failure rate auswerten (da LSQ vermutlich unreliable)

### Case 3
- [x] Extend to multiple initial conditions. (this probably works better)    
- [x] also compute failure rate
- [x] Plöt panelized:
  - [x] Plot first and second initial condition trajectory concatenated ... bla
  - [x] Plot convergence in L1
  - [x] plot failure rate
  - [x] PANELIZE [moritz]
- [x] add CV parameters to case_config.py [chris]

### General
- [x] Minimization function: use normal Frobenius norm, stack all samples of $X$ and $\dot{X}$
- [x] Try representation of reactions as a matrix with dots:
  - matrix of species and pairs -> ring for estimated -> dot for reference
  - look at matrices in [1]
- [x] Consider rescaling of rates to a unitless or normalized quantity (Mohsen’s idea):
  - rescale rates w.r.t. equilibrium concentration [can do that] $\dot{a} = bc \xi\quad\rightarrow\quad\dot{a}/\tilde{a} = bc\xi/\tilde{a}$
- [x] explain why minimizing instead of closed form $A^{-1}$ (dimensions blow up, need to use sparse solvers etc) -> we just use the objective function
- [x] ~~cross~~ validation: use different noise instances for validation, this suffices as our samples are i.i.d.

[1] M. Thattai and A. van Oudenaarden, “Intrinsic noise in gene regulatory networks,” Proc. Natl. Acad. Sci., vol. 98, no. 15, pp. 8614–8619, 2001.
