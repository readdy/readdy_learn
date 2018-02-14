### Case 1
- [ ] LSQ fits well but doesnt necessarily recover the rates (varying $\alpha$)
  - [x] LSQ fits well (qualitatively, see notebook)
  - [x] LSQ doesnt necessarily recover the rates
  - [x] LSQ + L1 can (almost) recover the rates
  - [ ] describe this in manuscript
- [x] use more basis functions (perhaps not needed)
- See Notebook `the_cases/case_1_no_noise_vary_alpha.ipynb`

### Case 2
- [ ] Same initial conditions, multiple measurements, noise is a variable.  -> in the limit of small noise you get a good fit, 
- [ ] obtain cutoff in noise free case (or case 1) -> compute failure rate, i.e., false positives and negatives
- [ ] $\| \Xi - \Xi_\mathrm{est} \|_1$ with cutoff and include $0$ noise as case in the plot

### Case 3
- [ ] Extend to multiple initial conditions. (this probably works better)    
- [ ] also compute failure rate

### General
- [x] Minimization function: use normal Frobenius norm, stack all samples of $X$ and $\dot{X}$
- [x] Try representation of reactions as a matrix with dots:
  - matrix of species and pairs -> ring for estimated -> dot for reference
  - look at matrices in [1]
- [ ] Consider rescaling of rates to a unitless or normalized quantity (Mohsen’s idea):
  - rescale rates w.r.t. equilibrium concentration [can do that] $\dot{a} = bc \xi\quad\rightarrow\quad\dot{a}/\tilde{a} = bc\xi/\tilde{a}$
- [ ] explain why minimizing instead of closed form $A^{-1}$ (dimensions blow up, need to use sparse solvers etc) -> we just use the objective function
- [ ] ~~cross~~ validation: use different noise instances for validation, this suffices as our samples are i.i.d.

[1] M. Thattai and A. van Oudenaarden, “Intrinsic noise in gene regulatory networks,” Proc. Natl. Acad. Sci., vol. 98, no. 15, pp. 8614–8619, 2001.