To reproduce the experiments in the paper:

1) Run the file mc_sampler_and_MCMC.py. This uses the optimal sampler to generate samples from the null manifold, then uses PyMC to generate 100,000 MCMC samples repeated across 10 random seeds.

2) Run the file main.py. This runs the OBBT routine.

3) Finally, plot the results with plot_results_1D_with_MCMC.py