This folder contains the files necessary to repeat the 2D Steady-State experiment. To reproduce it:

1) First, run the following files. Each generates the results for of one of the solutions:
	- main_pure_LP.py
	- main_base.py
 	- main_nocurl.py
	- main_flow_sign.py

2) Plot the results by running plot_results_2D_rectangular_v15.py. This script reads the checkpoint files generated in step 1, as well as pure_LP_bounds.p, and produces experiment_2D_steady_state.pdf and experiment_2D_steady_state.png.
