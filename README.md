# Bounding the Null Space: Interval-Based Uncertainty Quantification for Non-Identifiable Groundwater Models

**Research software accompanying the preprint**

**Authors:** Maximilian Ramgraber (Delft University of Technology) and Ksenia Bestuzheva (Zuse Institute Berlin)

**Preprint:** https://doi.org/10.48550/arXiv.2606.10836

**Source repository:** https://github.com/MaxRamgraber/Bounding-The-Null-Space

**Contact:** m.ramgraber@tudelft.nl

## Purpose

The scripts in this release accompany the paper *Bounding the Null Space: Interval-Based Uncertainty Quantification for Non-Identifiable Groundwater Models*. They support numerical experiments on conservative uncertainty quantification for non-identifiable groundwater-flow models. Rather than estimating variable ranges from a finite sample of model realizations, the method uses optimization-based bound tightening (OBBT) to compute outer bounds consistent with specified parameter limits, groundwater-flow equations, and observations.

The method represents groundwater flow using finite-volume discretizations of Darcy's law, relaxes bilinear products using McCormick inequalities, and applies linear programming to tighten intervals. The examples examine the impact of relaxation-induced nonphysical flow and the use of additional constraints, including prescribed flow directions and irrotationality, to improve bound tightening.

## Repository contents

- `experiments/`: Python scripts for the three numerical examples in the paper.
  - **1D steady-state groundwater flow:** illustrates interval tightening of hydraulic heads, flows, recharge and transmissivity, with comparisons against sampling-based uncertainty estimates.
  - **2D steady-state groundwater flow:** compares a reference configuration, basic OBBT, irrotationality constraints, and prescribed flow signs.
  - **2D transient groundwater flow:** demonstrates OBBT for a time-dependent model on a hexagonal grid.
- `figures/`: manuscript figure files and Python scripts used to produce figures.
- `README.md`: this document.
- `LICENSE`: the software license; include this file in the final deposited archive.

Preserve the folder structure when extracting the archive.

## Using the software

The software consists of Python scripts rather than a packaged Python library. The experimental scripts are organized in subfolders under `experiments/`, and figure-generation scripts are under `figures/`. The accompanying manuscript describes each model configuration and the meaning of the results.

The OBBT implementation described in the manuscript uses `scipy.optimize.linprog` and the HiGHS linear-programming solver. Other Python dependencies vary by script and should be checked against the imports in the files before running each example. For repeatable execution, document and retain the Python version and dependency versions used for the deposited release.

## Scope and interpretation

These are numerical research examples, not field observations from a particular geographic location. The OBBT intervals are conservative outer bounds for the stated constraints and relaxations; the paper discusses when McCormick relaxations require additional physics-informed constraints to provide informative bounds.

## Citation

If you use the software, please cite the associated paper:

Ramgraber, M., and Bestuzheva, K. (2026). *Bounding the Null Space: Interval-Based Uncertainty Quantification for Non-Identifiable Groundwater Models*. arXiv:2606.10836. https://doi.org/10.48550/arXiv.2606.10836

Once the 4TU.ResearchData software record has been published, please also cite its assigned 4TU DOI for the archived software version.

## License

The software is made available under the MIT License; see the `LICENSE` file in the archive. This statement assumes the rights holders have approved MIT licensing and that any included third-party code can be redistributed under compatible terms.
