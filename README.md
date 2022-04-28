

# Parameter inference on brain network models with unknown node dynamics and spatial heterogeneity


This repository contains the code used in [1].


### Structure

- Folder `hcp-proc/` contains the code needed for processing of Human Connectome Project data (in order to get the structural connectome and resting state fMRI, both in Desikan-Killiany parcellation).
- Folder `ndsvae/` contains the code for the inference method.
- Folder `study/` contains the scripts and notebooks for the simulations, inferences, and postprocessing. Entry point is the `Snakefile`.


### Environment

Python 3.8 with standard scientific and neuroscientific libraries is necessary. Use the environment file `study/env.yml` to prepare the conda environment.


### License

This work is licensed under MIT license. See LICENSE.txt for the full text.


### References

[1] Viktor Sip, Spase Petkoski, Meysam Hashemi, Timo Dickscheid, Katrin Amunts, Viktor Jirsa. Parameter inference on brain network models with unknown node dynamics and spatial heterogeneity. bioRxiv 2021.09.01.458521; doi: https://doi.org/10.1101/2021.09.01.458521
