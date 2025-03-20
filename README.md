

# Characterization of regional differences in resting-state fMRI with a data-driven network model of brain dynamics


This repository contains the code used in [1].


### Structure

- Folder `hcp-proc/` contains the code needed for processing of Human Connectome Project data (in order to get the structural connectome and resting state fMRI, both in Desikan-Killiany parcellation).
- Folder `ndsvae/` contains the code for the inference method.
- Folder `scripts/` contains simplified scripts for using the method.
- Folder `study/` contains the scripts and notebooks for the simulations, inferences, and postprocessing used in [1]. Entry point is the `Snakefile`.


### Environment

Python 3.11 with standard scientific and neuroscientific libraries is necessary. Use the environment file `env.yaml` to prepare the conda environment.


### License

This work is licensed under MIT license. See LICENSE.txt for the full text.


### References

[1] Viktor Sip, Meysam Hashemi, Timo Dickscheid, Katrin Amunts, Spase Petkoski, Viktor Jirsa. Characterization of regional differences in resting-state fMRI with a data-driven network model of brain dynamics. Science Advances 9, eabq7547(2023). doi:[10.1126/sciadv.abq7547](https://doi.org/10.1126/sciadv.abq7547)
