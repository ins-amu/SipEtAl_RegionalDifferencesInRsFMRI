

# HCP-proc

Preprocessing of Human Connectome Project data for use in whole-brain modeling.



### Goals

  - Build the structural connectome 
  - Extract the parcellated resting state time series 

Parcellation used is Desikan-Killiany (`aparc+aseg` from FreeSurfer), and both the connectome and the resting state time series are built for all cortical and subcortical regions (`res_all/` directory in the generated data) and for cortical regions only (`res_ctx/` directory in the generated data).


### Prerequisities

1. HCP data (available from [db.humanconnectome.org])
   - "Structural Preprocessed" in `data/Structural` directory
   - "Resting State fMRI FIX-Denoised (Compact)" in `data/rfMRI` directory
   - "Diffusion Preprocessed" in `data/Diffusion` directory

2. Software
   - Connectome workbench
   - MRtrix3
   - Python 3.7 with standard (neuro-)scientific packages

### Usage

1. Get the data and the software
2. From the working directory call `snakemake sc fmri`. Test it with a dry-run first (`-n` switch), and make sure you have sufficient computational resources for the tractography.
   
