

# Scripts for using NDSVAE package

## train_model.py

`train_model.py` is a script for training NDSVAE models. Usage:

```
train_model.py [-t THREADS] config_file
```

Before running the script, activate the environment created from [env.yaml](../env.yaml) in the parent directory and make sure that `ndsvae` package is in your Python path.
```
mamba activate py311tf
export PYTHONPATH=$PYTHOPATH:<PATH_TO_NDSVAE>
```

The script expect a single argument - a YAML configuration file with following elements.

- `dataset` *(str)*: Path to the npz file with the following elements:
    - `t`: Time. Shape: `(nt)`.
    - `y`: Observations. Shape: `(nsubjects, nregions, nobs, nt)`.
    - `w`: Structural connectivity. Shape: `(nsubjects, nregions, nregions)`.
    - `name`: Name of the dataset (optional).
    - `description`: Description of the dataset (optional).
    
        Additionaly, several other elements can be included, typically if the dataset comes from a simulation with known ground truth values. These are not used for model training.
    - `x`: Source activity (optional). Shape: `(nsubjects, nregions, nstates, nt)`.
    - `thetareg`: Regional parameters (optional). Shape: `(nsubjects, nregions, mreg)`.
    - `thetasub`: Subject parameters (optional). Shape: `(nsubjects, msub)`.
    
- `model`
    - `ns` *(int)*: State space size. 
    - `msub` *(int)*: Number of subject-specific parameters.
    - `mreg` *(int)*: Number of region-specific parameters.
    - `f_units` *(int)*: Number of units in the dynamical model.    
    - `encoder_units` *(int)*: Number of units in the encoder. Applies both to the parameter encoder and the state encoder.
    - `shared_input` *(bool)*: Switch for models with/without shared input.
    - `lambda_x` *(float)*: Regularization term $\lambda_x$ for state variables.
    - `lambda_A` *(float)*: Regularization term $\lambda_A$ for forward projection.
    - `alpha` *(float)*: Regularization term $\alpha$ for dynamical model.

- `training`
    - `seed` *(int)*: Seed for random number generator.
    - `learning_rate` *(float or list(float))*: Learning rate. If list is given, `lr_boundaries` must be present.
    - `lr_boundaries` *(list(float), optional)*: Epochs at which learning rates change according to `learning_rate` list. Must be one element shorter than `learning_rate`.
    - `batch_size` *(int)*: Batch size.
    - `epochs` *(int)*: Number of epochs.
    - `nsamples` *(int)*: Number of samples to calculate expectations from approximate posterior.
    - `clip_gradient` *(float, optional)*: Gradient clipping coefficient.
    - `betax` *(float or (int,int,float,float), optional)*. $\beta_x$ coefficient. Either a constant, or 4 element specification of clipped linear growth: `(epoch_from, epoch_to, start_value, end_value)`. Default = 1.
    - `betap` *(float or (int,int,float,float), optional)*. $\beta_p$ coefficient. Either a constant, or 4 element specification of clipped linear growth: `(epoch_from, epoch_to, start_value, end_value)`. Default = 1.
    - `test_ratio` *(float)*: Ratio of test datapoints to all datapoints, between 0 and 1.

- `output`
    - `output_dir` *(str)*: Output directory for created images and weights.
    - `log` *(str, optional)*: Log file. If absent, stdout is used.
    - `save_interval` *(int)*: Interval for plotting figures and saving weights and parameters.
    - `example_subjects` *(list(int))*: Indices of example subjects to be plotted. 
    - `example_regions` *(list((int,int)))*: Indices of example subjects and regions to be plotted.


Example configuration file `config.yaml`:
```
dataset: "data.npz"

model:
    ns: 5
    msub: 2
    mreg: 3
    f_units: 32
    encoder_units: 32
    shared_input: True
    lambda_x: 0.01
    alpha: 0.01

training:
    seed: 42
    learning_rate: 0.001
    batch_size: 64
    epochs: 1000
    nsamples: 8
    clip_gradients: 1000.
    betax: 1.
    betap: [0, 500, 0., 1.]
    test_ratio: 0.1

output:
    output_dir: "run/"
    log: "run/log.txt"
    save_interval: 50
    example_subjects: [0,1]
    example_regions: [[0,10], [1,10]]
```

