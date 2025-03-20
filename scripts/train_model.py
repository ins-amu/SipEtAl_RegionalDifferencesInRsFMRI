# -*- coding: utf-8 -*-

import os
import argparse
import yaml
import sys

import numpy as np
import tensorflow as tf

import ndsvae as ndsv


# Parse arguments
parser = argparse.ArgumentParser(description="Train the NDSVAE model.")
parser.add_argument('config_file', help="Configuration file", type=str)
parser.add_argument('-t', '--threads', type=int, default=1)
args = parser.parse_args()

# Set threads
print(f"Running with {args.threads} threads")
tf.config.threading.set_inter_op_parallelism_threads(args.threads)
tf.config.threading.set_intra_op_parallelism_threads(args.threads)

# Read the configuration file
with open(args.config_file, 'r') as fh:
    config = yaml.safe_load(fh)

# Set seeds
np.random.seed(config['training']['seed'])
tf.random.set_seed(config['training']['seed'])

# Load the dataset
dataset = ndsv.Dataset.from_file(config['dataset'], to_dtype=np.float32)

# Prepare output directories
outdir = config['output']['output_dir']
os.makedirs(outdir)
os.makedirs(os.path.join(outdir, "img"))
os.makedirs(os.path.join(outdir, "models"))

# Get and save training mask
train_mask = ndsv.util.get_training_mask(dataset, config['training']['test_ratio'])
np.save(f"{outdir}/train_mask.npy", train_mask)

# Create the model
model = ndsv.util.create_model(config['model'], dataset)

# Create the training runner
runner = ndsv.training.create_runner(config['training'], dataset)

# Define the saving/plotting callback
example_subjects = config['output']['example_subjects']
example_regions = config['output']['example_regions']
example_data = ndsv.util.get_example_data(dataset, model, example_regions)

def plot_and_save(state, model):
    if state.epoch % config['output']['save_interval'] != 0:
        return

    ndsv.viz.plot_params_reg(state, model, dataset, None, f"{outdir}/img/params-reg_{state.epoch:05d}.png", subj_inds=example_subjects)
    ndsv.viz.plot_params_sub(state, model, dataset, None, f"{outdir}/img/params-sub_{state.epoch:05d}.png")
    ndsv.viz.plot_simulation(state, model, dataset, example_subjects, f"{outdir}/img/pred_{state.epoch:05d}.png")
    ndsv.viz.plot_input(state, model, dataset, example_subjects, f"{outdir}/img/input_{state.epoch:05d}.png")
    ndsv.viz.plot_projection(state, model, example_data, f"{outdir}/img/proj_{state.epoch:05d}.png")

    params = model.encode_subjects(dataset.w, dataset.y, subject_batch_size=1)
    params.save(f"{outdir}/img/params_{state.epoch:05d}.npz")
    model.save_weights(f"{outdir}/models/model_{state.epoch:05d}.weights.h5")


# Get log file handle
logfile = config['output'].get('log', None)
fh = sys.stdout if logfile is None else open(logfile, 'w')

# Run the training
hist = ndsv.training.train(model, dataset, runner, fh=fh, callback=plot_and_save, mask_train=train_mask)

# Save the history
hist.as_dataframe().to_csv(os.path.join(outdir, "hist.csv"), index=False)

# Save the model
model.save_weights(os.path.join(outdir, "model.weights.h5"))

# Cleanup
if fh is not sys.stdout:
    fh.close()
