
import numpy as np
import tensorflow as tf

from . import models, training

def create_model_with_dataset(params, dataset):
    """Create a model from model configuration and specific dataset.
    
    Parameters
    ----------
    params: dict
        Configuration of the model
    dataset: ndsvae.Dataset
        Dataset to take the dimensions from.

    Returns
    -------
    ndsv.models.RegX
        Created model
    """

    return create_model(params, dataset.nsub, dataset.nreg, dataset.nt, dataset.nobs)


def create_model(params, nsub, nreg, nt, nobs):
    """Create a model from model configuration.
    
    Parameters
    ----------
    params: dict
        Configuration of the model
    nsub: int
        Number of subjects
    nreg: int
        Number of regions
    nt: int
        Number of timepoints
    nobs: int
        Number of observed variables

    Returns
    -------
    ndsv.models.RegX
        Created model
    """

    params.update(dict(nsub=nsub, nreg=nreg, nt=nt, nobs=nobs))
    model = models.RegX(**params)
    model.build(input_shape=None)
    return model


def get_example_data(dataset, model, examples):
    nsub, nreg, _, _ = dataset.y.shape
    tds = training._prep_training_dataset(dataset, batch_size=1, mode='region-upsampled',
                                          upsample_factor=model.upsample_factor, shuffle=False)
    tds = list(tds.as_numpy_iterator())
    data = []
    for isub, ireg in examples:
        data.append(tds[isub*nreg + ireg])
    return data


def get_training_mask(dataset, test_ratio):

    train_ratio = 1. - test_ratio
    ndata = int(train_ratio * dataset.nreg * dataset.nsub)

    train_mask = np.zeros((dataset.nsub, dataset.nreg), dtype=bool)
    train_mask[np.unravel_index(np.random.choice(dataset.nsub*dataset.nreg, ndata, replace=False),
                                (dataset.nsub, dataset.nreg))] = True
    
    return train_mask