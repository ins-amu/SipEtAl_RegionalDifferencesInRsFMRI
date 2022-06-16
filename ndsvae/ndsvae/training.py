
import tensorflow as tf
import numpy as np
import pandas as pd
from scipy import interpolate


from .Dataset import get_network_input_obs

def linclip(i1, i2, v1, v2):
    """
    Return function to calculate beta for clipped linear growth
    """

    def beta(i):
        a = (v2 - v1)/(i2 - i1)
        b = (i2*v1 - i1*v2)/(i2 - i1)
        return np.clip(a*i+b, min(v1, v2), max(v1, v2))

    return beta

def irregsaw(lst, default=1.0):
    def beta(i):
        for (ifr, ito, valfr, valto) in lst:
            if (i >= ifr) and (i <= ito):
                a = (valto - valfr)/(ito - ifr)
                b = (ito*valfr - ifr*valto)/(ito - ifr)
                return a*i + b
        return default

    return beta


class Runner:
    """
    Class to hold the info about the optimization process
    """

    def __init__(self, batch_size, optimizer, epochs, clip_gradients=None, nsamples=8, betax=1.0, betap=1.0):
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.epochs = epochs
        self.clip_gradients = clip_gradients
        self.nsamples = nsamples
        self.betax = betax if callable(betax) else (lambda e: betax)
        self.betap = betap if callable(betap) else (lambda e: betap)


def _prep_training_dataset(ds, batch_size, mode='region', upsample_factor=None, shuffle=True, mask=None):
    """
    Take dataset and prepare training dataset for a model.

    Two modes are (or are planned to be) supported:
    - 'region',  where each training sample consist of subject id and the region timeseries
    - 'subject', where each training sample consist of all time series
    """

    nsub, nreg, nobs, nt = ds.y.shape
    if mask is None:
        mask = np.ones((nsub, nreg), dtype=bool)

    if mode == 'region':
        isubs = []
        yaugs = []

        iext = get_network_input_obs(ds.w, ds.y, comp=0)

        for i in range(nsub):
            for j in range(nreg):
                if mask[i,j]:
                    yaug = np.vstack([ds.y[i, j, :, :], iext[i, j, :, :]])
                    isubs.append(i)
                    yaugs.append(yaug.T)

        ns = len(isubs)
        isubs = np.array(isubs, dtype=np.int32)
        yaugs = np.array(yaugs, dtype=np.float32)
        full_dataset = tf.data.Dataset.from_tensor_slices((isubs, yaugs))
        if shuffle:
            full_dataset = full_dataset.shuffle(ns)
        training_dataset = full_dataset.batch(batch_size)
        return training_dataset

    elif mode == 'region-upsampled':
        nsub, nreg, nobs, nt = ds.y.shape

        mask = np.reshape(mask, (nsub*nreg))
        subj_ind = np.repeat(np.r_[:nsub], nreg)
        yobs = np.reshape(np.swapaxes(ds.y, 2, 3), (nsub*nreg, nt, nobs))
        iext = np.reshape(get_network_input_obs(ds.w, ds.y, comp=0)[:,:,0,:], (nsub*nreg, nt))
        finterp = interpolate.interp1d(np.linspace(0, nt-1, nt), iext, axis=-1,
                                       fill_value=(iext[:,0], iext[:,-1]), bounds_error=False)
        iext_upsampled = finterp(np.linspace(-1./2. + 1./(2*upsample_factor), nt - 1./2. - 1./(2*upsample_factor),
                                            nt*upsample_factor))

        subj_ind = np.array(subj_ind, dtype=np.int32)[mask]
        yobs = np.array(yobs, dtype=np.float32)[mask]
        iext = np.array(iext, dtype=np.float32)[mask]
        iext_upsampled = np.array(iext_upsampled, dtype=np.float32)[mask]

        nsamples = subj_ind.shape[0]
        full_dataset = tf.data.Dataset.from_tensor_slices((subj_ind, yobs, iext, iext_upsampled))
        if shuffle:
            full_dataset = full_dataset.shuffle(nsamples)
        training_dataset = full_dataset.batch(batch_size)

        return training_dataset

    elif mode == 'subject':
        raise NotImplementedError("Subject-based dataset preparation not implemented")

    else:
        raise NotImplementedError(f"Dataset preparation mode {mode} not implemented")


def get_train_step_fn():
    @tf.function
    def train_step(model, training_batch, runner, betax, betap):
        with tf.GradientTape() as tape:
            loss = model.loss(training_batch, nsamples=runner.nsamples, betax=betax, betap=betap)

        gradients = tape.gradient(loss, model.trainable_variables)

        # Remove empty gradients
        trainable_variables = [v for (v,g) in zip(model.trainable_variables, gradients) if g is not None]
        gradients = [g for g in gradients if g is not None]

        if runner.clip_gradients is None:
            runner.optimizer.apply_gradients(zip(gradients, trainable_variables))
        else:
            gradients_clipped = [tf.clip_by_value(grad,
                                                  runner.clip_gradients[0],
                                                  runner.clip_gradients[1])
                                 for grad in gradients]
            runner.optimizer.apply_gradients(zip(gradients_clipped, trainable_variables))

        return loss

    return train_step


class History:
    def __init__(self, model):
        model_names, model_fmts = model.tracked_variables()
        self.names = ["epoch", "loss", "loss_test", "betax", "betap"] + model_names
        self.fmts = ["%6d", "%14.2f", "%14.2f", "%6.3f", "%6.3f"] + model_fmts
        self._hist = []

    def print_header(self, fh):
        fh.write(" ".join(self.names) + "\n")
        fh.flush()

    def add(self, epoch, loss, betax, betap, model, loss_test=0.):
        model_values = model.tracked_variables_values()
        self._hist.append([epoch, loss, loss_test, betax, betap] + model_values)

    def print_last(self, fh):
        line = " ".join(self.fmts) % tuple(self._hist[-1])
        fh.write(line + "\n")
        fh.flush()

    def as_dataframe(self):
        df = pd.DataFrame(self._hist, columns=self.names)
        return df

class State:
    def __init__(self, epoch, loss, betax, betap):
        self.epoch = epoch
        self.loss = loss
        self.betax = betax
        self.betap = betap


def train(model, dataset, runner, fh=None, callback=None, mask_train=None):

    try:
        upsample_factor = model.upsample_factor
    except AttributeError:
        upsample_factor = None

    dataset_train = _prep_training_dataset(dataset, runner.batch_size, model.training_mode, upsample_factor, mask=mask_train)
    dataset_test  = _prep_training_dataset(dataset, runner.batch_size, model.training_mode, upsample_factor, mask=~mask_train)

    hist = History(model)
    if fh:
        hist.print_header(fh)

    train_step = get_train_step_fn()


    for epoch in range(runner.epochs):
        betax = tf.constant(runner.betax(epoch), dtype=tf.float32)
        betap = tf.constant(runner.betap(epoch), dtype=tf.float32)

        loss = tf.keras.metrics.Mean()
        for training_batch in dataset_train:
            loss(train_step(model, training_batch, runner, betax, betap))

        loss_test = tf.keras.metrics.Mean()
        for test_batch in dataset_test:
            loss_test(model.loss(test_batch, nsamples=runner.nsamples, betax=betax, betap=betap))

        hist.add(epoch, loss.result().numpy(), betax.numpy(), betap.numpy(), model, loss_test.result())

        if fh:
            hist.print_last(fh)

        if callback:
            state = State(epoch, loss.result().numpy(), betax.numpy(), betap.numpy())
            callback(state, model)

        tf.keras.backend.clear_session()

    return hist
