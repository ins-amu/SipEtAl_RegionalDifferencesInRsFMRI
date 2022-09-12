import itertools
from collections import OrderedDict

import numpy as np

import tensorflow as tf
from tensorflow import keras

from .Dataset import get_network_input_obs


def log_normal_pdf(sample, mean, logvar, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    lnp = -0.5 * ((sample - mean) ** 2. * tf.exp(-logvar) + logvar + log2pi)
    if raxis is None:
        return lnp
    else:
        return tf.reduce_sum(lnp, axis=raxis)


def gen_normal(mu, logvar):
    """
    Sample from random normal distribution
    """
    eps = tf.random.normal(shape=mu.shape)
    return mu + tf.exp(0.5 * logvar) * eps


def unit_simplex(y):
    """Stick-breaking simplex transformation: y in R^(K-1) -> x [0,1]^K such that sum(x) == 1

    Source: Stan 2.22 documentation.

    Expected y.shape: (K-1, any)
    """

    K = y.shape[0] + 1
    n = y.shape[1]

    k = np.array(np.r_[:K-1], dtype=np.float32)[:,None]
    z = tf.math.sigmoid(y + tf.math.log(1./(K-k-1)))
    initializer = (np.zeros(n, dtype=np.float32), np.zeros(n, dtype=np.float32))
    _, x = tf.scan(lambda xk, zk: (xk[0]*(1.-zk)+zk, (1.-xk[0])*zk), z, initializer)
    x = tf.concat([x, (1.-tf.math.reduce_sum(x, axis=0)[None,:])], axis=0)

    return x


def zerosones(shape, dtype=None):
    """Create array of shape '(shape[0], ..., shape[N], 2)' filled with zeros and ones (differing along the last dim).
    Typical use is for the creation of parameters of standard normal distribution."""

    zeros = np.expand_dims(np.zeros(shape, dtype=dtype), -1)
    ones  = np.expand_dims(np.ones(shape, dtype=dtype), -1)

    return np.concatenate([zeros, ones], axis=-1)


def lpdf_ar1(x, tau, lv_process):
    """
    Log probability density of x under the assumption of AR(1) process with decay time tau
    and process log-variance lv_process.
    """
    alpha = tf.math.exp(-1./tau)
    lv_noise = lv_process + tf.math.log(1-alpha**2)
    logp = (log_normal_pdf(x[:,0], 0., lv_process, raxis=None)
            + log_normal_pdf(x[:,1:], alpha*x[:,:-1], lv_noise, raxis=1))
    return logp   # size: (batch_size)



class GeneratedData:
    """
    Object to hold the simulated time series
    """

    def __init__(self, x, y, thetareg, thetasub, us=None):
        self.x = x
        self.y = y
        self.thetareg = thetareg
        self.thetasub = thetasub
        self.us = us

    def save(self, filename):
        np.savez(filename, x=self.x, y=self.y, thetareg=self.thetareg, thetasub=self.thetasub, us=self.us)

    @classmethod
    def from_file(cls, filename):
        data = np.load(filename)
        try:
            us = data.get('us', None)
        except ValueError:
            us = None

        return cls(x=data['x'], y=data['y'], thetareg=data['thetareg'], thetasub=data['thetasub'], us=us)


class Params:
    """
    Object to hold the probabilistic representation of the parameters and state or noise
    """

    def __init__(self, thetasub, thetareg, ic, x=None, eta=None, us=None):
        """
        Expected shapes:
        thetasub ... (nsub, msub, 2)
        thetareg ... (nsub, nreg, mreg, 2)
        ic       ... (nsub, nreg, ns, 2)
        x        ... (nsub, nreg, ns, nt, 2) or None
        eta      ... (nsub, nreg, ns, nt, 2) or None
        us       ... (nsub, nt, 2) or None
        The last dimension stores the mean and standard variance of the Gaussian.
        """

        assert thetasub.ndim == 3
        assert thetareg.ndim == 4
        assert ic.ndim == 4

        self.thetasub = thetasub
        self.thetareg = thetareg
        self.ic = ic

        self.x = x
        self.eta = eta
        self.us = us


    @classmethod
    def from_file(cls, filename):
        data = np.load(filename)
        return cls(**data)


    def save(self, filename):
        data = dict(thetasub=self.thetasub, thetareg=self.thetareg, ic=self.ic)
        if self.x is not None:
            data['x'] = self.x
        if self.eta is not None:
            data['eta'] = self.eta
        if self.us is not None:
            data['us'] = self.us
        np.savez(filename, **data)


class Sm(tf.keras.Model):
    def __init__(self, ns, mreg, msub):
        super().__init__()
        self.ns = ns
        self.mreg = mreg
        self.msub = msub

    def call(self, inputs):
        return None

    def tracked_variables(self):
        """Returns an list of tuples with the tracked variable names and print formats."""
        return [], []

    def tracked_variables_values(self):
        """Returns an list of tuples with the tracked variable names and values."""
        return []


class SmNormal(Sm):
    def __init__(self, ns, mreg, msub, f_units, network_input=True, shared_input=False, alpha=0.0, activation='relu',
                 safe_init=False):
        super().__init__(ns, mreg, msub)

        self.network_input = network_input
        self.shared_input = shared_input
        input_shape = ns + mreg + msub + int(network_input) + int(shared_input)

        if activation == 'relu' and safe_init:
            q = 0.01
            sq = q**(1./2.)

            init_fw1 = np.random.uniform(-0.1*sq, 0.1*sq, size=(input_shape, f_units))
            init_fw1[np.r_[0:ns], np.r_[0:ns]]   =  sq
            init_fw1[np.r_[0:ns], np.r_[ns:2*ns]] = -sq
            init_fw2 = np.random.uniform(-0.1*sq, 0.1*sq, size=(f_units, ns))
            init_fw2[np.r_[0:ns],   np.r_[0:ns]] = -sq
            init_fw2[np.r_[ns:2*ns], np.r_[0:ns]] =  sq

            ki1, bi1 = tf.constant_initializer(init_fw1), 'zeros'
            ki2, bi2 = tf.constant_initializer(init_fw2), 'zeros'
        else:
            ki1, bi1 = None, None
            ki2, bi2 = None, None

        self.f = keras.Sequential([
            keras.layers.InputLayer(input_shape=(input_shape,)),
            keras.layers.Dense(f_units, activation=activation,
                               kernel_initializer=ki1, bias_initializer=bi1,
                               kernel_regularizer=keras.regularizers.l2(alpha),
                               bias_regularizer=keras.regularizers.l2(alpha)),
            keras.layers.Dense(ns,
                               kernel_initializer=ki2, bias_initializer=bi2,
                               kernel_regularizer=keras.regularizers.l2(alpha),
                               bias_regularizer=keras.regularizers.l2(alpha))
        ], name="f")

        self.slv = tf.Variable(-2*tf.ones(self.ns, dtype=tf.float32), trainable=True, name="slv")


    def tracked_variables(self):
        """Returns an list of tuples with the tracked variable names and print formats."""
        names = [f"slv[{i}]" for i in range(self.ns)]
        formats = ["%6.3f"   for i in range(self.ns)]
        return names, formats

    def tracked_variables_values(self):
        """Returns an list of tuples with the tracked variable names and values."""
        return [self.slv[i].numpy() for i in range(self.ns)]

    def simulate(self, x0, eta, u, us, treg, tsub, nt):
        """Tensorflow implementation of forward simulation"""
        eta = tf.exp(0.5 * self.slv) * eta

        # Simulation
        def cond(x, i, hist):
            return tf.less(i, nt)

        if self.network_input and self.shared_input:
            def body(x, i, hist):
                # x shape:       nsamples x nvars
                # fx shape:      nsamples x nvars
                xaug = tf.concat([x, treg, tsub, u[:,i,:], us[:,i,:]], axis=1)
                fx = self.f(xaug)
                xnext = x + fx + eta[:, i]
                return [xnext, tf.add(i, 1), hist.write(i, xnext)]
        elif self.network_input:
            def body(x, i, hist):
                # x shape:       nsamples x nvars
                # fx shape:      nsamples x nvars
                xaug = tf.concat([x, treg, tsub, u[:,i,:]], axis=1)
                fx = self.f(xaug)
                xnext = x + fx + eta[:, i]
                return [xnext, tf.add(i, 1), hist.write(i, xnext)]
        elif self.shared_input:
            def body(x, i, hist):
                # x shape:       nsamples x nvars
                # fx shape:      nsamples x nvars
                xaug = tf.concat([x, treg, tsub, us[:,i,:]], axis=1)
                fx = self.f(xaug)
                xnext = x + fx + eta[:, i]
                return [xnext, tf.add(i, 1), hist.write(i, xnext)]
        else:
            def body(x, i, hist):
                # x shape:       nsamples x nvars
                # fx shape:      nsamples x nvars
                xaug = tf.concat([x, treg, tsub], axis=1)
                fx = self.f(xaug)
                xnext = x + fx + eta[:, i]
                return [xnext, tf.add(i, 1), hist.write(i, xnext)]

        history = tf.TensorArray(tf.float32, size=nt)
        _, _, history = tf.while_loop(cond, body, [x0, 0, history])
        return tf.transpose(history.stack(), [1, 0, 2])


    def lpdf(self, x, u, us, treg, tsub):
        """Evaluate log probability density for given states, input, and region and subject parameters"""
        # x shape:    (batch_size, nt, ns)
        # u shape:    (batch_size, nt) or None
        # us shape:   (batch_size, nt) or None
        # treg shape: (batch_size, mreg)
        # tsub shape: (batch_size, msub)

        batch_size, nt, ns = x.shape

        xaug = [x, tf.repeat(treg[:,None,:], repeats=nt, axis=1), tf.repeat(tsub[:,None,:], repeats=nt, axis=1)]
        if self.network_input:
            xaug.append(u[:,:,None])
        if self.shared_input:
            xaug.append(us[:,:,None])
        xaug = tf.concat(xaug, axis=2)

        # xaug shape: (batch_size, nt, ns + mreg + msub [+1] [+1])
        xaug = tf.reshape(xaug, (batch_size*nt, -1))
        fx = self.f(xaug)
        fx = tf.reshape(fx, (batch_size, nt, -1))

        logp_x = (  log_normal_pdf(x[:,0, :], 0., 1., raxis=1)
                  + log_normal_pdf(x[:,1:,:], x[:,:-1,:] + fx[:,:-1,:], self.slv, raxis=[1,2]))

        return logp_x         # logp_x size: (batch_size)


    def simulate_subjects(self, w, nt, ic, us, treg, tsub, Ap, bp, olv):
        # w shape:    (nsub, nreg, nreg)
        # ic shape:   (n, nsub, nreg, ns)
        # us shape:   (n, nsub, nt)
        # treg shape: (n, nsub, nreg, mreg)
        # tsub shape: (n, nsub, msub)
        # Ap shape:   (nobs, ns)
        # bp shape:   (nobs)

        n, nsub, nreg, ns = ic.shape
        assert ns == self.ns
        nobs = Ap.shape[0]

        # Extend tsub along the regions
        tsub = np.repeat(tsub[:,:,None,:], nreg, axis=2)

        x = np.zeros((n, nsub, nreg, ns, nt))
        y = np.zeros((n, nsub, nreg, nobs, nt))
        x[:,:,:,:,0] = ic

        for isamp in range(n):
            for isub in range(nsub):
                etasys = np.exp(0.5 * self.slv)[None,:,None] * np.random.normal(0, 1, size=(nreg, ns, nt))
                etaobs = np.exp(0.5 * olv)                   * np.random.normal(0, 1, size=(nreg, nobs, nt))

                for i in range(nt-1):
                    xaug = [x[isamp,isub,:,:,i], treg[isamp,isub], tsub[isamp,isub]]
                    if self.network_input:
                        ytilda = (np.tensordot(x[isamp,isub,:,:,i], Ap, axes=[[1],[1]]) + bp)[:,0]
                        u = np.dot(w[isub], ytilda)
                        xaug.append(u[:,None])
                    if self.shared_input:
                        xaug.append(np.repeat(us[isamp,isub,i,None,None], nreg, axis=0))
                    xaug = np.hstack(xaug)
                    fx = self.f(xaug)
                    x[isamp,isub,:,:,i+1] = x[isamp,isub,:,:,i] + fx + etasys[:,:,i]
                y[isamp,isub] = np.swapaxes(np.tensordot(x[isamp,isub], Ap, axes=[[1],[1]]), 1, 2) + etaobs + bp

        return x, y


class SmGmm(Sm):
    def __init__(self, ns, mreg, msub, f_units, ng, network_input=True, shared_input=False, alpha=0.0):
        super().__init__(ns, mreg, msub)

        self.ng = ng

        self.network_input = network_input
        self.shared_input = shared_input
        input_shape = ns + mreg + msub + int(network_input) + int(shared_input)

        self.f = keras.Sequential([
            keras.layers.InputLayer(input_shape=(input_shape,)),
            keras.layers.Dense(f_units, activation='tanh',
                               kernel_regularizer=keras.regularizers.l2(alpha),
                               bias_regularizer=keras.regularizers.l2(alpha)),
            # Mean and variance for each Gaussian and each state, and (non-constrained) ratios
            keras.layers.Dense(2*ng*ns + ng-1,
                               kernel_regularizer=keras.regularizers.l2(alpha),
                               bias_regularizer=keras.regularizers.l2(alpha))

        ], name="f")

    def simulate(self, x0, eta, iext, treg, tsub, nt):
        raise NotImplementedError()

    def lpdf(self, x, u, us, treg, tsub):
        """Evaluate log probability density for given states, input, and region and subject parameters"""

        # x shape:    (batch_size, nt, ns)
        # u shape:    (batch_size, nt) or None
        # us shape:   (batch_size, nt) or None
        # treg shape: (batch_size, mreg)
        # tsub shape: (batch_size, msub)

        batch_size, nt, ns = x.shape

        xaug = [x, tf.repeat(treg[:,None,:], repeats=nt, axis=1), tf.repeat(tsub[:,None,:], repeats=nt, axis=1)]
        if self.network_input:
            xaug.append(u[:,:,None])
        if self.shared_input:
            xaug.append(us[:,:,None])
        xaug = tf.concat(xaug, axis=2)

        # xaug shape: (batch_size, nt, ns+1+mreg+msub)
        xaug = tf.reshape(xaug, (batch_size*nt, -1))

        fx, gx, phi = tf.split(self.f(xaug), num_or_size_splits=[self.ng*self.ns, self.ng*self.ns, self.ng-1], axis=-1)
        phi = tf.transpose(unit_simplex(tf.transpose(phi)))

        fx = tf.reshape(fx,   (batch_size, nt, self.ns, self.ng))
        gx = tf.reshape(gx,   (batch_size, nt, self.ns, self.ng))
        phi = tf.reshape(phi, (batch_size, nt, self.ng))

        logp_x = (log_normal_pdf(x[:,0,:], 0., 1., raxis=1)
                  + tf.reduce_sum(tf.math.reduce_logsumexp(
                      tf.math.log(phi[:,:-1,None,:]) +
                      log_normal_pdf(x[:,1:,:,None], x[:,:-1,:,None] + fx[:,:-1,:], gx[:,:-1,:], raxis=None),
                    axis=3), axis=[1,2]))

        return logp_x        # logp_x size: (batch_size)


    def simulate_subjects(self, w, nt, ic, us, treg, tsub, Ap, bp, olv):
        # w shape:    (nsub, nreg, nreg)
        # ic shape:   (n, nsub, nreg, ns)
        # treg shape: (n, nsub, nreg, mreg)
        # tsub shape: (n, nsub, msub)
        # Ap shape:   (nobs, ns)
        # bp shape:   (nobs)

        n, nsub, nreg, ns = ic.shape
        assert ns == self.ns
        nobs = Ap.shape[0]

        # Extend tsub along the regions
        tsub = np.repeat(tsub[:,:,None,:], nreg, axis=2)

        x = np.zeros((n, nsub, nreg, ns, nt))
        y = np.zeros((n, nsub, nreg, nobs, nt))
        x[:,:,:,:,0] = ic

        for isamp in range(n):
            for isub in range(nsub):
                etaobs = np.exp(0.5 * olv) * np.random.normal(0, 1, size=(nreg, nobs, nt))

                for i in range(nt-1):
                    xaug = [x[isamp,isub,:,:,i], treg[isamp,isub], tsub[isamp,isub]]
                    if self.network_input:
                        ytilda = (np.tensordot(x[isamp,isub,:,:,i], Ap, axes=[[1],[1]]) + bp)[:,0]
                        u = np.dot(w[isub], ytilda)
                        xaug.append(u[:,None])
                    if self.shared_input:
                        xaug.append(np.repeat(us[isamp,isub,i,None,None], nreg, axis=0))
                    xaug = np.hstack(xaug)

                    fx, gx, phi = tf.split(self.f(xaug), num_or_size_splits=[self.ng*ns, self.ng*ns, self.ng-1], axis=1)

                    fx = tf.reshape(fx, (nreg, ns, self.ng)).numpy()
                    gx = tf.reshape(gx, (nreg, ns, self.ng)).numpy()
                    phi = tf.transpose(unit_simplex(tf.transpose(phi))).numpy()

                    gm_inds = np.array([np.random.choice(self.ng, p=phi[i]) for i in range(nreg)])
                    dx = np.random.normal(fx[np.r_[:nreg],:,gm_inds], np.exp(0.5*gx[np.r_[:nreg],:,gm_inds]))

                    x[isamp,isub,:,:,i+1] = x[isamp,isub,:,:,i] + dx

                y[isamp,isub] = np.swapaxes(np.tensordot(x[isamp,isub], Ap, axes=[[1],[1]]), 1, 2) + etaobs + bp

        return x, y


def get_source_model(prediction, ns, mreg, msub, f_units, ng, network_input=True, shared_input=False,
                     alpha=0.0, activation='relu'):
    if prediction == 'normal':
        return SmNormal(ns, mreg, msub, f_units,  network_input=network_input, shared_input=shared_input,
                        alpha=alpha, activation=activation)
    elif prediction == 'gmm':
        return SmGmm(ns, mreg, msub, f_units, ng, network_input=network_input, shared_input=shared_input, alpha=alpha)


class RegModel(tf.keras.Model):
    def __init__(self, ns, mreg, msub, nreg, nsub, nt, nobs, upsample_factor=1, fix_observation=False,
                 prediction='normal', network_input=True, shared_input=False,
                 ng=1, f_units=32, lambda_a=0.0, lambda_x=0.0, alpha=0.0, activation='relu'):
        super().__init__()

        self.training_mode = 'region-upsampled'
        self.ns = ns
        self.mreg = mreg
        self.msub = msub
        self.nreg = nreg
        self.nsub = nsub
        self.nt = nt
        self.nobs = nobs
        self.upsample_factor = upsample_factor
        self.lambda_a = lambda_a
        self.lambda_x = lambda_x

        self.source_model = get_source_model(prediction, ns, mreg, msub, f_units, ng=ng,
                                             network_input=network_input, shared_input=shared_input,
                                             alpha=alpha, activation=activation)
        self.shared_input = shared_input

        # Subject-specific parameters (theta_subject)
        self.tsub = tf.Variable(tf.random.normal((nsub, msub, 2), mean=0, stddev=0.01), trainable=True, name="tsub")

        # Shared input (if needed)
        self.logtau = tf.Variable(tf.math.log(10.), dtype=tf.float32, trainable=True, name="logtau")
        if shared_input:
            ntup = nt * upsample_factor
            self.us = tf.Variable(tf.random.normal((nsub, ntup, 2), mean=0, stddev=0.1), trainable=True, name="us")
        else:
            self.us = None

        # Projection
        self.olv = tf.Variable(0*tf.ones(self.nobs), dtype=tf.float32, trainable=True, name="olv")
        if not fix_observation:
            self.Ap = tf.Variable(tf.random.normal(mean=0, stddev=0.3, shape=(self.nobs, self.ns)),
                                  trainable=True, name="Ap")
            self.bp = tf.Variable(tf.zeros(self.nobs), dtype=tf.float32, trainable=True, name="bp")
        else:
            self.Ap = tf.Variable(tf.eye(self.nobs, self.ns), trainable=False, name="Ap", dtype=tf.float32)
            self.bp = tf.Variable(tf.zeros(self.nobs),        trainable=False, name="bp", dtype=tf.float32)


    def call(self, inputs):
        """Dummy function for tensorflow to work nicely"""
        return None

    def tracked_variables(self):
        """Returns an list of tuples with the tracked variable names and print formats."""
        names = [f"Ap[{i},{j}]" for i in range(self.nobs) for j in range(self.ns)]
        names += [f"bp[{i}]" for i in range(self.nobs)] + ["olv"] + ["logtau"]
        formats = ["%6.3f" for name in names]
        names_sm, formats_sm = self.source_model.tracked_variables()
        return names + names_sm, formats + formats_sm

    def tracked_variables_values(self):
        """Returns an list of tuples with the tracked variable names and values."""
        values = [self.Ap[i,j].numpy() for i in range(self.nobs) for j in range(self.ns)]
        values += [self.bp[i].numpy() for i in range(self.nobs)] + [self.olv[0].numpy()] + [self.logtau.numpy()]
        return values + self.source_model.tracked_variables_values()

    def simulate_subjects(self, w, nt, ic=None, thetareg=None, thetasub=None, us=None, n=1):
        nsub = w.shape[0]
        assert w.shape == (nsub, self.nreg, self.nreg)

        ntup = nt * self.upsample_factor

        # Use default N(0,1) where no values are provided
        if ic is None:       ic       = zerosones((nsub, self.nreg, self.ns))
        if thetareg is None: thetareg = zerosones((nsub, self.nreg, self.mreg))
        if thetasub is None: thetasub = zerosones((nsub, self.msub))

        # Repeat samples
        ic = np.repeat(ic[None,:,:,:,:], n, axis=0)
        thetareg = np.repeat(thetareg[None,:,:,:,:], n, axis=0)
        thetasub = np.repeat(thetasub[None,:,:,:], n, axis=0)

        # Sample parameters
        ic_samples = np.random.normal(ic[:,:,:,:,0], ic[:,:,:,:,1])
        thetareg_samples = np.random.normal(thetareg[:,:,:,:,0], thetareg[:,:,:,:,1])
        thetasub_samples = np.random.normal(thetasub[:,:,:,0], thetasub[:,:,:,1])

        # Shared input: AR process
        if self.shared_input:
            if us is None:
                sig_p = 1.0
                alpha = np.exp(-1./np.exp(self.logtau.numpy()))
                sig_s = sig_p * np.sqrt(1 - alpha**2)
                us_samples = np.zeros((n, nsub, ntup))
                us_samples[:,:,0] = np.random.normal(0, sig_p, size=(n, nsub))
                eta_s = np.random.normal(0, sig_s, size=(n, nsub, ntup))
                for i in range(ntup-1):
                    us_samples[:,:,i+1] = alpha*us_samples[:,:,i] + eta_s[:,:,i+1]
            else:
                us = np.repeat(us[None,:,:,:], n, axis=0)
                us_samples = np.random.normal(us[:,:,:,0], us[:,:,:,1])
        else:
            us_samples = None

        # Let source model do the simulation
        x, y = self.source_model.simulate_subjects(w, nt*self.upsample_factor, ic_samples, us_samples,
                                                   thetareg_samples, thetasub_samples,
                                                   self.Ap.numpy(), self.bp.numpy(), self.olv.numpy())

        # Average if needed
        y = tf.math.reduce_mean(tf.reshape(y, (n, nsub, self.nreg, self.nobs, nt, self.upsample_factor)), axis=-1)
        return GeneratedData(x=x, y=y, thetareg=thetareg_samples, thetasub=thetasub_samples, us=us_samples)


class RegEta(RegModel):
    """
    Networked dynamical system VAE in region mode with eta as latents.
    """
    def __init__(self, ns, mreg, msub, nreg, nsub, nt, nobs, upsample_factor=1, fix_observation=False,
                 prediction='normal', network_input=True, shared_input=False,
                 ng=None, f_units=32, encoder_units=32, lambda_a=0.0, lambda_x=0.0, lambda_fx=None, alpha=0.0,
                 activation='relu', kl_sub_factor=1.0):

        super().__init__(ns, mreg, msub, nreg, nsub, nt, nobs, upsample_factor=upsample_factor,
                         fix_observation=fix_observation, prediction=prediction, network_input=network_input,
                         shared_input=shared_input, ng=ng, f_units=f_units, lambda_a=lambda_a, lambda_x=lambda_x,
                         alpha=alpha, activation=activation)

        self.param_encoder = keras.Sequential([
            keras.layers.Bidirectional(keras.layers.LSTM(units=encoder_units, return_sequences=False,
                                                         input_shape=(None, None, nobs+1+self.nsub))),
            keras.layers.Dense(units=ns*2 + mreg*2),
        ], name="param_encoder")

        self.noise_encoder = keras.Sequential([
            keras.layers.Bidirectional(keras.layers.LSTM(units=encoder_units, return_sequences=True,
                                                         input_shape=(None, None, nobs+1+self.nsub))),
            keras.layers.Dense(units=upsample_factor*ns*2),
        ], name="noise_encoder")

        self.latent = 'eta'
        self.lambda_fx = lambda_fx if (lambda_fx is not None) else tf.zeros(ns, dtype=tf.float32)
        self.kl_sub_factor = kl_sub_factor


    def encode(self, subj_ind, yobs, u):
        """
        Take the data (region timeseries + network input + one-hot vector for subject ID) and apply the encoders
        """

        # subj_ind shape: (batch_size)
        # yobs shape:     (batch_size, nt, nobs)
        # u shape:        (batch_size, nt)

        batch_size, nt, nobs = yobs.shape
        subj_onehot = tf.repeat(tf.one_hot(subj_ind, depth=self.nsub)[:,None,:], repeats=nt, axis=1)
        yaug = tf.concat([yobs, u[:,:,None], subj_onehot], axis=2)

        uf = self.upsample_factor
        etamu, etalv = tf.split(self.noise_encoder(yaug), num_or_size_splits=[uf*self.ns, uf*self.ns], axis=2)
        x0mu, x0lv, tregmu, treglv = tf.split(self.param_encoder(yaug),
                                              num_or_size_splits=[self.ns, self.ns, self.mreg, self.mreg], axis=1)
        # etamu, etalv shape:   (batch_size, nt, upsample_factor*ns)
        # x0mu, x0lv shape:     (batch_size, ns)
        # tregmu, treglv shape: (batch_size, mreg)

        # Squash the interpolated values
        if self.upsample_factor > 1:
            etamu = tf.reshape(etamu, (batch_size, nt*self.upsample_factor, self.ns))
            etalv = tf.reshape(etalv, (batch_size, nt*self.upsample_factor, self.ns))

        # Get the subject parameters
        tsubmu, tsublv = tf.unstack(tf.gather(self.tsub, subj_ind, axis=0), axis=-1)
        return etamu, etalv, x0mu, x0lv, tregmu, treglv, tsubmu, tsublv

    def loss(self, training_batch, nsamples=8, betax=1.0, betap=1.0):
        """
        Return loss for a given training batch
        """

        subj_ind, yobs, u, u_upsampled = training_batch
        # subj_ind shape:       (batch_size)
        # yobs shape:           (batch_size, nt, nobs)
        # u shape:              (batch_size, nt)
        # u_upsampled shape     (batch_size, nt*upsample_factor)

        batch_size, nt, nobs = yobs.shape
        ntup = nt*self.upsample_factor

        # Add samples
        subj_ind    = tf.repeat(subj_ind,    nsamples, axis=0)
        yobs        = tf.repeat(yobs,        nsamples, axis=0)
        u           = tf.repeat(u,           nsamples, axis=0)
        u_upsampled = tf.repeat(u_upsampled, nsamples, axis=0)

        # Encode
        etamu, etalv, x0mu, x0lv, tregmu, treglv, tsubmu, tsublv = self.encode(subj_ind, yobs, u)

        # Sample
        treg = gen_normal(tregmu, treglv)
        tsub = gen_normal(tsubmu, tsublv)
        x0   = gen_normal(x0mu, x0lv)
        eta  = gen_normal(etamu, etalv)

        us = None
        if self.shared_input:
            usmu, uslv = tf.unstack(tf.gather(self.us, subj_ind, axis=0), axis=-1)
            us = gen_normal(usmu, uslv)

        x = self.source_model.simulate(x0=x0, eta=eta, u=u_upsampled[:,:,None], us=us, treg=treg, tsub=tsub, nt=ntup)

        # Likelihood
        xave = tf.math.reduce_mean(tf.reshape(x, (batch_size*nsamples, nt, self.upsample_factor, self.ns)), axis=2)
        ypre = tf.tensordot(xave, self.Ap, axes=[[2],[1]]) + self.bp
        logp_y_x = log_normal_pdf(yobs, ypre, self.olv, raxis=[1,2])

        # KL divergences for noise
        kleta = 0.5 * tf.reduce_sum(etamu**2 + tf.exp(etalv) - etalv - 1, axis=[1,2])
        # etamu_bar = etamu / tf.exp(0.5 * self.source_model.slv)
        # etalv_bar = etalv - self.source_model.slv
        # kleta = 0.5 * tf.reduce_sum(etamu_bar**2 + tf.exp(etalv_bar) - etalv_bar - 1, axis=[1,2])

        # KL divergences for IC and parameters
        klx0  =                  0.5 * tf.reduce_sum(x0mu**2   + tf.exp(x0lv)   - x0lv   - 1, axis=1)
        klreg =                  0.5 * tf.reduce_sum(tregmu**2 + tf.exp(treglv) - treglv - 1, axis=1)
        klsub = (1./self.nreg) * 0.5 * tf.reduce_sum(tsubmu**2 + tf.exp(tsublv) - tsublv - 1, axis=1)

        # Penalties
        xpen = self.lambda_x * tf.reduce_sum(tf.square(x), axis=[1,2])
        apen = self.lambda_a * tf.linalg.norm(self.Ap, ord=1)
        wpen = tf.reduce_sum(self.source_model.losses)
        elbo = logp_y_x - betax*(klx0 + kleta) - betap*klreg - self.kl_sub_factor*klsub

        # fx penalties
        xdiff = x[:,1:,:] - x[:,:-1,:]
        fxpen = tf.reduce_sum(self.lambda_fx * tf.reduce_sum(tf.square(xdiff), axis=1), axis=1)

        if self.shared_input:
            # Prior and approximate posterior for the shared input
            usmu, uslv = tf.unstack(tf.gather(self.us, subj_ind, axis=0), axis=-1)
            us = gen_normal(usmu, uslv)
            logp_us = lpdf_ar1(us, tf.math.exp(self.logtau), 0.0)
            logq_us = log_normal_pdf(us, usmu, uslv, raxis=1)
            elbo += 1./self.nreg * (logp_us - logq_us)

        return tf.reduce_mean(-elbo + apen + xpen + fxpen + wpen)


    def encode_subjects(self, w, yobs, subj_ind=None):
        # TODO: merge/clean up with encode_subjects of RegX?

        nsub = w.shape[0]
        if subj_ind is None:
            # We are encoding all subjects
            assert nsub == self.nsub
            subj_ind = np.r_[:nsub]
        else:
            assert len(subj_ind) == nsub

        assert w.shape == (nsub, self.nreg, self.nreg)
        assert yobs.ndim == 4
        assert yobs.shape[:3] == (nsub, self.nreg, self.nobs)
        nt = yobs.shape[3]

        u = get_network_input_obs(w, yobs, comp=0)[:,:,0,:]

        us = None
        if self.shared_input:
            usmu, uslv = tf.unstack(tf.gather(self.us, subj_ind, axis=0), axis=-1)
            us = np.zeros((nsub, self.nt*self.upsample_factor, 2))
            us[:,:,0] =            np.reshape(usmu, (nsub, -1))
            us[:,:,1] = np.exp(0.5*np.reshape(uslv, (nsub, -1)))

        # Reshape etc
        subj_ind = np.repeat(subj_ind, self.nreg, axis=0)
        yobs = np.reshape(np.swapaxes(yobs, 2, 3), (nsub*self.nreg, nt, self.nobs))
        u = np.reshape(u, (nsub*self.nreg, nt))

        # Encode
        etamu, etalv, x0mu, x0lv, tregmu, treglv, tsubmu, tsublv = self.encode(subj_ind, yobs, u)

        ic = np.zeros((nsub, self.nreg, self.ns, 2))
        ic[:,:,:,0] =            np.reshape(x0mu, (nsub, self.nreg, self.ns))
        ic[:,:,:,1] = np.exp(0.5*np.reshape(x0lv, (nsub, self.nreg, self.ns)))

        treg = np.zeros((nsub, self.nreg, self.mreg, 2))
        treg[:,:,:,0] =            np.reshape(tregmu, (nsub, self.nreg, self.mreg))
        treg[:,:,:,1] = np.exp(0.5*np.reshape(treglv, (nsub, self.nreg, self.mreg)))

        tsub = np.zeros((nsub, self.msub, 2))
        # All regions should be the same, so we take the first one only
        tsub[:,:,0] =            np.reshape(tsubmu, (nsub, self.nreg, self.msub))[:,0,:]
        tsub[:,:,1] = np.exp(0.5*np.reshape(tsublv, (nsub, self.nreg, self.msub)))[:,0,:]

        params = Params(ic=ic, thetareg=treg, thetasub=tsub, us=us)
        return params


class RegX(RegModel):
    """
    Networked dynamical system VAE in region mode with X as latents.
    """

    def __init__(self, ns, mreg, msub, nreg, nsub, nt, nobs, upsample_factor=1, fix_observation=False,
                 prediction='normal', network_input=True, shared_input=False, ng=None,
                 f_units=32, encoder_units=32, lambda_a=0.0, lambda_x=0.0, lambda_fx=None, alpha=0.0,
                 activation='relu', kl_sub_factor=1.0):

        super().__init__(ns, mreg, msub, nreg, nsub, nt, nobs, upsample_factor=upsample_factor,
                         fix_observation=fix_observation, prediction=prediction,
                         network_input=network_input, shared_input=shared_input, ng=ng, f_units=f_units,
                         lambda_a=lambda_a, lambda_x=lambda_x, alpha=alpha, activation=activation)

        self.param_encoder = keras.Sequential([
            keras.layers.Bidirectional(keras.layers.LSTM(units=encoder_units, return_sequences=False,
                                                         input_shape=(None, None, nobs+1+self.nsub))),
            keras.layers.Dense(units=mreg*2)
        ], name="param_encoder")

        self.state_encoder = keras.Sequential([
            keras.layers.Bidirectional(keras.layers.LSTM(units=encoder_units, return_sequences=True,
                                                         input_shape=(None, None, nobs+1+self.nsub))),
            keras.layers.Dense(units=upsample_factor*ns*2)
        ], name="state_encoder")


        self.latent = 'x'
        self.lambda_fx = lambda_fx if (lambda_fx is not None) else tf.zeros(ns, dtype=tf.float32)
        self.elbo = None
        self.kl_sub_factor = kl_sub_factor


    def encode(self, subj_ind, yobs, u):
        """
        Take the data (region timeseries + network input + one-hot vector for subject ID) and apply the encoders
        """

        # subj_ind shape: (batch_size)
        # yobs shape:     (batch_size, nt, nobs)
        # u shape:        (batch_size, nt)

        batch_size, nt, nobs = yobs.shape
        subj_onehot = tf.repeat(tf.one_hot(subj_ind, depth=self.nsub)[:,None,:], repeats=nt, axis=1)
        yaug = tf.concat([yobs, u[:,:,None], subj_onehot], axis=2)

        uf = self.upsample_factor
        xmu, xlv = tf.split(self.state_encoder(yaug),  num_or_size_splits=[uf*self.ns, uf*self.ns], axis=2)
        tregmu, treglv = tf.split(self.param_encoder(yaug), num_or_size_splits=[self.mreg, self.mreg], axis=1)
        # xmu, xlv shape:         (batch_size, nt, upsample_factor*ns)
        # tregmu, treglv shape:   (batch_size, mreg)

        # Squash the interpolated values
        if self.upsample_factor > 1:
            xmu = tf.reshape(xmu, (batch_size, nt*self.upsample_factor, self.ns))
            xlv = tf.reshape(xlv, (batch_size, nt*self.upsample_factor, self.ns))

        # Get the subject parameters
        tsubmu, tsublv = tf.unstack(tf.gather(self.tsub, subj_ind, axis=0), axis=-1)

        return xmu, xlv, tregmu, treglv, tsubmu, tsublv


    def loss(self, training_batch, nsamples=8, betax=1.0, betap=1.0):
        """
        Return loss for a given training batch
        """

        subj_ind, yobs, u, u_upsampled = training_batch
        # subj_ind shape:    (batch_size)
        # yobs shape:        (batch_size, nt, nobs)
        # u shape:           (batch_size, nt)
        # u_upsampled shape  (batch_size, nt*upsample_factor)

        batch_size, nt, nobs = yobs.shape
        ntup = nt*self.upsample_factor

        # Add samples
        subj_ind    = tf.repeat(subj_ind,    nsamples, axis=0)
        yobs        = tf.repeat(yobs,        nsamples, axis=0)
        u           = tf.repeat(u,           nsamples, axis=0)
        u_upsampled = tf.repeat(u_upsampled, nsamples, axis=0)

        # Encode
        xmu, xlv, tregmu, treglv, tsubmu, tsublv = self.encode(subj_ind, yobs, u)

        # Sample
        x    = gen_normal(xmu, xlv)
        treg = gen_normal(tregmu, treglv)
        tsub = gen_normal(tsubmu, tsublv)

        us = None
        if self.shared_input:
            usmu, uslv = tf.unstack(tf.gather(self.us, subj_ind, axis=0), axis=-1)
            us = gen_normal(usmu, uslv)


        # Likelihood
        xave = tf.math.reduce_mean(tf.reshape(x, (batch_size*nsamples, nt, self.upsample_factor, self.ns)), axis=2)
        ypre = tf.tensordot(xave, self.Ap, axes=[[2],[1]]) + self.bp
        logp_y_x = log_normal_pdf(yobs, ypre, self.olv, raxis=[1,2])

        # Prior for states
        logp_x = self.source_model.lpdf(x, u_upsampled, us, treg, tsub)

        # Approximate posterior
        logq_x = log_normal_pdf(x, xmu, xlv, raxis=[1,2])

        # KL divergences for parameters
        klreg =                  0.5 * tf.reduce_sum(tregmu**2 + tf.exp(treglv) - treglv - 1, axis=1)
        klsub = (1./self.nreg) * 0.5 * tf.reduce_sum(tsubmu**2 + tf.exp(tsublv) - tsublv - 1, axis=1)

        # Penalties
        xdiff = x[:,1:,:] - x[:,:-1,:]
        fxpen = tf.reduce_sum(self.lambda_fx * tf.reduce_sum(tf.square(xdiff), axis=1), axis=1)

        xpen = self.lambda_x * tf.reduce_sum(tf.square(x), axis=[1,2])
        apen = self.lambda_a * tf.linalg.norm(self.Ap, ord=1)
        elbo = logp_y_x + betax*(logp_x - logq_x) - betap*klreg - self.kl_sub_factor*klsub
        wpen = tf.reduce_sum(self.source_model.losses)

        if self.shared_input:
            # Prior and approximate posterior for the shared input
            logp_us = lpdf_ar1(us, tf.math.exp(self.logtau), 0.0)
            logq_us = log_normal_pdf(us, usmu, uslv, raxis=1)
            elbo += 1./self.nreg * (logp_us - logq_us)

        self.elbo = tf.reshape(elbo, (batch_size, nsamples))
        return tf.reduce_mean(-elbo + apen + xpen + fxpen + wpen)


    def encode_subjects(self, w, yobs, subj_ind=None):
        nsub = w.shape[0]
        if subj_ind is None:
            # We are encoding all subjects
            assert nsub == self.nsub
            subj_ind = np.r_[:nsub]
        else:
            assert len(subj_ind) == nsub

        assert w.shape == (nsub, self.nreg, self.nreg)
        assert yobs.ndim == 4
        assert yobs.shape[:3] == (nsub, self.nreg, self.nobs)
        nt = yobs.shape[3]

        u = get_network_input_obs(w, yobs, comp=0)[:,:,0,:]

        us = None
        if self.shared_input:
            usmu, uslv = tf.unstack(tf.gather(self.us, subj_ind, axis=0), axis=-1)
            us = np.zeros((nsub, self.nt*self.upsample_factor, 2))
            us[:,:,0] =            np.reshape(usmu, (nsub, -1))
            us[:,:,1] = np.exp(0.5*np.reshape(uslv, (nsub, -1)))

        # Reshape etc
        subj_ind = np.repeat(subj_ind, self.nreg, axis=0)
        yobs = np.reshape(np.swapaxes(yobs, 2, 3), (nsub*self.nreg, nt, self.nobs))
        u = np.reshape(u, (nsub*self.nreg, nt))

        # Encode
        xmu, xlv, tregmu, treglv, tsubmu, tsublv = self.encode(subj_ind, yobs, u)

        ic = np.zeros((nsub, self.nreg, self.ns, 2))
        ic[:,:,:,0] =            np.reshape(xmu[:,0,:], (nsub, self.nreg, self.ns))
        ic[:,:,:,1] = np.exp(0.5*np.reshape(xlv[:,0,:], (nsub, self.nreg, self.ns)))

        treg = np.zeros((nsub, self.nreg, self.mreg, 2))
        treg[:,:,:,0] =            np.reshape(tregmu, (nsub, self.nreg, self.mreg))
        treg[:,:,:,1] = np.exp(0.5*np.reshape(treglv, (nsub, self.nreg, self.mreg)))

        tsub = np.zeros((nsub, self.msub, 2))
        # All regions should be the same, so we take the first one only
        tsub[:,:,0] =            np.reshape(tsubmu, (nsub, self.nreg, self.msub))[:,0,:]
        tsub[:,:,1] = np.exp(0.5*np.reshape(tsublv, (nsub, self.nreg, self.msub)))[:,0,:]

        nts = xmu.shape[1]
        x = np.zeros((nsub, self.nreg, self.ns, nts, 2))
        x[:,:,:,:,0] =            np.reshape(np.swapaxes(xmu, 1, 2), (nsub, self.nreg, self.ns, nts))
        x[:,:,:,:,1] = np.exp(0.5*np.reshape(np.swapaxes(xlv, 1, 2), (nsub, self.nreg, self.ns, nts)))


        params = Params(ic=ic, thetareg=treg, thetasub=tsub, x=x, us=us)
        return params


def evalf(model, x, thetareg, thetasub, u=None, ushared=None):
    xaug_ = list(x) + list(thetareg) + list(thetasub)
    if model.source_model.network_input:
        xaug_.extend([u])
    if model.source_model.shared_input:
        xaug_.extend([ushared])

    # Expand non-iterables
    xaug = []
    out_shape = []
    for maybe_iterable in xaug_:
        try:
            xaug.append([e for e in maybe_iterable])
            out_shape.append(len(maybe_iterable))
        except TypeError:
            xaug.append([maybe_iterable])

    xaug = np.array(list(itertools.product(*xaug)))

    fx = model.source_model.f(xaug).numpy()
    if len(out_shape) == 0:
        out_shape = [1]

    return np.reshape(fx, out_shape + [fx.shape[1]])
