import sys
import os
import itertools

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import scipy.optimize as spopt
import scipy.integrate as spi
import scipy.stats as sps

import tensorflow as tf
import pandas as pd

import ndsvae as ndsv

import util



def init_points(method, n, init_range, ns):
    if method == 'grid':
        x0 = np.array(np.meshgrid(*(ns * [np.linspace(init_range[0], init_range[1], n)]))).reshape((ns, -1)).T
    elif method == 'random':
        x0 = np.random.uniform(init_range[0], init_range[1], size=(n, ns))
    return x0


def deduplicate(xs, threshold):
    xds = []
    for x in xs:
        if np.any([(np.linalg.norm(x - xd) < threshold) for xd in xds]):
            continue
        else:
            xds.append(x)
    return np.array(xds)


def find_fixed_points_node(model, thetareg, thetasub, u=0., us=0., init='random', n=5, init_range=(-2,2),
                           threshold=1e-4):

    assert len(thetareg) == model.mreg
    assert len(thetasub) == model.msub
    if model.source_model.network_input: assert u  is not None
    if model.source_model.shared_input:  assert us is not None

    x0 = init_points(init, n, init_range, model.ns)
    n_ = len(x0)

    fixed_input = [thetareg, thetasub]
    if model.source_model.network_input: fixed_input.append([u])
    if model.source_model.shared_input:  fixed_input.append([us])
    fixed_input = np.concatenate(fixed_input)

    input_size = (model.ns + model.mreg + model.msub
                  + int(model.source_model.network_input) + int(model.source_model.shared_input))
    xaug = np.zeros((input_size))
    xaug[model.ns:] = fixed_input

    def f(x):
        xaug_ = np.copy(xaug)
        xaug_[:model.ns] = x
        fx = model.source_model.f(xaug_[None,:])[0].numpy()
        return fx

    @tf.function
    def jacobian(x):
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as g:
            g.watch(x)
            xaug_ = tf.concat([x, xaug[model.ns:]], axis=0)
            fx = model.source_model.f(xaug_[None,:])[0]
        jac = g.jacobian(fx, x)
        return jac

    xs = []
    fails = 0
    for i in range(n_):
        sol = spopt.root(f, x0[i], jac=jacobian, method='hybr', tol=1e-6)

        if not sol.success:
            fails += 1
        else:
            xs.append(sol.x)

    xs = np.array(xs)
    xs = deduplicate(xs, threshold)

    # Evaluate stability
    stability = np.zeros(len(xs), dtype=bool)
    eigvals = np.zeros((len(xs), model.ns), dtype=complex)
    eigvecs = np.zeros((len(xs), model.ns, model.ns), dtype=complex)
    jacs = np.zeros((len(xs), model.ns, model.ns))

    for i, x in enumerate(xs):
        jac = jacobian(x).numpy()
        evals, evecs = np.linalg.eig(jac)
        order = np.argsort(-evals.real)
        eigvals[i] = evals[order]
        eigvecs[i] = evecs[:,order]
        stability[i] = np.all(np.real(eigvals[i]) < 0.)

    return xs, stability, eigvals, eigvecs



def find_fixed_points_network(model, w, thetareg, thetasub, us=0., n=10, init_range=(-2,2), threshold=1e-4):
    nreg = model.nreg

    assert thetareg.shape == (nreg, model.mreg)
    assert thetasub.shape == (model.msub,)

    x0 = np.random.uniform(init_range[0], init_range[1], size=(n, nreg * model.ns))
    Ap = model.Ap.numpy()
    bp = model.bp.numpy()
    w = w.astype('float32')

    has_network = int(model.source_model.network_input)
    has_shared  = int(model.source_model.shared_input)
    input_size = (model.ns + model.mreg + model.msub + has_network + has_shared)

    xaugf = np.zeros((nreg, input_size))
    xaugf[:, model.ns:model.ns+model.mreg] = thetareg
    xaugf[:, model.ns+model.mreg:model.ns+model.mreg+model.msub] = thetasub
    if has_shared: xaugf[:, -1] = us

    def f(x):
        x = x.reshape((nreg, model.ns))
        y = (np.dot(Ap, x.T) + bp)[0,:]
        u = np.dot(w, y)

        xaug = np.copy(xaugf)
        xaug[:, :model.ns] = x
        if has_network:
            xaug[:, -has_network-has_shared] = u

        fx = model.source_model.f(xaug).numpy().reshape(nreg * model.ns)
        return fx

    @tf.function
    def jacobian(x):
        x = tf.cast(x, 'float32')
        with tf.GradientTape(persistent=True, watch_accessed_variables=False) as g:
            g.watch(x)
            xr = tf.reshape(x, (nreg, model.ns))
            y = tf.tensordot(Ap, xr, axes=[[1], [1]])[0] + bp
            u = tf.linalg.matvec(w, y)

            if not has_network:
                xaug = tf.concat([xr, xaugf[:, model.ns:]], axis=1)
            else:
                xaug = tf.concat([xr,
                                  xaugf[:, model.ns:model.ns+model.mreg+model.msub],
                                  u[:,None],
                                  xaugf[:, model.ns+model.mreg+model.msub+has_network:],
                                 ], axis=1)

            fx = model.source_model.f(xaug)
            fx = tf.reshape(fx, (nreg*model.ns,))

        jac = g.jacobian(fx, x)
        return jac

    xs = []
    fails = 0
    for i in range(n):
        sol = spopt.root(f, x0[i], jac=jacobian, method='hybr', tol=1e-6)

        if not sol.success:
            fails += 1
        else:
            xs.append(sol.x)

    xs = np.array(xs)
    xs = deduplicate(xs, threshold)
    xs = xs.reshape(-1, nreg, model.ns)

    # Evaluate stability
    stability = np.zeros(len(xs), dtype=bool)
    eigvals = np.zeros((len(xs), nreg*model.ns), dtype=complex)
    eigvecs = np.zeros((len(xs), nreg*model.ns, nreg*model.ns), dtype=complex)

    for i, x in enumerate(xs):
        jac = jacobian(np.reshape(x, nreg*model.ns)).numpy()
        evals, evecs = np.linalg.eig(jac)
        order = np.argsort(-evals.real)
        eigvals[i] = evals[order]
        eigvecs[i] = evecs[:,order]
        stability[i] = np.all(np.real(eigvals[i]) < 0.)

    return xs, stability, eigvals, eigvecs



def is_converging(x, t, nsteps, threshold):
    nsteps = min(len(t), nsteps)
    dist = np.linalg.norm(x[:,:,-nsteps:] - x[:,:,[-1]], axis=1)
    converging = np.zeros(x.shape[0], dtype=bool)

    for i in range(x.shape[0]):
        if np.all(dist[i] < threshold):
            converging[i] = True
        else:
            last_above = np.where(dist[i] >= threshold)[0][-1]
            dist_above = dist[i,:last_above+1]
            converging[i] = np.all(dist_above[:-1] > dist_above[1:])

    return converging


def is_periodic(x, t, threshold, min_period=0.):
    n = len(t)
    max_offset = int(len(t) / 4)

    diff = np.zeros((max_offset, n-max_offset))
    for i in range(n-max_offset):
        diff[:,i] = np.linalg.norm(x[:,[i]] - x[:,i:i+max_offset], axis=0)

    min_ind = np.argwhere(t > t[0] + min_period)[0][0]
    candidate_ind = np.where(np.max(diff[min_ind:,:], axis=1) < threshold)[0][0] + min_ind

    if np.max(diff[candidate_ind,:]) < threshold:
        return True, t[candidate_ind] - t[0]
    else:
        return False, np.nan


def find_attractors_node(model, thetareg, thetasub, u=0., us=0., init='random', n=5, init_range=(-2,2),
                         T1=100, T2=100, max_step=1.):

    assert len(thetareg) == model.mreg
    assert len(thetasub) == model.msub
    if model.source_model.network_input: assert u  is not None
    if model.source_model.shared_input:  assert us is not None

    x0 = init_points(init, n, init_range, model.ns)
    n_ = len(x0)

    fixed_input = [thetareg, thetasub]
    if model.source_model.network_input: fixed_input.append([u])
    if model.source_model.shared_input:  fixed_input.append([us])
    fixed_input = np.concatenate(fixed_input)

    input_size = (model.ns + model.mreg + model.msub
                  + int(model.source_model.network_input) + int(model.source_model.shared_input))
    xaug = np.zeros((n_, input_size))
    xaug[:, model.ns:] = fixed_input

    def f(t, x):
        xaug_ = np.copy(xaug)
        xaug_[:, :model.ns] = np.reshape(x, (n_, model.ns))
        fx = model.source_model.f(xaug_[None,:])[0].numpy()
        return np.reshape(fx, (n_ * model.ns))

    x0 = np.reshape(x0, (model.ns * n_))

    # Simulate all
    sol = spi.solve_ivp(f, (0, T1), x0, method='RK45', max_step=max_step)
    if not sol.success:
        raise ValueError(message)

    x = np.reshape(sol.y, (n_, model.ns, -1))
    t = sol.t
    converging = is_converging(x, t, int(0.1*len(t)), 1e-3)

    # Simulate the non-converging
    periodic = np.zeros_like(converging, dtype=bool)
    period = np.full(len(periodic), np.nan)

    if np.any(~converging):
        sol2 = spi.solve_ivp(f, (0, T2), x0, method='RK45', max_step=min(0.5, max_step))

        if not sol2.success:
            raise ValueError(message)

        x = np.reshape(sol2.y, (n_, model.ns, -1))
        t = sol2.t

        mask = t > T1
        for i in range(n_):
            if not converging[i]:
                periodic[i], period[i] = is_periodic(x[i,:][:,mask], t[mask], 0.1, min_period=2.)

    return (t, x, converging, periodic, period)



def find_attractors_network(model, w, thetareg, thetasub, us=0., n=10, init_range=(-2,2), T1=400, T2=1200, max_step=1.):

    nreg = model.nreg
    assert thetareg.shape == (nreg, model.mreg)
    assert thetasub.shape == (model.msub,)

    x0 = np.random.uniform(init_range[0], init_range[1], size=(n, nreg * model.ns))
    Ap = model.Ap.numpy()
    bp = model.bp.numpy()
    w = w.astype('float32')

    has_network = int(model.source_model.network_input)
    has_shared  = int(model.source_model.shared_input)
    input_size = (model.ns + model.mreg + model.msub + has_network + has_shared)

    xaugf = np.zeros((nreg, input_size))
    xaugf[:, model.ns:model.ns+model.mreg] = thetareg
    xaugf[:, model.ns+model.mreg:model.ns+model.mreg+model.msub] = thetasub
    if has_shared: xaugf[:, -1] = us

    def f(t, x):
        x = x.reshape((nreg, model.ns))
        y = (np.dot(Ap, x.T) + bp)[0,:]
        u = np.dot(w, y)

        xaug = np.copy(xaugf)
        xaug[:, :model.ns] = x
        if has_network:
            xaug[:, -has_network-has_shared] = u

        fx = model.source_model.f(xaug).numpy().reshape(nreg * model.ns)
        return fx

    # Simulate all
    xs = []
    ts = []

    converging = np.zeros(n, dtype=bool)
    periodic = np.zeros(n, dtype=bool)
    period = np.full(n, np.nan)

    for i in range(n):
        sol = spi.solve_ivp(f, (0, T1), x0[i], method='RK45', max_step=max_step)
        if not sol.success:
            raise ValueError(message)

        x = np.reshape(sol.y, (nreg, model.ns, -1))
        t = sol.t

        converging[i] = is_converging(np.reshape(x, (1, nreg*model.ns, len(t))), t, int(0.1*len(t)), 1e-3)[0]

        if not converging[i]:
            sol2 = spi.solve_ivp(f, (0, T2), x0[i], method='RK45', max_step=min(0.5, max_step))
            if not sol2.success:
                raise ValueError(message)

            x = np.reshape(sol2.y, (nreg, model.ns, -1))
            t = sol2.t

            # Recheck convergence using this detailed simulation
            converging[i] = is_converging(np.reshape(x, (1, nreg*model.ns, len(t))), t, int(0.1*len(t)), 1e-3)[0]
            mask = t > T1
            if not converging[i]:
                periodic[i], period[i] = is_periodic(sol2.y[:,mask], sol2.t[mask], 0.1, min_period=2.)

        ts.append(t)
        xs.append(x)

    return (ts, xs, converging, periodic, period)
