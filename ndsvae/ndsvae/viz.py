
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import tensorflow as tf

from .Dataset import calc_dfc
from . import models

def paramplot2(p1, p2, names=None):
    nsub, nreg, m = p1.shape
    assert p2.shape == (nsub, nreg, 2, 2)
    if names is not None:
        assert len(names) == m


    fig = plt.figure(figsize=(6*m, 4*nsub))

    gs = GridSpec(nsub, 6, width_ratios=[4, 0.3, 1.5]*m, wspace=0.1)

    cmap = matplotlib.cm.get_cmap('plasma')

    for i in range(nsub):
        for j in range(m):
            ax = plt.subplot(gs[i,3*j])

            ax.add_patch(matplotlib.patches.Ellipse((0,0), 2, 2, color='k', fill=False))
            norm = matplotlib.colors.Normalize(vmin=np.min(p1[:,:,j]), vmax=np.max(p1[:,:,j]))

            for k in range(nreg):
                plt.errorbar(p2[i,k,0,0], p2[i,k,1,0],
                             xerr=p2[i,k,0,1], yerr=p2[i,k,1,1],
                             color=cmap(norm(p1[i,k,j])), lw=2)
            plt.xlim(-3.0, 3.0); plt.ylim(-3.0, 3.0)

            if j == 0:
                plt.ylabel(f"Subject {i}")
            if i == 0 and (names is not None):
                plt.title(names[j])

            ax2 = plt.subplot(gs[i, 3*j+1])
            cb = matplotlib.colorbar.ColorbarBase(ax2, cmap=cmap, norm=norm, orientation='vertical')

    return fig


def paramplot1d(ax, theta, title, xlabel, ylabel, scalar):
    n, _ = theta.shape

    if scalar is None:
        scalar = np.r_[:n]

    assert len(scalar) == n

    plt.sca(ax)
    for i in range(n):
        plt.plot([scalar[i], scalar[i]], [theta[i,0]-theta[i,1], theta[i,0]+theta[i,1]], color='k')
        plt.scatter(scalar[i], theta[i,0], s=10, color='k')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.ylim(-3,3)


def paramplot2d(ax, theta, title=None, xlabel=None, ylabel=None, scalar=None):
    """Two-dimensional cross plot of parameters.

    Parameters
    ----------
    ax: matplotlib.axes._axes.Axes
        Axes to plot in.    
    theta: np.array
        Parameters to plot. Shape: (n, 2, 2). 
        n is the number of points. Dimension 1 is the x/y position.
        Dimension 2 are the means and standard deviations.
    title: str, optional
        Title of the plot
    xlabel: str, optional:
        x-label
    ylabel: str, optional:
        y-label
    scalar: np.array, optional
        Optional scalar to color the crosses. Shape: (n)
    """

    cmap = matplotlib.cm.magma

    plt.sca(ax)

    if scalar is not None:
        norm = matplotlib.colors.Normalize(vmin=np.min(scalar), vmax=np.max(scalar))

    plt.title(title)

    n = theta.shape[0]
    for j in range(n):
        color = cmap(norm(scalar[j])) if (scalar is not None) else 'k'
        plt.plot([theta[j,0,0], theta[j,0,0]], [theta[j,1,0]-theta[j,1,1], theta[j,1,0]+theta[j,1,1]], color=color)
        plt.plot([theta[j,0,0]-theta[j,0,1], theta[j,0,0]+theta[j,0,1]], [theta[j,1,0], theta[j,1,0]], color=color)
    plt.grid()
    plt.xlim(-3,3)
    plt.ylim(-3,3)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if scalar is not None:
        plt.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm))


def plot_params_reg(state, model, ds, param_ind, filename, subj_inds=None):
    if subj_inds is None:
        subj_inds = np.r_[:ds.nsub]

    params = model.encode_subjects(ds.w[subj_inds], ds.y[subj_inds], subj_inds)
    treg = params.thetareg
    nsub = len(subj_inds)

    view_dict = {2: [(0,1)],
                 3: [(0,1), (0,2), (1,2)],
                 4: [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)],
                 5: [(0,1), (0,2), (0,3), (0,4), (1,2), (3,4)]}

    if treg.shape[2] == 1:
        plt.figure(figsize=(5,nsub*5))
        for i, isub in enumerate(subj_inds):
            ax = plt.subplot(nsub, 1, i+1)
            scalar = ds.thetareg[isub,:,param_ind] if (param_ind is not None) else None
            paramplot1d(ax, treg[isub,:,0,:], "", "Scalar", "theta 0", scalar)

    elif treg.shape[2] in view_dict:
        views = view_dict[treg.shape[2]]
        nviews = len(views)

        plt.figure(figsize=(nviews*5, nsub*5))
        for i, isub in enumerate(subj_inds):
            for j, inds in enumerate(views):
                ax = plt.subplot2grid((nsub, nviews), (i,j))
                scalar = ds.thetareg[isub,:,param_ind] if (param_ind is not None) else None
                paramplot2d(ax, treg[i][:,inds,:], f"Subject {isub}", f"theta {inds[0]}", f"theta {inds[1]}", scalar)
    else:
        return

    plt.suptitle(f"Epoch {state.epoch}")
    plt.tight_layout()
    plt.savefig(filename, transparent=False, facecolor='white')
    plt.close()


def plot_params_sub(state, model, ds, param_ind, filename, subj_inds=None):
    if subj_inds is None:
        subj_inds = np.r_[:ds.nsub]
    tsub = model.tsub.numpy()[subj_inds,:,:]
    theta = np.zeros_like(tsub)
    theta[:,:,0] = tsub[:,:,0]
    theta[:,:,1] = np.exp(0.5 * tsub[:,:,1])
    scalar = ds.thetasub[:,param_ind] if param_ind is not None else None

    view_dict = {2: [(0,1)],
                 3: [(0,1), (0,2), (1,2)],
                 4: [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)],
                 5: [(0,1), (0,2), (0,3), (0,4), (1,2), (3,4)]}

    if theta.shape[1] == 1:
        plt.figure(figsize=(5,5))
        paramplot1d(plt.gca(), theta[:,0,:], "", "Scalar", "theta 0", scalar)

    elif theta.shape[1] in view_dict:
        views = view_dict[theta.shape[1]]
        nviews = len(views)

        plt.figure(figsize=(nviews*5,5))
        for j, inds in enumerate(views):
            ax = plt.subplot2grid((1, nviews), (0, j))
            paramplot2d(ax, theta[:,inds,:], "", f"theta {inds[0]}", f"theta {inds[1]}", scalar)

    else:
         return

    plt.suptitle(f"Epoch {state.epoch}")
    plt.tight_layout()
    plt.savefig(filename, transparent=False, facecolor='white')
    plt.close()



def plot_simulation(state, model, ds, subjects, filename):
    nt = ds.y.shape[-1]
    ntdrop = int(nt // 4)
    nsamples = 1
    nsub = len(subjects)

    params = model.encode_subjects(ds.w[subjects], ds.y[subjects], subj_ind=subjects)
    sims = model.simulate_subjects(ds.w[subjects], nt+ntdrop, thetareg=params.thetareg, thetasub=params.thetasub, ic=params.ic, n=nsamples)

    plt.figure(figsize=(24, 8*nsub))
    plt.suptitle(f"Epoch {state.epoch}")

    for i, isub in enumerate(subjects):
        plt.subplot2grid((nsub, 3), (i, 0))
        plt.imshow(sims.y[0,i,:,0,ntdrop:], aspect='auto', vmin=-2.5, vmax=2.5, interpolation='none')
        plt.colorbar()
        plt.title(f"Subject {isub}")

        plt.subplot2grid((nsub, 3), (i, 1))
        plt.imshow(np.corrcoef(sims.y[0,i,:,0,ntdrop:]), vmin=-1, vmax=1, cmap='bwr')
        plt.colorbar()

        plt.subplot2grid((nsub, 3), (i, 2))
        dfc = calc_dfc(sims.y[0,i,:,0,ntdrop:], 41)
        plt.imshow(dfc, cmap='inferno', vmin=0.0, vmax=1)
        plt.colorbar()

    plt.tight_layout()

    plt.savefig(filename, transparent=False, facecolor='white')
    plt.close()


def plot_input(state, model, ds, subjects, filename):
    nsub = len(subjects)
    params = model.encode_subjects(ds.w[subjects], ds.y[subjects], subj_ind=subjects)
    if params.us is None:
        return

    nsub, nt, _ = params.us.shape

    plt.figure(figsize=(10, 2*nsub))
    for i, isub in enumerate(subjects):
        plt.subplot(nsub, 1, i+1)
        plt.plot(np.r_[:nt], params.us[i,:,0], color='b')
        plt.fill_between(np.r_[:nt], params.us[i,:,0] - params.us[i,:,1], params.us[i,:,0] + params.us[i,:,1],
                         color='b', alpha=0.2)
        plt.ylim(-2,2)
        plt.xlim(0, nt-1)
        plt.ylabel(f"Subject {isub}")

    plt.suptitle(f"Common input (Epoch {state.epoch})")
    plt.tight_layout()
    plt.savefig(filename, transparent=False, facecolor='white')
    plt.close()


def phaseplot2d(ax, model, c1, c2, x, thetareg, thetasub, u=0., ushared=0.):
    plt.sca(ax)
    fx = models.evalf(model, x, thetareg=thetareg, thetasub=thetasub, u=u, ushared=ushared)
    plt.quiver(x[c1], x[c2], fx[:,:,c1].T, fx[:,:,c2].T, scale=6)


def plot_projection(state, model, examples, filename):
    nsamples = 20

    n = len(examples)
    nrows = 2 + 2*model.ns
    nsub = model.nsub
    nreg = model.nreg
    nobs = model.nobs
    nt = model.nt

    fig = plt.figure(figsize=(n*12, 2*nrows))
    gs = GridSpec(2*model.ns + 2, 2*n, width_ratios=[1.6,1]*n)

    for k, (subj_ind, yobs, u, u_upsampled) in enumerate(examples):
        subj_ind, yobs, u, u_upsampled = [tf.repeat(d, nsamples, axis=0) for d in [subj_ind, yobs, u, u_upsampled]]

        us = None
        if model.shared_input:
            usmu, uslv = tf.unstack(tf.gather(model.us, subj_ind, axis=0), axis=-1)
            us = models.gen_normal(usmu, uslv)

        if model.latent == 'eta':
            etamu, etalv, x0mu, x0lv, tregmu, treglv, tsubmu, tsublv = model.encode(subj_ind, yobs, u)
            eta  = models.gen_normal(etamu, etalv)
            x0   = models.gen_normal(x0mu, x0lv)
            treg = models.gen_normal(tregmu, treglv)
            tsub = models.gen_normal(tsubmu, tsublv)

            x = model.source_model.simulate(x0, eta, u_upsampled[:,:,None], us, treg, tsub, model.nt*model.upsample_factor)
            xmu = np.mean(x, axis=0)
            xstd = np.std(x, axis=0)
            etamu = etamu[0]
            etastd = tf.exp(0.5*etalv[0])

        elif model.latent == 'x':
            xmu, xlv, tregmu, treglv, tsubmu, tsublv = model.encode(subj_ind, yobs, u)
            treg = models.gen_normal(tregmu, treglv)
            tsub = models.gen_normal(tsubmu, tsublv)

            x = models.gen_normal(xmu, xlv)
            xmu = xmu[0]
            xstd = tf.exp(0.5 * xlv[0])

            xaug = [x, tf.repeat(treg[:,None,:], repeats=nt, axis=1), tf.repeat(tsub[:,None,:], repeats=nt, axis=1)]
            if model.source_model.network_input:
                xaug.append(u[:,:,None])
            if model.source_model.shared_input:
                xaug.append(us[:,:,None])
            xaug = tf.concat(xaug, axis=2)

            xaug = tf.reshape(xaug, (nsamples*nt, -1))
            fx = model.source_model.f(xaug)
            fx = tf.reshape(fx, (nsamples, nt, -1))

            eta = np.zeros_like(x)
            eta[:,:-1,:] = (x[:,1:,:] - x[:,:-1,:] - fx[:,:-1,:]) / tf.exp(0.5 * model.source_model.slv)
            etamu = np.mean(eta, axis=0)
            etastd = np.std(eta, axis=0)

        etaobs = np.random.normal(0, 1, size=(nsamples,nt,nobs))
        ypred = tf.tensordot(x, model.Ap, axes=[[2],[1]]) + model.bp + tf.exp(0.5*model.olv)*etaobs
        yobs = yobs[0]
        u = u[0]

        # Plotting --------------------------------------------------------------------------------

        # Observations
        plt.subplot(gs[0, 2*k])
        plt.plot(yobs[:,0], color='r', label='Observation')
        plt.plot(np.mean(ypred[:,:,0], axis=0), color='tab:blue', label='Fit')
        plt.fill_between(np.r_[:nt], *np.percentile(ypred[:,:,0], [16.0, 84.0], axis=0), alpha=0.3, color='tab:blue')
        plt.xlim(0, nt-1)
        plt.ylabel("Observation")
        plt.ylim(-2.0, 2.0)
        plt.grid()

        # Input
        plt.subplot(gs[1, 2*k])
        plt.plot(u, color='g')
        plt.xlim(0, nt-1)
        plt.ylabel("Network input")
        plt.ylim(-2.5, 2.5)
        plt.grid()

        # Latent states
        for j in range(model.ns):
            plt.subplot(gs[j+2, 2*k])
            plt.plot(xmu[:,j], color='tab:blue')
            plt.fill_between(np.r_[:nt], xmu[:,j]-xstd[:,j], xmu[:,j]+xstd[:,j], alpha=0.3, color='tab:blue')
            plt.xlim(0, nt-1)
            plt.ylabel(f"State [{j}]")
            plt.ylim(-2, 2)
            plt.grid()

        # Noise
        sstd = np.exp(0.5* model.source_model.slv)
        for j in range(model.ns):
            plt.subplot(gs[j+2+model.ns, 2*k])

            # Scaled noise
            plt.plot(etamu[:,j], color='tab:blue')
            plt.fill_between(np.r_[:nt], etamu[:,j]-etastd[:,j], etamu[:,j]+etastd[:,j], alpha=0.3, color='tab:blue')

            # Unscaled noise
            plt.plot(sstd[j] * etamu[:,j], color='tab:red')
            plt.fill_between(np.r_[:nt], sstd[j]*(etamu[:,j]-etastd[:,j]), sstd[j]*(etamu[:,j]+etastd[:,j]),
                             alpha=0.3, color='tab:red')

            plt.ylabel(f"Noise [{j}]")
            plt.xlim(0, nt-1)
            plt.ylim(-2, 2)
            plt.grid()


        coords_dict = { 2: [(0,1)],
                        3: [(0, 1), (0, 2), (1, 2)],
                        4: [(0, 1), (0, 2), (0, 3), (1, 2), (2, 3)],
                        5: [(0, 1), (0, 2), (0, 3), (0, 4), (1, 3), (3, 4)]}
        nquiver = 10
        rng = [-3., 3.]

        if model.ns in coords_dict:
            coords = coords_dict[model.ns]
            for j, (c1, c2) in enumerate(coords):
                plt.subplot(gs[2*j:2*(j+1), 2*k+1])
                plt.plot(np.mean(x[:,:,c1], axis=0), np.mean(x[:,:,c2], axis=0), color='tab:blue', alpha=1.0, lw=0.3)
                for i in range(min(10, nsamples)):
                    plt.plot(x[i,:,c1], x[i,:,c2], color='g', alpha=0.3, lw=1, zorder=-1)
                plt.xlabel(f"State [{c1}]")
                plt.ylabel(f"State [{c2}]")
                xq = [0. for _ in range(model.ns)]
                xq[c1] = np.linspace(rng[0], rng[1], nquiver)
                xq[c2] = np.linspace(rng[0], rng[1], nquiver)
                phaseplot2d(plt.gca(), model, c1, c2, x=xq, thetareg=tregmu[0], thetasub=tsubmu[0], u=0., ushared=0.)
                plt.xlim(rng); plt.ylim(rng)

    plt.tight_layout()
    plt.savefig(filename, transparent=False, facecolor='white')
    plt.close()    