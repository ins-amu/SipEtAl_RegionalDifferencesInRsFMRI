
import os
import sys
import argparse
import glob

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.gridspec import GridSpec
import pandas as pd
import tensorflow as tf

import ndsvae as ndsv


def paramplot2d(ax, treg, title, xlabel, ylabel, scalar):
    cmap = matplotlib.cm.magma

    if scalar is not None:
        norm = matplotlib.colors.Normalize(vmin=np.min(scalar), vmax=np.max(scalar))

    plt.title(title)

    nreg = treg.shape[0]
    for j in range(nreg):
        color = cmap(norm(scalar[j])) if (scalar is not None) else 'k'
        plt.plot([treg[j,0,0], treg[j,0,0]], [treg[j,1,0]-treg[j,1,1], treg[j,1,0]+treg[j,1,1]], color=color)
        plt.plot([treg[j,0,0]-treg[j,0,1], treg[j,0,0]+treg[j,0,1]], [treg[j,1,0], treg[j,1,0]], color=color)
    plt.grid()
    plt.xlim(-3,3)
    plt.ylim(-3,3)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    if scalar is not None:
        plt.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap, norm=norm))


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


def plot_prediction(state, model, ds, subjects, filename):
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
        dfc = ndsv.calc_dfc(sims.y[0,i,:,0,ntdrop:], 41)
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
            us = ndsv.models.gen_normal(usmu, uslv)

        if model.latent == 'eta':
            etamu, etalv, x0mu, x0lv, tregmu, treglv, tsubmu, tsublv = model.encode(subj_ind, yobs, u)
            eta  = ndsv.models.gen_normal(etamu, etalv)
            x0   = ndsv.models.gen_normal(x0mu, x0lv)
            treg = ndsv.models.gen_normal(tregmu, treglv)
            tsub = ndsv.models.gen_normal(tsubmu, tsublv)

            x = model.source_model.simulate(x0, eta, u_upsampled[:,:,None], us, treg, tsub, model.nt*model.upsample_factor)
            xmu = np.mean(x, axis=0)
            xstd = np.std(x, axis=0)
            etamu = etamu[0]
            etastd = tf.exp(0.5*etalv[0])

        elif model.latent == 'x':
            xmu, xlv, tregmu, treglv, tsubmu, tsublv = model.encode(subj_ind, yobs, u)
            treg = ndsv.models.gen_normal(tregmu, treglv)
            tsub = ndsv.models.gen_normal(tsubmu, tsublv)

            x = ndsv.models.gen_normal(xmu, xlv)
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


def phaseplot2d(ax, model, c1, c2, x, thetareg, thetasub, u=0., ushared=0.):
    plt.sca(ax)
    fx = ndsv.models.evalf(model, x, thetareg=thetareg, thetasub=thetasub, u=u, ushared=ushared)
    plt.quiver(x[c1], x[c2], fx[:,:,c1].T, fx[:,:,c2].T, scale=6)


def pairwise(t):
    it = iter(t)
    return zip(it,it)


def rename_params(d):
    rename_table = {"nf": "f_units", "a": "lambda_a"}
    return {rename_table.get(k, k): v for k, v in d.items()}

def retype_params(d):
    type_table = {"ns": int, "mreg": int, "msub": int, "f_units": int, "lambda_a": float}
    return {k: (type_table[k](v) if k in type_table else v) for k, v in d.items()}


def get_model(modelname, param_string, ds):
    nt = ds.y.shape[-1]
    params = retype_params(rename_params(dict(pairwise(param_string.split("_")))))
    common_params = dict(nreg=ds.nreg, nsub=ds.nsub, nt=nt, nobs=1, lambda_x=0.01, alpha=0.01, lambda_a=0.0,
                         fix_observation=False, encoder_units=32, activation='relu')

    common_params.update(params)

    # Because GPU implementation cannot handle zero parameters
    kl_sub_factor = 1.0
    if common_params['msub'] == 0:
        common_params['msub'] = 1
        common_params['kl_sub_factor'] = 1000.

    if modelname == 'AN':
        model = ndsv.models.RegX(**common_params, prediction='normal', shared_input=False)
    elif modelname == 'AS':
        model = ndsv.models.RegX(**common_params, prediction='normal', shared_input=True, network_input=False)
    elif modelname == 'AB':
        model = ndsv.models.RegX(**common_params, prediction='normal', shared_input=True)
    elif modelname == 'BN':
        model = ndsv.models.RegX(**common_params, prediction='gmm', ng=2, shared_input=False)
    elif modelname == 'BS':
        model = ndsv.models.RegX(**common_params, prediction='gmm', ng=2, shared_input=True, network_input=False)
    elif modelname == 'BB':
        model = ndsv.models.RegX(**common_params, prediction='gmm', ng=2, shared_input=True)
    elif modelname == 'CN':
        model = ndsv.models.RegEta(**common_params, prediction='normal', shared_input=False)
    else:
        raise ValueError(f"Model '{modelname}' not recognized")

    return model

def strip_param(params_string, param_name):
    params = params_string.split("_")
    occ_indices = [i for i,p in enumerate(params) if p == param_name]
    if len(occ_indices) == 0:
        return params_string, None
    ind = occ_indices[0]
    params.pop(ind)
    param_value = params.pop(ind)
    return "_".join(params), param_value


def get_test_data(ds, model, examples):
    nsub, nreg, _, _ = ds.y.shape
    tds = ndsv.training._prep_training_dataset(ds, batch_size=1, mode='region-upsampled',
                                               upsample_factor=model.upsample_factor, shuffle=False)
    tds = list(tds.as_numpy_iterator())
    data = []
    for isub, ireg in examples:
        data.append(tds[isub*nreg + ireg])
    return data


def fit_dataset(dataset_file, modelname, param_string, run_id, output_dir, train_ratio=1.0):
    os.makedirs(output_dir)

    ds = ndsv.Dataset.from_file(dataset_file)
    model = get_model(modelname, param_string, ds)

    ndata = int(train_ratio * ds.nreg * ds.nsub)
    batch_size = 64
    ipe = int(np.ceil(ndata/batch_size))

    train_mask = np.zeros((ds.nsub, ds.nreg), dtype=bool)
    train_mask[np.unravel_index(np.random.choice(ds.nsub*ds.nreg, ndata, replace=False),
                                (ds.nsub, ds.nreg))] = True
    np.save(os.path.join(output_dir, "train_mask.npy"), train_mask)

    lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay([300000*ipe, 600000*ipe], [1e-3, 3e-4, 1e-4])
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

    runner = ndsv.training.Runner(optimizer=optimizer, batch_size=batch_size,
                                  epochs=1001,
                                  nsamples=8,
                                  clip_gradients=(-1000,1000),
                                  betax=1.,
                                  betap=ndsv.training.linclip(0, 500, 0., 1.),
                                 )

    # Create callback function
    test_data = get_test_data(ds, model, [(0,50), (1,50), (2,50)])
    test_subjects = [0,1,2]

    os.makedirs(os.path.join(output_dir, "img"))
    os.makedirs(os.path.join(output_dir, "models"))
    def callback(state, model):
        if (state.epoch % 50 != 0):
            return
        for i in (np.r_[:ds.thetareg.shape[2]] if ds.thetareg.shape[2] > 0 else [None]):
            plot_params_reg(state, model, ds, i,  os.path.join(output_dir, f"img/params-reg{i}_{state.epoch:05d}.png"), subj_inds=test_subjects)
        for i in (np.r_[:ds.thetasub.shape[1]] if ds.thetasub.shape[1] > 0 else [None]):
            plot_params_sub(state, model, ds, i,  os.path.join(output_dir, f"img/params-sub{i}_{state.epoch:05d}.png"))
        plot_prediction(state, model, ds, test_subjects, os.path.join(output_dir, f"img/pred_{state.epoch:05d}.png"))
        plot_input(state, model, ds, test_subjects, os.path.join(output_dir, f"img/input_{state.epoch:05d}.png"))
        plot_projection(state, model, test_data, os.path.join(output_dir, f"img/proj_{state.epoch:05d}.png"))

        params = model.encode_subjects(ds.w, ds.y)
        params.save(os.path.join(output_dir, f"img/params_{state.epoch:05d}.npz"))

        model.save_weights(os.path.join(output_dir, f"models/model_{state.epoch:05d}"))


    # Run the training
    hist = ndsv.training.train(model, ds, runner, fh=sys.stdout, callback=callback, mask_train=train_mask)
    dfh = hist.as_dataframe()

    # Save the history
    dfh.to_csv(os.path.join(output_dir, "hist.csv"), index=False)

    # Save the model
    model.save_weights(os.path.join(output_dir, "model"))



def simulate(dataset_file, subject, modelname, param_string, fit_direc, nsamples, param_file, sim_file):
    ds = ndsv.Dataset.from_file(dataset_file)
    model = get_model(modelname, param_string, ds)
    model.load_weights(os.path.join(fit_direc, "model"))

    if subject == 'all':
        # Get the parameters and save
        params = model.encode_subjects(ds.w, ds.y)
        params.save(param_file)

        # Run simulations
        nt = ds.y.shape[-1]
        warmup = int(nt // 2)
        sims = model.simulate_subjects(ds.w, nt+warmup, n=nsamples, ic=params.ic,
                                       thetareg=params.thetareg, thetasub=params.thetasub)
        sims.save(sim_file)
    else:
        isub = int(subject)

        # Get the parameters and save
        params = model.encode_subjects(ds.w[[isub]], ds.y[[isub]], subj_ind=[isub])
        params.save(param_file)

        # Run simulations
        nt = ds.y.shape[-1]
        warmup = int(nt // 2)
        sims = model.simulate_subjects(ds.w[[isub]], nt+warmup, n=nsamples, ic=params.ic,
                                       thetareg=params.thetareg, thetasub=params.thetasub)
        sims.save(sim_file)



def select_run(testcase, modelname, config, runs, scenario="main"):
    """Select run based on FC similarity"""

    ds = ndsv.Dataset.from_file(f"../run/{scenario}/{testcase}/dataset.npz")
    nsub, nreg, _, nt = ds.y.shape
    triu = np.triu_indices(nreg, k=1)

    fc1 = np.array([np.corrcoef(ds.y[i,:,0,:]) for i in range(nsub)])
    meanpc = np.zeros(len(runs))

    for k, run in enumerate(runs):
        sims = ndsv.GeneratedData.from_file(
            f"../run/{scenario}/{testcase}/model{modelname}/{config}/run{run:02d}/simulations.npz")
        nsamples = sims.y.shape[0]

        pc = np.zeros((nsub, nsamples))
        for i in range(nsub):
            for j in range(nsamples):
                fc2 = np.corrcoef(sims.y[j,i,:,0,-nt:])
                pc[i,j] = np.corrcoef(fc1[i][triu], fc2[triu])[1,0]
        meanpc[k] = np.mean(pc)

    return runs[np.nanargmax(meanpc)]


def select_run_fc(testcase, modelname, config, runs, scenario="main"):
    """Select run based on precomputed FC similarity"""

    ds = ndsv.Dataset.from_file(f"../run/{scenario}/{testcase}/dataset.npz")
    nsub, nreg, _, nt = ds.y.shape

    fcs = [np.load(f"../run/{scenario}/{testcase}/model{modelname}/{config}/run{run:02d}/simulations/fc.npz")
           for run in runs]
    meanpc = [np.mean(fc['similarity']) for fc in fcs]

    return runs[np.nanargmax(meanpc)]


def join_params(ps):
    nsub = len(ps)
    thetasub = np.concatenate([p.thetasub for p in ps], axis=0)
    thetareg = np.concatenate([p.thetareg for p in ps], axis=0)
    ic = np.concatenate([p.ic for p in ps], axis=0)
    x = np.concatenate([p.x for p in ps], axis=0)

    if all([p.us is None for p in ps]):
        us = None
    elif all([p.us is not None for p in ps]):
        us = np.concatenate([p.us for p in ps], axis=0)
    else:
        raise ValueError("Mixed existence of us")

    return ndsv.Params(thetasub, thetareg, ic, x=x, eta=None, us=us)


def load_params(direc, subjects):
    filenames = [os.path.join(direc, f"params_{isub:03d}.npz") for isub in subjects]
    params = [ndsv.Params.from_file(filename) for filename in filenames]
    return join_params(params)


def calc_fc(dataset_file, input_files, output_file):
    nsub = len(input_files)
    fcs = []

    nsamples = ndsv.GeneratedData.from_file(input_files[0]).y.shape[0]
    ds = ndsv.Dataset.from_file(dataset_file)
    nsub, nreg, _, nt = ds.y.shape

    fc_emp = np.zeros((nsub, nreg, nreg))
    fc_sim = np.zeros((nsub, nreg, nreg, nsamples))
    similarity = np.zeros((nsub, nsamples))

    for i in range(nsub):
        fc_emp[i] = np.corrcoef(ds.y[i,:,0,:])

    for i, simfile in enumerate(input_files):
        sims = ndsv.GeneratedData.from_file(simfile)
        # sims.y shape: (nsim, nsub, nreg, nobs, nt), with nsub == 1 and nobs == 1

        for j in range(nsamples):
            fc_sim[i,:,:,j] = np.corrcoef(sims.y[j,0,:,0,-nt:])

    # PC similarity
    triu = np.triu_indices(nreg, k=1)
    for i in range(nsub):
        for j in range(nsamples):
            similarity[i,j] = np.corrcoef(fc_emp[i,:,:][triu], fc_sim[i,:,:,j][triu])[0,1]

    np.savez(output_file, emp=fc_emp, sim=fc_sim, similarity=similarity)



if __name__ == "__main__":
    cmd = sys.argv.pop(1)

    if cmd == "fit":
        parser = argparse.ArgumentParser()
        parser.add_argument('--train-ratio', default=1.0, type=float)
        parser.add_argument('dataset_file')
        parser.add_argument('model')
        parser.add_argument('param_string')
        parser.add_argument('run_id', type=int)
        parser.add_argument('output_dir')
        parser.add_argument('nthreads', type=int)
        args = parser.parse_args()

        # print(f"Running with {args.nthreads} threads")
        # tf.config.threading.set_inter_op_parallelism_threads(args.nthreads)
        # tf.config.threading.set_intra_op_parallelism_threads(args.nthreads)
        fit_dataset(args.dataset_file, args.model, args.param_string, args.run_id, args.output_dir, train_ratio=args.train_ratio)

    elif cmd == "simulate":
        parser = argparse.ArgumentParser()
        parser.add_argument('--subject', default='all', type=str)
        parser.add_argument('--nthreads', default=1, type=int)
        parser.add_argument('dataset_file')
        parser.add_argument('model')
        parser.add_argument('param_string')
        parser.add_argument('fit_direc')
        parser.add_argument('nsamples', type=int)
        parser.add_argument('param_file')
        parser.add_argument('sim_file')
        args = parser.parse_args()

        tf.config.threading.set_inter_op_parallelism_threads(args.nthreads)
        tf.config.threading.set_intra_op_parallelism_threads(args.nthreads)

        simulate(args.dataset_file, args.subject, args.model, args.param_string, args.fit_direc, args.nsamples,
                 args.param_file, args.sim_file)


    elif cmd == "fc":
        parser = argparse.ArgumentParser()

        parser.add_argument('--dataset', required=True, type=str)
        parser.add_argument('--input', nargs='+', required=True, type=str)
        parser.add_argument('--output', required=True, type=str)
        args = parser.parse_args()

        calc_fc(args.dataset, args.input, args.output)
