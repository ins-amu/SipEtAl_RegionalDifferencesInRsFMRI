
import os
import sys
import multiprocessing as mp

import numpy as np

import ndsvae as ndsv

from balloon_windkessel import balloon_windkessel


def normalize(w):
    w = w/np.max(w)
    return w


def tavg(x, window_len):
    y = np.zeros((x.shape[0], x.shape[1], x.shape[2] // window_len))
    for i in range(y.shape[2]):
        y[:, :, i] = np.mean(x[:, :, i*window_len:(i+1)*window_len], axis=2)
    return y


def generate_rww(name, ws, gs, Io, sig, T, dt, dtrep, Tskip=0.0, wp=1.0, observation_model='avg', seed=None):
    """
    Generate RWW dynamics

    ws ... connectomes of shape (nsub, nreg, nreg)
    gs ... coupling coefficients (nsub)
    Io ... RWW parameters (nsub, nreg)
    T  ... simulation length (in ms)
    dt ... simulation time step (in ms)
    dtrep ... reporting time step (take temporal average; in ms)
    Tskip ... time to be skipped at the beginning
    """

    if seed is not None:
        np.random.seed(seed)

    nsub, nreg, _ = ws.shape

    wp = wp * np.ones((nsub, nreg))
    Io = Io * np.ones((nsub, nreg))
    sig = sig * np.ones((nsub, nreg))
    gs = gs * np.ones(nsub)

    ntf = T // dt

    tau_s = 100
    gamma = 0.641/1000.
    Jn = 0.2609

    a = 270.
    b = 108.
    d = 0.154
    h = lambda x: (a*x - b) / (1 - np.exp(-d*(a*x - b)))

    s = np.zeros((nsub, nreg, ntf))
    s[:, :, 0] = np.random.uniform(0.2, 0.8, size=(nsub, nreg))
    bold = np.zeros((nsub, nreg, ntf))

    # Simulate
    for isub in range(nsub):
        eta = np.random.normal(size=(nreg, ntf))
        w = ws[isub]
        G = gs[isub]

        for i in range(ntf-1):
            x = wp[isub,:]*Jn*s[isub,:,i] + Io[isub,:] + G*Jn*np.dot(w, s[isub,:,i])
            s[isub,:,i+1] = s[isub,:,i] + dt *(-s[isub,:,i]/tau_s
                                               + (1 - s[isub,:,i])*gamma*h(x)) + sig[isub,:]*eta[:, i]

        bold[isub], _, _, _, _ = balloon_windkessel(s[isub,:,:], 0.001*dt)

    # Subsample, skip initial, add state dimension, normalize
    subsample_factor = int(dtrep // dt)

    if observation_model == 'bold':
        y = bold[:,:,::subsample_factor]
    elif observation_model == 'avg':
        y = tavg(s,  subsample_factor)

    s_subsampled = s[:,:,::subsample_factor]
    t = np.linspace(0, T, y.shape[2])

    # Skipt time
    y = y[:,:,t > Tskip]
    t = t[t > Tskip]

    # Normalize
    y -= np.mean(y)
    y /= np.std(y)

    ds = ndsv.Dataset(name, t=t, x=s_subsampled[:,:,None,:], y=y[:,:,None,:],
                      thetareg=np.stack([sig, Io, wp], axis=-1),
                      thetasub=gs[:,None], w=ws)

    return ds


def generate_hopf(name, ws, gs, f, a, sig, T, dt, dtrep, Tskip=0.0, seed=None):
    """
    Hopf oscillators as desccribed in Deco et al, Sci Rep (2017)
    """

    if seed is not None:
        np.random.seed(seed)

    nsub, nreg, _ = ws.shape
    omega = 2*np.pi*f

    ntf = int(T / dt)

    x = np.zeros((nsub, nreg, 2, ntf))
    x[:, :, :, 0] = np.random.normal(0, 0.3, size=(nsub, nreg, 2))

    # Simulate
    for isub in range(nsub):
        eta = np.random.normal(size=(nreg, 2, ntf))
        w = ws[isub]
        G = gs[isub]

        for i in range(ntf-1):
            for j in range(nreg):
                x[isub,j,0,i+1] = (x[isub,j,0,i]
                                   + dt * ((a[isub,j] - x[isub,j,0,i]**2 - x[isub,j,1,i]**2)*x[isub,j,0,i]
                                           - omega[isub,j]*x[isub,j,1,i]
                                           + G*np.dot(w[j,:], x[isub,:,0,i]-x[isub,j,0,i]))
                                   + sig * eta[j,0,i+1])
                x[isub,j,1,i+1] = (x[isub,j,1,i]
                                   + dt * ((a[isub,j] - x[isub,j,0,i]**2 - x[isub,j,1,i]**2)*x[isub,j,1,i]
                                           + omega[isub,j]*x[isub,j,0,i]
                                           + G*np.dot(w[j,:], x[isub,:,1,i]-x[isub,j,1,i]))
                                   + sig * eta[j,1,i+1])

    # Subsample
    t = np.linspace(0, T, ntf)
    fac = int(dtrep / dt)
    t = t[::fac]
    x = x[:,:,:,::fac]

    # Skip beginning
    mask = (t > Tskip)
    x = x[:, :, :, mask]

    # Build observations
    y = x[:, :, [0], :]

    # Normalize
    y -= np.mean(y)
    y /= np.std(y)

    thetareg = np.moveaxis(np.array([a, f]), [0,1,2], [2,0,1])
    ds = ndsv.Dataset(name, t=t, x=x, y=y, thetareg=thetareg, thetasub=gs[:,None], w=ws)
    return ds


def generate_burster(n, sig):
    # Fold/homoclinic model from Izhikevich (2000)
    # Eq (25) and (26)
    # Constants from Fig 59

    T = 250
    dt = 0.01
    nt = int(T / dt) + 1
    t = np.linspace(0, T, nt)
    x = np.zeros((n, 3, nt))

    V1 = -0.01
    V2 = 0.15
    V3 = 0.1
    V4 = 0.05
    El = -0.5
    Ek = -0.7
    Eca = 1.0
    gl = 0.5
    gk = 2.0
    gca = 1.2
    mu = 0.005

    minf = lambda V: 0.5*(1 + np.tanh((V - V1)/V2))
    winf = lambda V: 0.5*(1 + np.tanh((V - V3)/V4))
    lamb = lambda V: (1./3.)*(np.cosh((V - V3)/(2*V4)))

    sig = np.array(sig)[:, None]

    for j in range(n):
        eta = np.random.normal(0, sig, size=(3, nt))

        x[j,:, 0] = [np.random.uniform(-0.4, 0.0), np.random.uniform(0, 0.2), np.random.uniform(-0.08, -0.05)]
        for i in range(nt-1):
            # V equation
            x[j,0,i+1] = x[j,0,i] + dt*(-x[j,2,i] - gl*(x[j,0,i] - El) - gk*x[j,1,i]*(x[j,0,i] - Ek)
                                    - gca*minf(x[j,0,i])*(x[j,0,i] - Eca)) + eta[0,i]

            # w equation
            x[j,1,i+1] = x[j,1,i] + dt*(lamb(x[j,0,i]) * (winf(x[j,0,i]) - x[j,1,i])) + eta[1,i]

            # u equation
            x[j,2,i+1] = x[j,2,i] + dt*(mu*(0.2 + x[j,0,i])) + eta[2,i]

    t = t[    nt//5::40]
    x = x[:,:,nt//5::40]

    return t, x


def dss2gendata(dss):
    gendata = ndsv.models.GeneratedData(
        x=np.array([ds.x for ds in dss]),
        y=np.array([ds.y for ds in dss]),
        thetareg=np.array([ds.thetareg for ds in dss]),
        thetasub=np.array([ds.thetasub for ds in dss])
    )
    return gendata


def pmfm_get_meanfc(args):
    g, w, Io, wp, sig = args

    np.random.seed(42)
    gs = g * np.ones(nsub)
    ds = generate_rww("pMFM", w, gs=gs, Io=Io, sig=sig, wp=wp, T=240000, dt=10, dtrep=720, Tskip=120000,
                      observation_model='avg')
    triu = np.triu_indices(nreg, k=1)
    fc = [np.mean(np.corrcoef(ds.y[i,:,0,:])[triu]) for i in range(nsub)]
    return fc

def pmfm_get_g(w, Io, wp, sig, nthreads):
    gs = np.linspace(0.16, 0.22, 31)
    args = [(g, w, Io, wp, sig) for g in gs]
    with mp.Pool(nthreads) as pool:
        mfc = pool.map(pmfm_get_meanfc, args)
    mfc = np.array(mfc)
    inds = np.argmax(mfc, axis=0)
    return gs[inds]


def add_homotopic_connections(w):
    HOMOTOPIC_CONN_STRENGTH_PERCENTILE = 97

    nreg = w.shape[0]
    triu = np.triu_indices(nreg, k=1)

    val = np.percentile(w[triu], HOMOTOPIC_CONN_STRENGTH_PERCENTILE)
    inds = np.r_[:nreg//2]

    whom = np.copy(w)
    whom[inds, inds + nreg//2] = np.maximum(whom[inds, inds + nreg//2], val)
    whom[inds + nreg//2, inds] = np.maximum(whom[inds + nreg//2, inds], val)

    return whom


if __name__ == "__main__":

    testcase, *args = sys.argv[1].split("_")
    outfile = sys.argv[2]
    imgdir = sys.argv[3]
    nsim = int(sys.argv[4])
    surrfile = sys.argv[5]
    nthreads = int(sys.argv[6])

    np.random.seed(42)

    if testcase == "hopf":
        name = "Hopf-s08-n68"

        subjects = ['100307', '100408', '101107', '101309', '101915', '103111', '103414', '103818']
        w = np.array([normalize(np.genfromtxt(f"data/{subj}/weights.txt")) for subj in subjects])
        nsub, nreg, _ = w.shape

        gs = np.linspace(0, 0.7, nsub)
        a = np.random.uniform(-1, 1, size=(nsub, nreg))
        f = np.random.uniform(0.03, 0.07, size=(nsub, nreg))

        def fun(seed):
            np.random.seed(seed)
            return generate_hopf(name, w, gs, f, a, sig=0.02, T=205, dt=0.02, dtrep=1.0, Tskip=25)


    elif testcase == "pmfm":
        name = "pMFM-s08-n68"

        subjects = ['100307', '100408', '101107', '101309', '101915', '103111', '103414', '103818']
        w = np.array([normalize(np.genfromtxt(f"data/{subj}/weights.txt")) for subj in subjects])
        nsub, nreg, _ = w.shape

        # gs = np.array([0.192, 0.175, 0.176, 0.185, 0.190, 0.174, 0.177, 0.186])
        Io = 0.295 * np.ones((nsub, nreg))
        wp = 1.0 * np.ones((nsub, nreg))
        sig = np.maximum(np.random.normal(0.04, 0.01, (nsub, nreg)), 0.001)
        gs = pmfm_get_g(w, Io, wp, sig, nthreads)

        observation_model = args.pop()

        def fun(seed):
            np.random.seed(seed)
            ds = generate_rww("pMFM", w, gs=gs, Io=Io, sig=sig, wp=wp, T=984000, dt=10, dtrep=720, Tskip=120000,
                              observation_model=observation_model)

            return ds


    elif testcase in ["hcp", "hcp100"]:
        name = "hcp"
        with open(os.environ["SUBJECTS_FILE"]) as fh:
            subjects_all = fh.read().splitlines()

        if   testcase == "hcp":    subjects = subjects_all[:8]
        elif testcase == "hcp100": subjects = subjects_all[:4]

        w = np.array([normalize(np.genfromtxt(f"data/{subj}/weights.txt")) for subj in subjects])
        if args[0] == "linw":
            pass
        elif args[0] == "logw":
            q = 3
            w = np.log10(10**q * w + 1.)/q
        elif args[0] == "linwhom":
            w = np.array([add_homotopic_connections(w_) for w_ in w])
        elif args[0] == "logwhom":
            q = 3
            w = np.log10(10**q * w + 1.)/q
            w = np.array([add_homotopic_connections(w_) for w_ in w])

        nsub, nreg, _ = w.shape

        sampling_period = 0.72   # seconds

        preproc = args[1]
        y = np.array([np.load(f"data/{subj}/rfMRI_REST1-LR_{preproc}.npz")['data'] for subj in subjects])[:,:,None,:]

        nsub, nreg, _, nt = y.shape
        x = np.full((nsub, nreg, 1, nt), np.nan)
        t = sampling_period * np.r_[:nt]
        thetareg = np.full((nsub, nreg, 0), np.nan)
        thetasub = np.full((nsub, 0), np.nan)

        nsim = 0  # Overwrite it, we cannot get no surrogates
        def fun(seed):
            return ndsv.Dataset(name, t, x, y, thetareg, thetasub, w)


    # Now, run the thing
    with mp.Pool(processes=nthreads) as pool:
        dss = pool.map(fun, range(nsim+1))

    # Main dataset
    ds = dss[0]
    ds.save(outfile)
    ds.plot_obs(imgdir)

    # Surrogates
    surr = dss2gendata(dss[1:])
    surr.save(surrfile)
