
import os

import numpy as np
import matplotlib.pyplot as plt


def calc_dfc(y, half_window):
    nreg, nt = y.shape
    fc = np.zeros((nt, nreg, nreg))

    for i in range(half_window, nt-half_window):
        fc[i] = np.corrcoef(y[:,i-half_window:i+half_window])

    inds = np.triu_indices(nreg, k=1)
    fc = fc[:, inds[0], inds[1]][half_window:-half_window]
    dfc = np.corrcoef(fc)

    return dfc


class Dataset:
    """Object to hold the simulated or empirical data"""

    def __init__(self, name, t, x, y, thetareg, thetasub, w, description=None):
        self.name = name

        assert t.ndim == 1
        assert x.ndim == 4
        assert y.ndim == 4
        assert thetareg.ndim == 3
        assert thetasub.ndim == 2
        assert w.ndim == 3

        nsub, nreg, ns, nt = x.shape
        nobs = y.shape[2]

        assert y.shape[:3] == (nsub, nreg, nobs)
        assert thetareg.shape[0:2] == (nsub, nreg)
        assert thetasub.shape[0] == nsub
        assert w.shape == (nsub, nreg, nreg)

        self.t = t
        self.x = x
        self.y = y
        self.thetareg = thetareg
        self.thetasub = thetasub
        self.w = w

        self.nsub = nsub
        self.nreg = nreg
        self.nk = ns
        self.nobs = nobs

        self.description = description if description is not None else ""

    @classmethod
    def from_file(cls, filename):
        data = np.load(filename)
        return cls(name=data['name'], t=data['t'], x=data['x'], y=data['y'],
                   thetareg=data['thetareg'], thetasub=data['thetasub'],
                   w=data['w'], description=data['description'])


    def save(self, filename):
        np.savez(filename, t=self.t, x=self.x, y=self.y, thetareg=self.thetareg, thetasub=self.thetasub,
                 w=self.w, name=self.name, description=self.description)


    def plot_obs(self, direc):
        if not os.path.isdir(direc):
            os.makedirs(direc)

        nsub, nreg, nobs, nt = self.y.shape

        for i in range(nsub):
            plt.figure(figsize=(24, 5*nobs))

            for j in range(nobs):
                plt.subplot2grid((nobs, 6), (j, 0), colspan=3)
                plt.imshow(self.y[i, :, j, :], aspect='auto', interpolation='none')
                plt.xlabel("Time")
                plt.ylabel("Regions")
                plt.colorbar()

                plt.subplot2grid((nobs, 6), (j, 3))
                plt.title("FC")
                im = plt.imshow(np.corrcoef(self.y[i, :, j, :]), vmin=-1, vmax=1, cmap='bwr', interpolation='none')
                plt.colorbar(im, fraction=0.03)

                plt.subplot2grid((nobs, 6), (j, 4))
                plt.title("dFC")
                hw = 41
                im = plt.imshow(calc_dfc(self.y[i, :, j, :], half_window=hw), vmin=0, vmax=1, cmap='magma',
                                extent=[hw,nt-hw,nt-hw,hw], interpolation='none')
                plt.colorbar(im, fraction=0.03)

                plt.subplot2grid((nobs, 6), (j, 5))
                plt.title("SC")
                im = plt.imshow(self.w[i], cmap='viridis', interpolation='none')
                plt.colorbar(im, fraction=0.03)


            plt.tight_layout()
            plt.savefig(f"{direc}/obs_sub{i:03d}.png")
            plt.close()


def get_network_input_obs(w, y, comp=0):
    """
    Calculate the network input, using the comp component of the observations
    """

    nsub, nreg, nobs, nt = y.shape
    assert w.shape == (nsub, nreg, nreg)

    yinp = np.zeros((nsub, nreg, 1, nt))
    for i in range(nsub):
        for j in range(nreg):
            yinp[i, j, 0, :] = np.dot(w[i,j,:], y[i, :, comp, :])

    return yinp
