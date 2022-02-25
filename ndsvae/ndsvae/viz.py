
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


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
