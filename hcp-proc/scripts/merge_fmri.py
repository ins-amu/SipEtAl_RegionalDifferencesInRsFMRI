
import sys

import numpy as np
import scipy.stats as stats
import nibabel as nib

dict_subcort = {
    "Left-Cerebellum": "Left-Cerebellum-Cortex",  "Right-Cerebellum": "Right-Cerebellum-Cortex",
    "Left-Thalamus": "Left-Thalamus-Proper", "Right-Thalamus": "Right-Thalamus-Proper",
    "Left-Accumbens": "Left-Accumbens-area", "Right-Accumbens": "Right-Accumbens-area"
}

def translate(name, group):
    if group == 'cort':
        if   name[:2] == "L_":   return f"ctx-lh-{name[2:]}"
        elif name[:2] == "R_":   return f"ctx-rh-{name[2:]}"
        else:                    return name
    elif group == 'subcort':
        name = "-".join([a.capitalize() for a in name.split("_")])
        if   name[-5:] == "-Left":  name = f"Left-{name[:-5]}"
        elif name[-6:] == "-Right": name = f"Right-{name[:-6]}"
        return dict_subcort.get(name, name)
    else:
        raise ValueError("Unexpected group {group}")

def regress_mean(x):
    nreg, nt = x.shape
    mean = np.mean(x, axis=0)
    xr = np.zeros_like(x)
    for i in range(nreg):
        a, b, rval, pval, stderr = stats.linregress(mean, x[i])
        xr[i] = x[i] - (a*mean + b)
    return xr


def merge_fmri(ptseries_cort_file, ptseries_subcort_file, target_regions_file, outfile):

    # Existing data
    ptseries_cort    = nib.load(ptseries_cort_file)
    ptseries_subcort = nib.load(ptseries_subcort_file)
    regions = (  [translate(p.name, 'cort')    for p in ptseries_cort.header.matrix[1].parcels]
               + [translate(p.name, 'subcort') for p in ptseries_subcort.header.matrix[1].parcels])
    pdata = np.concatenate([ptseries_cort.get_fdata().T, ptseries_subcort.get_fdata().T])
    _, nt = pdata.shape

    # Desired data
    target_regions = list(np.genfromtxt(target_regions_file, usecols=(0,), dtype=str))
    nreg = len(target_regions)
    data = np.zeros((nreg, nt))
    for i, region in enumerate(target_regions):
        ind = regions.index(region)
        data[i, :] = pdata[ind]

    # Regress and normalize
    # data = regress_mean(data)
    # data -= np.mean(data, axis=0)
    data -= np.mean(data, axis=1)[:, None]
    data /= np.std(data, axis=1)[:, None]

    # Save
    np.savez(outfile, data=data)


if __name__ == "__main__":
    ptseries_cort = sys.argv[1]
    ptseries_subcort = sys.argv[2]
    target_regions_file = sys.argv[3]
    outfile = sys.argv[4]

    merge_fmri(ptseries_cort, ptseries_subcort, target_regions_file, outfile)
