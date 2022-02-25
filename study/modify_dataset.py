
import sys

import numpy as np
import ndsvae as ndsv


def perturb_connectome(w, eps, seed=42):
    np.random.seed(seed)

    wr = np.random.normal(0, 1, size=w.shape)
    wpert = np.maximum(w + eps*wr, 0.)
    wpert /= np.max(wpert)

    return wpert


def modify_dataset(input_file, output_file, img_dir, variant):
    ds = ndsv.Dataset.from_file(input_file)

    args = variant.split("_")

    if args[0] == "logw":
        q = 3
        ds.w = np.log10(10**q * ds.w + 1.)/q
    elif args[0] == "eps":
        eps = float(args[1])
        seed = int(args[2])
        ds.w = np.array([perturb_connectome(w, eps, seed) for w in ds.w])

    ds.save(output_file)
    ds.plot_obs(img_dir)


if __name__ == "__main__":
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    img_dir = sys.argv[3]
    variant = sys.argv[4]

    modify_dataset(input_file, output_file, img_dir, variant)
