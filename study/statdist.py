"""
Measures of distance of two distributions
"""

import numpy as np
from ortools.linear_solver import pywraplp
from scipy.special import gamma

def kl_divergence(p, q, k):
    """
    k-nearest-neighbor estimation of Kullback-Leibler divergence between samples from two multivariate distributions.

    See: Pérez-Cruz F.: Kullback-Leibler Divergence Estimation of Continuous Distributions. ISIT 2008.
    """
    n, d = p.shape
    m, d2 = q.shape

    assert d == d2
    assert n > k
    assert m > k

    # Equation (14)
    rk = np.array([np.sort(np.linalg.norm(p - p[i], axis=1))[k]   for i in range(n)])
    sk = np.array([np.sort(np.linalg.norm(q - p[i], axis=1))[k-1] for i in range(n)])
    dk = d/ n * np.sum(np.log(sk / rk)) + np.log(m / (n - 1))   # Typo in the original paper: rk and sk should be switched!

    return dk


def log_likelihood(p, q, k):
    """
    k-nearest-neighbor approximation of the likelihood of samples p given model M represented by samples q.
    """

    n, d = p.shape
    m, d2 = q.shape

    assert d == d2
    assert m > k

    sk = np.array([np.sort(np.linalg.norm(q - p[i], axis=1))[k-1] for i in range(n)])
    qpdf = k / m * gamma(d/2 + 1) / (np.pi**(d/2) * sk**d)

    return np.sum(np.log(qpdf))


def bhattacharyya(p, q, bins):
    """
    Bhattacharyya distance estimation from samples
    """
    n, d = p.shape
    m, d2 = q.shape
    assert d == d2
    assert len(bins) == d

    hp, edges = np.histogramdd(p, bins=[b[2] for b in bins], range=[(b[0], b[1]) for b in bins])
    hq, edges = np.histogramdd(q, bins=[b[2] for b in bins], range=[(b[0], b[1]) for b in bins])

    hp /= np.sum(hp)
    hq /= np.sum(hq)

    bc = np.sum(np.sqrt(hp.flatten() * hq.flatten()))
    db = -np.log(bc)

    return db


def wasserstein(p, q, bins):
    """
    Wasserstein distance estimation from samples
    """

    n, d = p.shape
    m, d2 = q.shape
    assert d == d2
    assert len(bins) == d

    hp, edges = np.histogramdd(p, bins=[b[2] for b in bins], range=[(b[0], b[1]) for b in bins])
    hq, edges = np.histogramdd(q, bins=[b[2] for b in bins], range=[(b[0], b[1]) for b in bins])
    hp = hp.flatten() / np.sum(hp)
    hq = hq.flatten() / np.sum(hq)
    nbins = len(hp)

    centers = [(es[1:] + es[:-1]) / 2. for es in edges]
    coords = np.array([mg.flatten() for mg in np.meshgrid(*centers, indexing='ij')]).T

    solver = pywraplp.Solver("EMDSolver", pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)
    objective = solver.Objective()
    objective.SetMinimization()

    sum_fr = [0 for _ in range(nbins)]
    sum_to = [0 for _ in range(nbins)]

    for ifr in range(nbins):
        for ito in range(nbins):
            f_ij = solver.NumVar(0, solver.infinity(), f"f_{ito},{ifr}")
            sum_fr[ifr] += f_ij
            sum_to[ito] += f_ij
            objective.SetCoefficient(f_ij, np.linalg.norm(coords[ifr] - coords[ito]))

    for i in range(nbins):
        solver.Add(sum_fr[i] == hq[i])
        solver.Add(sum_to[i] == hp[i])

    status = solver.Solve()
    if status not in [solver.OPTIMAL, solver.FEASIBLE]:
        raise Exception("EMDSolver failed")

    return objective.Value()
