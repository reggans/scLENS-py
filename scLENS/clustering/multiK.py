from sklearn.metrics import silhouette_score
import numpy as np
import scipy
import torch
from joblib import Parallel, delayed, wrap_non_picklable_objects
import scipy.stats

import random
from collections import Counter

from .utils import find_clusters, calculate_score
from ..scLENS import scLENS

from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib
import matplotlib.pyplot as plt

def multiK(X,
           resolutions=None,
           reps=100,
           size=0.8,
           x1=0.1, x2=0.9,
           metric=None,
           device='gpu',
           n_jobs=None,
           silent=False,
           sclens_kwargs=None,
           **kwargs):
    """
    Predicts the optimal number of clusters k by 
    testing a range of different resolution parameters,
    and scoring the results based on the observed frequency 
    and rPAC for each k
    
    Parameters
    ----------
    X:  np.ndarray
        Raw data to be clustered
    resolutions: list of float
        Range of resolutions to be tested
    reps: int
        Number of sampling repetitions for each resolution
    size: float
        Portion of data to be subsampled. Must be between 0 and 1
    x1: float
        Argument for evaluating rPAC. Must be between 0 and 1
    x2: float
        Argument for evaluating rPAC. Must be between 0 and 1
    metric: function or None,
        Metric to choose the final resolution parameter
    device: One of ['cpu', 'gpu']
        Device to run the scoring on. 'cpu' will run scoring on CPU with n_jobs parallel jobs
    n_jobs: int or None
        joblib n_jobs; Number of CPU jobs to run in parallel
    
    Returns
    -------
    float
        The chosen best resolution
    """
    if resolutions is None:
        resolutions = np.arange(0.05, 2, 0.05)
    else:
        resolutions = np.array(resolutions)
    
    if sclens_kwargs is None:
        sclens_kwargs = dict()
    
    n_res = len(resolutions)
    n = n_res * reps
    clusters = np.zeros((n, X.shape[0]))
    ks = np.zeros(n)

    for i in tqdm(range(reps), 
                  desc='Constructing samples', 
                  total=reps, 
                  disable=silent):
        sample_idx = random.sample(range(X.shape[0]), int(X.shape[0] * size))
        full_cls = np.zeros((n_res, X.shape[0]))
        full_cls.fill(-1)

        scl = scLENS(**sclens_kwargs)
        scl.preprocess(X[sample_idx, :])
        X_transform = scl.fit_transform()

        del scl
        torch.cuda.empty_cache()

        offset = n_res * i
        sample_cls = construct_sample_clusters(X_transform,
                                               reps=reps, 
                                               size=size,  
                                               res=resolutions, 
                                               n_jobs=n_jobs,
                                               disable=True)
        
        full_cls[:, sample_idx] = sample_cls
        
        clusters[offset:offset+n_res] = full_cls
        ks[offset:offset+n_res] = [len(np.unique(cls)) - 1 for cls in full_cls] # accomodate for label of dropped data
    
    k_runs = [x[1] for x in sorted(Counter(ks).items())]
    k_unique = np.unique(ks)
    
    with tqdm_joblib(desc='Calculating rPAC', total=len(k_unique)):
        parallel = Parallel(n_jobs=n_jobs)
        one_minus_rpac = parallel(calculate_one_minus_rpac(clusters[ks==k], 
                                                            X.shape[0],
                                                            x1,
                                                            x2,
                                                            device=device)
                                                            for k in k_unique)

    points = np.array(list(zip(one_minus_rpac, k_runs, strict=True)))

    if len(points) < 3:
        hull_points = points
        hull_k = k_unique
    else:
        chull = scipy.spatial.ConvexHull(points)
        hull_points = points[chull.vertices]
        hull_k = k_unique[chull.vertices]
    best_rpac = np.argmax(hull_points[:, 0])
    best_run = np.argmax(hull_points[:, 1])

    if best_rpac == best_run:
        opt_points = np.array([hull_points[best_rpac]])
        opt_k = np.array([hull_k[best_rpac]])
    elif best_rpac < best_run:
        opt_points = hull_points[best_rpac:best_run + 1]
        opt_k = hull_k[best_rpac:best_run + 1]
    else:
        opt_points = np.concatenate([hull_points[:best_run + 1], hull_points[best_rpac:]])
        opt_k = np.concatenate([hull_k[:best_run + 1], hull_k[best_rpac:]])

    plt.plot(points[:, 0], points[:, 1], 'ko')
    plt.plot(opt_points[:, 0], opt_points[:, 1], 'ro')
    for i, k in enumerate(opt_k):
        plt.annotate(str(k), (opt_points[i, 0], opt_points[i, 1]))
    plt.show()

    result = list()
    for k in opt_k:
        idx = np.nonzero(ks == k)[0] // reps
        res = Counter(list(resolutions[idx])).most_common(1)[0][0]
        result.append(res)
    
    print(f'Optimal resolutions: {result}')
    
    if len(result) == 1:
        opt_res = result[0]
    else:
        if metric is None:
            metric = silhouette_score

        best = 0
        opt_res = None
        for res in result:
            cls = find_clusters(X, res=res)
            if len(np.unique(cls)) == 1 and metric == silhouette_score:
                continue
            score = metric(X, cls, **kwargs)
            if score > best:
                best = score
                opt_res = res

    return opt_res

# Helper functions

def construct_sample_clusters(X,
                              filler=-1,
                              reps=100,
                              size=0.8,
                              res=1.2,
                              n_jobs=None,
                              **kwargs):
    """
    Creates clusterings based on a subset of the dataset
    """
    k = int(X.shape[0] * size)

    with tqdm_joblib(desc='Constructing samples', total=reps, **kwargs):
        parallel = Parallel(n_jobs=n_jobs)
        clusters = parallel(cluster(X, res=r) for r in res)
    return clusters

@delayed
@wrap_non_picklable_objects
def cluster(X, res=1.2):
    cls = find_clusters(X, res=res)
    return cls

def rPAC(consensus, x1=0.1, x2=0.9):
    """"""
    # Get triangular matrix, conserve original 0s
    consensus[consensus == 0] = -1
    consensus = np.ravel(np.tril(consensus))
    consensus = consensus[consensus != 0]
    consensus[consensus == -1] = 0

    cdf = scipy.stats.ecdf(consensus).cdf
    pac = cdf.evaluate(x2) - cdf.evaluate(x1)
    zeros = np.sum(consensus == 0) / consensus.shape[0]
    return pac / (1 - zeros)

@delayed
def calculate_one_minus_rpac(cluster, n, x1, x2, device='gpu'):
    n_k = cluster.shape[0]
    
    consensus = calculate_score(cluster, n, n_k, device=device)

    res = 1 - rPAC(consensus, x1, x2)
    return res