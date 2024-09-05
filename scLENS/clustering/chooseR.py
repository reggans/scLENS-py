import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_samples
import scipy.stats
from joblib import Parallel, delayed, wrap_non_picklable_objects

import random

from .utils import find_clusters, calculate_score

from tqdm.auto import tqdm 
from tqdm_joblib import tqdm_joblib

def chooseR(X,
            reps=100,
            size=0.8,
            resolutions=None,
            device='gpu',
            n_jobs=None,
            silent=False):
    """
    Chooses the best resolution from a set of possible resolutions
    by repeatedly subsampling and clustering the data,
    calculating the silhouette score for each clustering,
    and selecting the lowest resolution whose median score passes a threshold

    Parameters
    ----------
    X: np.ndarray
        Data to be clustered
    reps: int
        Number of sampling repetitions for each resolution
    size: float
        Portion of data to be subsampled. Must be between 0 and 1
    resolutions: list of float
        Possible resolutions to choose from
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
        resolutions = [0.3, 0.5, 0.8, 1, 1.2, 1.6, 2, 4, 6, 8]
    resolutions = set(resolutions)
    stats = list()
    for res in tqdm(resolutions,
                    desc='Calculating scores', 
                    total=len(resolutions),
                    disable=silent):
        stats_row = [res]
        cls = find_clusters(X, res=res)
        stats_row.append(len(np.unique(cls)))
        stats_row.append(cls)

        clusters = construct_sample_clusters(X, 
                                             reps=reps, 
                                             size=size, 
                                             res=res, 
                                             n_jobs=n_jobs, 
                                             disable=True)

        score = calculate_score(clusters, X.shape[0], reps, device=device)
        
        score = 1 - score
        np.fill_diagonal(score, 0)

        if len(np.unique(cls)) == 1:
            stats_row.append(0)
            stats_row.append(0)
            stats.append(stats_row)
            continue
        
        sil = silhouette_samples(score, cls, metric='precomputed')
        sil_grp = group_silhouette(sil, cls)
        sil_grp = (sil_grp,)

        # stats_row.append(confidence_interval(np.median, sil_grp)[0])
        stats_row.append(scipy.stats.bootstrap(sil_grp, np.median, n_resamples=25000).confidence_interval.low)
        stats_row.append(np.median(sil_grp))

        stats.append(stats_row)
    
    stats = pd.DataFrame(stats, columns=['res', 'n_clusters', 'clustering', 'low_med', 'med']).sort_values(by=['n_clusters'])
    threshold = max(stats['low_med'])
    stats = stats[stats['med'] >= threshold]

    if len(stats) == 1:
        return list(stats['clustering'])
    return list(stats['clustering'].iloc[0])

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
        clusters = parallel(sample_cluster(X, k=k, res=res, filler=filler) for _ in range(reps))
    return clusters

@delayed
@wrap_non_picklable_objects
def sample_cluster(X, k, res=1.2, filler=-1):
    """
    Sample and cluster data
    """
    row = np.zeros(X.shape[0])
    row.fill(filler)
    sample = random.sample(range(X.shape[0]), k)
    cls = find_clusters(X[sample], res=res)
    np.put(row, sample, cls)
    return row

def group_silhouette(sil, labels):
    """
    Computes average per-cluster silhouette score 
    """
    sil_grp = list()
    for cls in set(labels):
        idx = np.where(labels == cls)
        sil_grp.append(np.mean(sil[idx]))
    return sil_grp   