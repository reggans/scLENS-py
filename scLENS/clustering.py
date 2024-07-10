import numpy as np
import pandas as pd
import scipy
from scipy.cluster.hierarchy import linkage
from sklearn.metrics import silhouette_samples
from resample.bootstrap import confidence_interval
from joblib import Parallel

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib

from .cluster_utils import find_clusters, construct_sample_clusters, calculate_score, \
    group_silhouette, calculate_one_minus_rpac, Dendrogram, test_significance

from collections import Counter

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

        clusters = construct_sample_clusters(X, 
                                             reps=reps, 
                                             size=size, 
                                             res=res, 
                                             n_jobs=n_jobs, 
                                             disable=True)

        score = calculate_score(clusters, X.shape[0], reps, device=device)
        
        score = 1 - score
        np.fill_diagonal(score, 0)

        sil = silhouette_samples(score, cls, metric='precomputed')
        sil_grp = group_silhouette(sil, cls)

        stats_row.append(confidence_interval(np.median, sil_grp)[0])
        stats_row.append(np.median(sil_grp))

        stats.append(stats_row)
    
    stats = pd.DataFrame(stats, columns=['res', 'n_clusters', 'low_med', 'med']).sort_values(by=['n_clusters'])
    threshold = max(stats['low_med'])
    stats = stats[stats['med'] >= threshold]

    if len(stats) == 1:
        return stats['res']
    return stats['res'].iloc[0]


def multiK(X,
           resolutions=None,
           reps=100,
           size=0.8,
           x1=0.1, x2=0.9,
           metric=None,
           device='gpu',
           n_jobs=None,
           silent=False,
           **kwargs):
    """
    Predicts the optimal number of clusters k by 
    testing a range of different resolution parameters,
    and scoring the results based on the observed frequency 
    and rPAC for each k
    
    Parameters
    ----------
    X:  np.ndarray
        Data to be clustered
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
    
    n = len(resolutions) * reps
    clusters = np.zeros((n, X.shape[0]))
    ks = np.zeros(n)

    for i, res in tqdm(enumerate(resolutions), 
                       desc='Constructing samples', 
                       total=len(resolutions), 
                       disable=silent):
        offset = reps * i
        sample_cls = construct_sample_clusters(X,
                                               reps=reps, 
                                               size=size, 
                                               res=res, 
                                               n_jobs=n_jobs,
                                               disable=True)
        
        clusters[offset:offset+reps] = sample_cls
        ks[offset:offset+reps] = [len(np.unique(cls)) - 1 for cls in sample_cls] # accomodate for label of dropped data
    
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
    
    if len(result) == 1:
        opt_res = result[0]
    else:
        if metric is None:
            from sklearn.metrics import silhouette_score
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

def scSHC(X,
          X_transform,
          alpha=0.05,
          device=None,
          n_jobs=None):
    """
    Daniel Müllner, fastcluster: Fast Hierarchical, 
    Agglomerative Clustering Routines for R and Python, 
    Journal of Statistical Software, 53 (2013), no. 9, 1-18,
    https://doi.org/10.18637/jss.v053.i09.
    """
    nPC = X_transform.shape[1]
    dist = scipy.spatial.distance.pdist(X_transform, 'cosine')
    dend = Dendrogram(linkage(dist, method='ward'))
    test_queue = [dend.root]
    clustering = np.zeros(X.shape[0]) - 1 # -1 for unassigned cluster
    cluster_idx = 0

    while(test_queue):
        test = test_queue.pop()
        test_leaves = np.array(dend.get_subtree_leaves(test))
        score = dend.get_score(test)

        alpha_level = alpha * ((test_leaves.shape[0] - 1) / (X.shape[0] - 1))

        X_test = X[test_leaves[:, 0]]
        label_test = test_leaves[:, 1]
        X_test = X_test[:, np.sum(X_test, 0) > 0]

        n_cluster1 = np.sum(label_test)
        n_cluster0 = len(label_test) - n_cluster1

        print(f'ClusterID: {test}, Test Shape: {X_test.shape}')
        
        if min(n_cluster1, n_cluster0) < 20:
            sig = 1
        else:
            sig = test_significance(X_test, label_test, nPC, score, alpha_level, n_jobs)

        if (sig < alpha_level):
            test_queue.extend(dend.get_children(test))
        else:
            test_idx = [x for (x, _) in test_leaves]
            clustering[test_idx] = cluster_idx
            cluster_idx += 1

            print(f'CLUSTER IDENTIFIED; Significance: {sig}, Total clusters: {cluster_idx}')
    
    return clustering