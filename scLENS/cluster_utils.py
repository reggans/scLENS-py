import numpy as np
import torch
import scipy.spatial
import scipy
import igraph as ig
import leidenalg as la
from sklearn.neighbors import kneighbors_graph
from sklearn.metrics.pairwise import cosine_distances
from scipy.stats import norm
from scipy.cluster.hierarchy import linkage

from numba import cuda
from joblib import Parallel, delayed, wrap_non_picklable_objects
from tqdm_joblib import tqdm_joblib

from .scLENS import scLENS

import random, math

# -----------------------GENERAL FUNCTIONS-----------------------

def snn(X, n_neighbors=20, min_weight=1/15, metric='cosine'):
    graph = kneighbors_graph(X, n_neighbors=n_neighbors, metric=metric).toarray()
    # graph = torch.tensor(graph).to(device)
    
    dist = np.stack([get_snn_distance(graph, node) for node in range(graph.shape[0])])

    dist[dist < min_weight] = 0

    for i, j in np.argwhere(dist):
        dist[j, i] = dist[i, j]

    return dist

def get_snn_distance(graph, node):
    row = graph[node]
    idx = np.nonzero(row)
    dist = np.zeros_like(row, dtype=np.float32)

    for i in idx:
        union = graph[i] + row
        dist[i] = 1 - np.sum(union == 2) / np.count_nonzero(union)
    
    return dist.flatten()
    
def find_clusters(X, 
                  n_neighbors=20, 
                  min_weight=1/15, 
                  metric='cosine',
                  res=1.2,
                  n_iterations=-1):
    """
    Find the clustering of the data using the Leiden algorithm,
    using SNN to construct graph an calculate weights

    Parameters
    ----------
    X: np.ndarray
        Data to be clustered
    n_neighbors: int
        Number of nearest neighbors considered in NN graph
    min_weight:
        Minimum weight of the resulting SNN graph
    res: float
        Resolution of the Leiden algorithm; Higher values tend to yield more clusters
    
    Returns
    -------
    np.ndarray
        Array of cluster number of each data point
    """

    dist = snn(X, 
               n_neighbors=n_neighbors, 
               min_weight=min_weight, 
               metric=metric)
    
    G = ig.Graph.Weighted_Adjacency(dist, mode='undirected')
    partition = la.find_partition(G,
                                  la.RBConfigurationVertexPartition,
                                  weights=G.es['weight'],
                                  n_iterations=n_iterations,
                                  resolution_parameter=res)
    
    labels = np.zeros(X.shape[0])
    for i, cluster in enumerate(partition):
        for element in cluster:
            labels[element] = i + 1
    
    return labels

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

def calculate_score(clusters, n, reps, device='cpu'):
    if device == 'gpu':
        if cuda.is_available():
            return calculate_score_gpu(clusters, n, reps)
        else:
            print('GPU is not available, function will be run in CPU')
            return calculate_score_cpu(clusters, n, reps)
    elif device == 'cpu':
        return calculate_score_cpu(clusters, n, reps)
    else:
        raise Exception("Device not recognized. Please choose one of 'cpu' or 'gpu'")

def calculate_score_gpu(clusters, n, reps):
    """
    Score calculation on GPU
    """
    score = np.zeros((n, n), dtype=np.csingle)
    score_device = cuda.to_device(score)

    threadsPerBlock = (16, 16)
    blocksPerGrid_x = math.ceil(n / threadsPerBlock[0])
    blocksPerGrid_y = math.ceil(n / threadsPerBlock[1])
    blocksPerGrid = (blocksPerGrid_x, blocksPerGrid_y)
    
    for row in clusters:
        x_device = cuda.to_device(row)
        outer_equality_kernel[blocksPerGrid, threadsPerBlock](x_device, score_device)
    
    score = score_device.copy_to_host()
    score = np.where(score.real > 0, percent_match(score, reps), 0)
    
    del score_device
    cuda.current_context().memory_manager.deallocations.clear()
    return score

@cuda.jit
def outer_equality_kernel(x, out):
    """
    GPU kernel score calculation algorithm
    """
    tx, ty = cuda.grid(2)

    if tx < x.shape[0] and ty < x.shape[0]:
        if x[tx] == -1 or x[ty] == -1:
            out[tx, ty] += 1j
        elif x[tx] == x[ty]:
            out[tx, ty] += 1

def calculate_score_cpu(clusters, n, reps, n_jobs=None):
    """
    Calculate score on CPU
    """
    score = np.zeros((n, n), dtype=np.csingle)

    for row in clusters:
        parallel = Parallel(n_jobs=n_jobs)
        parallel(outer_equality(row, idx, score) for idx in range(row.shape[0]))
    
    score = np.where(score.real > 0, percent_match(score, reps), 0)
    return score

@delayed
def outer_equality(x, idx, out):
    """
    CPU score calculation algorithm
    """
    if x[idx] == -1:
        out[:, idx] += 1j
        return
    
    for i in range(x.shape[0]):
        if x[i] == x[idx]:
            out[i, idx] += 1
        elif x[i] == -1:
            out[i, idx] += 1j

# -----------------------CHOOSER FUNCTIONS-----------------------
            
def group_silhouette(sil, labels):
    """
    Computes average per-cluster silhouette score 
    """
    sil_grp = list()
    for cls in set(labels):
        idx = np.where(labels == cls)
        sil_grp.append(np.mean(sil[idx]))
    return sil_grp   

def percent_match(x, reps):
    """
    Percentage of co-clustering
    """
    return np.divide(x.real, (reps - x.imag), where=x.imag!=reps)

# -----------------------MULTIK FUNCTIONS-----------------------

def rPAC(consensus, x1=0.1, x2=0.9):
    """"""
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
    
# ------------------------SCSHC FUNCTIONS-------------------------
def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = np.linalg.cholesky(B)
        return True
    except np.linalg.LinAlgError:
        return False

# From https://github.com/alan-turing-institute/bocpdms/blob/master/nearestPD.py
def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd
    
    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """

    B = (A + A.T) / 2
    _, s, V = np.linalg.svd(B)

    H = np.dot(V.T, np.dot(np.diag(s), V))

    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3

    spacing = np.spacing(np.linalg.norm(A))
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = np.eye(A.shape[0])
    k = 1
    while not isPD(A3):
        mineig = np.min(np.real(np.linalg.eigvals(A3)))
        A3 += I * (-mineig * k**2 + spacing)
        k += 1

    return A3

class Dendrogram():
    def __init__(self, linkage) -> None:
        self.linkage = linkage
        self.n_samples = linkage.shape[0] + 1
        self.root = int(np.max(linkage[:, :2]) + 1)
        self.cache = [None] * self.n_samples
    
    def get_subtree_leaves(self, root):
        # TODO MEMORY OPTIMIZATION: Delete cache if called non-recursively
        if root < self.n_samples:
            return [(root, 0)]
        
        idx = root - self.n_samples
        if self.cache[idx] is not None:
            return self.cache[idx]
        
        leaves = list()
        left, right = self.get_children(root)
        leaves.extend([(x, 0) for (x, _) in self.get_subtree_leaves(left)])
        leaves.extend([(x, 1) for (x, _) in self.get_subtree_leaves(right)])
        self.cache[idx] = leaves.copy()

        assert len(leaves) == self.linkage[idx][3] # Sanity check

        return leaves
    
    def get_children(self, root):
        if root < self.n_samples:
            return []
        idx = root - self.n_samples
        return [int(x) for x in self.linkage[idx][:2]]
    
    def get_score(self, root):
        if root < self.n_samples:
            return 0
        idx = root - self.n_samples
        return self.linkage[idx][2] 

def ward_linkage(X, labels):
    # ess1 = calculate_ess(X[labels==0])
    # ess2 = calculate_ess(X[labels==1])
    # ess = calculate_ess(X)
    # return (ess - (ess1 + ess2)) / X.shape[0]
    n1 = np.sum(labels == 0)
    n2 = np.sum(labels == 1)
    mean1 = np.mean(X[labels==0], 0)
    mean2 = np.mean(X[labels==1], 0)
    dist = cosine_distances(mean1.reshape(1, -1), mean2.reshape(1, -1)).item()
    return np.sqrt(2 * n1 * n2 / (n1 + n2)) * dist

def calculate_ess(X):
    return np.sum(cosine_distances(X, np.mean(X, 0).reshape(1, -1)))

def poisson_dispersion_stats(X):
    n = np.sum(X, 1)
    pis = np.sum(X, 0) / np.sum(X)
    mu = pis.reshape(-1, 1) @ n.reshape(1, -1)
    mu = mu.T
    y2 = np.square(X - mu) / mu

    disp = np.sum(y2, 0) / y2.shape[0]

    return np.sqrt(y2.shape[0]) * (disp - 1) / np.sqrt(np.var(y2, 0))

def fit_model(X, on_genes, nPC):
    on_counts = X[:, on_genes] # c x g
    cov = np.cov(on_counts.T) # g x g
    cov = np.atleast_2d(cov)
    means = np.mean(on_counts, 0) # g
    
    sigmas = np.log(((np.diag(cov) - means) / means**2) + 1) # g
    mus = np.log(means) - 0.5 * sigmas # g
    mus_sum = mus.reshape(-1, 1) @ np.ones((1, mus.shape[0])) + np.ones((mus.shape[0], 1)) @ mus.reshape(1, -1) # g x g
    sigmas_sum = sigmas.reshape(-1, 1) @ np.ones((1, sigmas.shape[0])) + np.ones((sigmas.shape[0], 1)) @ sigmas.reshape(1, -1) # g x g
    with np.errstate(divide='ignore', invalid='ignore'):
        rhos = np.log(cov / np.exp(mus_sum + 0.5 * sigmas_sum) + 1) # g x g
    rhos[np.isnan(rhos)] = -10
    rhos[np.isinf(rhos)] = -10
    np.fill_diagonal(rhos, sigmas)

    vals, vecs = np.linalg.eigh(rhos)
    nPC = min(nPC, vals.shape[0])
    vals = vals[-nPC:]
    vecs = vecs[:, -nPC:]
    pos = vals > 0

    on_cov_sub = vecs[:, pos] @ np.sqrt(np.diag(vals[pos]))
    on_cov = on_cov_sub @ on_cov_sub.T
    np.fill_diagonal(on_cov, np.diag(rhos))
    on_cov_PD = nearestPD(on_cov)
    on_cov_sqrt = scipy.linalg.cholesky(on_cov_PD).T

    return np.mean(X, 0), mus, on_cov_sqrt 

def generate_null_stats(X, params, on_genes, nPC):
    lambdas = params[0].astype(np.float64)
    on_means = params[1].astype(np.float64)
    on_cov_sqrt = params[2].astype(np.float64)

    num_gen = min(X.shape[0], 1000)
    null = np.zeros((num_gen, X.shape[1]))

    idx = np.zeros(X.shape[1], dtype=bool)
    idx[on_genes] = True

    rng = np.random.default_rng()
    null[:, np.logical_not(idx)] = rng.poisson(lambdas[np.logical_not(idx)], (num_gen, np.sum(np.logical_not(idx))))

    y = on_cov_sqrt @ rng.normal(size=(num_gen, int(np.sum(idx)))).reshape(int(np.sum(idx)), num_gen) + on_means.reshape(-1, 1)
    y = np.exp(y).T
    null[:, idx] = rng.poisson(y)
    
    null = null[np.sum(null, 1) > 0][:, np.sum(null, 0) > 0]
    null_gm = truncated_sclens(null, nPC)
    dist = scipy.spatial.distance.pdist(null_gm, 'cosine')
    hc = Dendrogram(linkage(dist, method='ward'))
    # qual = hc.get_score(hc.root)
    leaves = np.array(hc.get_subtree_leaves(hc.root))
    leaf_idx = leaves[:, 0]
    leaf_labels = leaves[:, 1]
    qual = ward_linkage(null_gm[leaf_idx], leaf_labels)
    return qual

def test_significance(X, labels, nPC, score, alpha_level, n_jobs=None):
    if X.shape[0] < 2:
        return 1
    
    nPC = min(nPC, X.shape[1])
    
    X_transform = truncated_sclens(X, nPC)

    score = ward_linkage(X_transform, labels)

    phi_stat = poisson_dispersion_stats(X)
    check_means = np.sum(X, 0)
    on_genes = np.nonzero(((norm.sf(phi_stat)) < 0.05) & (check_means != 0.0))[0]

    params = fit_model(X, on_genes, nPC)

    pool = list()
    parallel = Parallel(n_jobs=n_jobs)
    pool.extend(parallel(delayed(generate_null_stats)(X, params, on_genes, nPC) for _ in range(10)))
    
    mean, std = norm.fit(pool)
    pval = norm.sf(score, loc=mean, scale=std)
    if pval < 0.1 * alpha_level or pval > 10 * alpha_level:
        print('Mean:', mean, 'Std:', std, 'Score:', score)
        return pval
    
    pool.extend(parallel(delayed(generate_null_stats)(X, params, on_genes, nPC) for _ in range(40)))
    
    mean, std = norm.fit(pool)
    pval = norm.sf(score, loc=mean, scale=std)
    print('Mean:', mean, 'Std:', std, 'Score:', score)
    return pval

def preprocess(X):
    l1_norm = np.linalg.norm(X, ord=1, axis=1)
    X = np.transpose(X.T / l1_norm)
    X = np.log(1 + X)

    # Z-score normalization
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    X = (X - mean) / std

    # L2 normalization
    l2_norm = np.linalg.norm(X, ord=2, axis=1)
    X = np.transpose(X.T / l2_norm) * np.mean(l2_norm)
    X = np.transpose(X.T - np.mean(X, axis=1))
    return X

def truncated_sclens(X, nPC):
    X = preprocess(X)
    _, vecs = np.linalg.eigh(X @ X.T / X.shape[1])
    vecs = vecs[:, -nPC:]
    return X @ X.T @ vecs