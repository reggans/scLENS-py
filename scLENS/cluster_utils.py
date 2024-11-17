import numpy as np
import torch
import scipy.spatial
import scipy
import igraph as ig
import leidenalg as la
from sklearn.metrics.pairwise import cosine_distances, euclidean_distances
from scipy.stats import norm
from scipy.cluster.hierarchy import linkage, fcluster
import scipy.stats as stats
from scipy.sparse.linalg import eigsh
from sklearn.decomposition import TruncatedSVD

from numba import cuda
from joblib import Parallel, delayed, wrap_non_picklable_objects
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib

from .scLENS import scLENS

import random, math

import scanpy as scpy
from sklearn.neighbors import NearestNeighbors

# -----------------------GENERAL FUNCTIONS-----------------------

def snn(X, n_neighbors=20, min_weight=1/15, metric='cosine'):
    # graph = kneighbors_graph(X, n_neighbors=n_neighbors, metric=metric)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors+1, metric=metric).fit(X)
    indices = nbrs.kneighbors(X,return_distance=False)
    indices = indices[:, 1:]

    n_samples = indices.shape[0]
    edges = []
    for i, neighbors in enumerate(indices):
        for neighbor in neighbors:
            edges.append((i,neighbor))
    
    g = ig.Graph(n=n_samples,edges=edges,directed=False)
    weights = np.array(g.similarity_jaccard(pairs=g.get_edgelist()))
    g.es['weight'] = weights
    
    edges_to_delete = [i for i, w in enumerate(weights) if w < min_weight]
    g.delete_edges(edges_to_delete)
    
    return g


def find_clusters(X, 
                n_neighbors=20, 
                min_weight=1/15, 
                metric='cosine',
                res=1.2,
                n_iterations=-1):
    
    G = snn(X, n_neighbors=n_neighbors, min_weight=min_weight, metric=metric)
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
                              metric='cosine',
                              batch_size=20,
                              **kwargs):
    """
    Creates clusterings based on a subset of the dataset
    """
    k = int(X.shape[0] * size)
    clusters = []

    if reps is None:
        if not isinstance(res, (list, tuple, np.ndarray)):
            res_list = [res]
        else:
            res_list = res

        total_tasks = len(res_list)
        for batch_start in tqdm(range(0, total_tasks, batch_size), desc='Batched Sampling', **kwargs):
            batch_end = min(batch_start + batch_size, total_tasks)
            batch_res_list = res_list[batch_start:batch_end]

            with tqdm_joblib(desc='Constructing samples', total=len(batch_res_list), **kwargs):
                parallel = Parallel(n_jobs=n_jobs)
                batch_clusters = parallel(
                    delayed(sample_cluster)(X, k=k, res=res_i, filler=filler, sample=False, metric=metric)
                    for res_i in batch_res_list
                )
            clusters.extend(batch_clusters)
    else:
        total_tasks = reps
        for batch_start in tqdm(range(0, total_tasks, batch_size), desc='Batched Sampling', **kwargs):
            batch_end = min(batch_start + batch_size, total_tasks)
            batch_reps = batch_end - batch_start

            with tqdm_joblib(desc='Constructing samples', total=batch_reps, **kwargs):
                parallel = Parallel(n_jobs=n_jobs)
                batch_clusters = parallel(
                    delayed(sample_cluster)(X, k=k, res=res, filler=filler, metric=metric)
                    for _ in range(batch_reps)
                )
            clusters.extend(batch_clusters)

    return clusters

def sample_cluster(X, k, res=1.2, filler=-1, sample=True, metric='cosine'):
    """
    Sample and cluster data
    """
    if not sample:
        cls = find_clusters(X, res=res, metric=metric)
        return cls
    
    row = np.zeros(X.shape[0])
    row.fill(filler)
    sample = random.sample(range(X.shape[0]), k)
    cls = find_clusters(X[sample], res=res, metric=metric)
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

def calculate_score_gpu(clusters, n, reps, batch_size=3000):
    """
    Score calculation on GPU
    """
    score = np.zeros((n, n), dtype=np.csingle)
    score_device = cuda.to_device(score)

    threadsPerBlock = (16, 16)
    blocksPerGrid_x = math.ceil(n / threadsPerBlock[0])
    blocksPerGrid_y = math.ceil(batch_size / threadsPerBlock[1])
    blocksPerGrid = (blocksPerGrid_x, blocksPerGrid_y)
    
    batch_num = math.ceil(n / batch_size)

    for row in clusters:
        for i in range(batch_num):
            x_batch_start = i * batch_size
            x_batch_end = min((i + 1) * batch_size, n)
            x_batch = row[x_batch_start:x_batch_end]
            x_device = cuda.to_device(x_batch)

            for j in range(batch_num):
                y_batch_start = j * batch_size
                y_batch_end = min((j + 1) * batch_size, n)
                y_batch = row[y_batch_start:y_batch_end]
                y_device = cuda.to_device(y_batch)
                outer_equality_kernel[blocksPerGrid, threadsPerBlock](x_device, y_device, score_device, x_batch_start, y_batch_start)

                del y_device
                cuda.current_context().memory_manager.deallocations.clear()

            del x_device
            cuda.current_context().memory_manager.deallocations.clear()
    
    score = score_device.copy_to_host()
    score = np.where(score.real > 0, percent_match(score, reps), 0)
    
    del score_device
    cuda.current_context().memory_manager.deallocations.clear()
    return score

@cuda.jit
def outer_equality_kernel(x, y, out, x_start, y_start):
    """
    GPU kernel score calculation algorithm
    """
    tx, ty = cuda.grid(2)

    if tx < x.shape[0] and ty < y.shape[0]:
        if x[tx] == -1 or y[ty] == -1:
            out[tx + x_start, ty + y_start] += 1j
        elif x[tx] == y[ty]:
            out[tx + x_start, ty + y_start] += 1

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
def poisson_dispersion_stats(X):
    n = np.sum(X, 1)
    pis = np.sum(X, 0) / np.sum(X)
    mu = pis.reshape(-1, 1) @ n.reshape(1, -1)
    mu = mu.T
    y2 = np.square(X - mu) / mu

    disp = np.sum(y2, 0) / y2.shape[0]

    return np.sqrt(y2.shape[0]) * (disp - 1) / np.sqrt(np.var(y2, 0))



def test_split(X, ids1, ids2, nPC, alpha_level, n_jobs=None, old_preprocessing=False, device='gpu'):
    if X.shape[0] < 2:
        return 1
    
    new_X = X[ids1 + ids2,:]
    nPC = min(nPC, X.shape[1])
    if old_preprocessing:
        metric='euclidean'
        X_pre = scshc_preprocess(new_X)
        X_transform = truncated_pca(X_pre,nPC)
    else:
        metric='cosine'
        scl = scLENS(device=torch.device('cuda') if device == 'gpu' else torch.device('cpu'))
        scl.preprocess(new_X)
        X_transform = scl.fit_transform()
        nPC = X_transform.shape[1]
    
    labels = np.array([0] * len(ids1) + [1] * len(ids2))

    if metric== 'euclidean':
        stat = ward_linkage(X_transform,labels)
    elif metric == 'cosine':
        stat = average_linkage_cosine(X_transform,labels)

    phi_stat = poisson_dispersion_stats(new_X)
    check_means = np.sum(new_X, axis=0)
    p_values = norm.sf(phi_stat)
    # p-value가 0.05 미만이고, check_means가 0이 아닌 인덱스 추출
    on_genes = np.where((p_values < 0.05) & (check_means != 0))[0]

    params = fit_model(new_X,on_genes,nPC)
    
    if old_preprocessing:
        pool = list()
        parallel = Parallel(n_jobs=n_jobs)
        pool.extend(parallel(delayed(generate_null_stats)(new_X, params, on_genes, nPC=nPC, old_preprocessing=old_preprocessing) for _ in range(10)))
        mean, std = norm.fit(np.array(pool))
        pval = 1 - norm.cdf(stat, loc=mean, scale=std)

        if pval < 0.1 * alpha_level or pval > 10 * alpha_level:
            return pval

        pool.extend(parallel(delayed(generate_null_stats)(new_X, params, on_genes, nPC=nPC, old_preprocessing=old_preprocessing) for _ in range(40)))
        mean, std = norm.fit(np.array(pool))
    
    else:
        tmp_pool = generate_null_stats(new_X, params, on_genes, nPC=nPC, old_preprocessing=old_preprocessing,sclens_flag=True)
        pool = [tmp_pool[0]]
        nPC = tmp_pool[1]
        parallel = Parallel(n_jobs=n_jobs)
        pool.extend(parallel(delayed(generate_null_stats)(new_X, params, on_genes, nPC=nPC, old_preprocessing=old_preprocessing,sclens_flag=False) for _ in range(9)))
        mean, std = norm.fit(np.array(pool))
        pval = 1 - norm.cdf(stat, loc=mean, scale=std)

        if pval < 0.1 * alpha_level or pval > 10 * alpha_level:
            return pval
        
        pool.extend(parallel(delayed(generate_null_stats)(new_X, params, on_genes, nPC=nPC, old_preprocessing=old_preprocessing,sclens_flag=False) for _ in range(40)))
        mean, std = norm.fit(np.array(pool))
        

    return 1 - norm.cdf(stat, loc=mean, scale=std)

    
    
def generate_null_stats(new_X,params,on_genes,old_preprocessing=False,nPC=30,sclens_flag=True,device="gpu"):
    null_X = generate_null(new_X,params,on_genes)
    
    if old_preprocessing:
        metric = 'euclidean'
        null_pre = scshc_preprocess(null_X)
        null_gm = truncated_pca(null_pre,nPC)
    else:
        metric = 'cosine'
        if sclens_flag:
            scl = scLENS(device=torch.device('cuda') if device == 'gpu' else torch.device('cpu'))
            scl.preprocess(null_X)
            null_gm = scl.fit_transform()
            nPC = null_gm.shape[1]
        else:
            null_pre = preprocess(null_X)
            null_gm = truncated_svd(null_pre,nPC)
    
    null_gm_d = scipy.spatial.distance.pdist(null_gm, metric)

    if metric=='euclidean':
        Z = linkage(null_gm_d, method='ward')
        hc2 = fcluster(Z, 2, criterion='maxclust') # 클러스터 분할 (2개의 클러스터로 분할)
        Qclust2 = ward_linkage(null_gm,hc2)
    elif metric=='cosine':
        Z = linkage(null_gm_d, method='weighted')
        hc2 = fcluster(Z, 2, criterion='maxclust') # 클러스터 분할 (2개의 클러스터로 분할)
        Qclust2 = average_linkage_cosine(null_gm,hc2)

    if sclens_flag:
        return Qclust2, nPC
    else:
        return Qclust2
    

def fit_model(X, on_genes, nPC):
    on_counts = X[:, on_genes] # c x g
    cov = np.cov(on_counts.T) # g x g
    means = np.mean(on_counts, 0) # g
    
    sigmas = np.log(((np.diag(cov) - means) / means**2) + 1) # g
    mus = np.log(means) - 0.5 * sigmas # g

    # mus_sum = mus.reshape(-1, 1) @ np.ones((1, mus.shape[0])) + np.ones((mus.shape[0], 1)) @ mus.reshape(1, -1) # g x g
    mus_sum = mus[:, None] + mus[None, :]
    
    # sigmas_sum = sigmas.reshape(-1, 1) @ np.ones((1, sigmas.shape[0])) + np.ones((sigmas.shape[0], 1)) @ sigmas.reshape(1, -1) # g x g
    sigmas_sum = sigmas[:, None] + sigmas[None, :]

    
    with np.errstate(divide='ignore', invalid='ignore'):
        rhos = np.log(cov / np.exp(mus_sum + 0.5 * sigmas_sum) + 1) # g x g
    rhos[np.isnan(rhos)] = -10
    rhos[np.isinf(rhos)] = -10
    np.fill_diagonal(rhos, sigmas)

    vals, vecs = eigsh(rhos,k=min([nPC,rhos.shape[1]]),which='LM')
    b_idx = vals > 0
    # num_pos = sum(b_idx)
    vals = vals[b_idx][::-1]
    vecs = vecs[:,b_idx][:,::-1]
    on_cov_sub = vecs*np.sqrt(vals)[np.newaxis,:]

    on_cov = on_cov_sub @ on_cov_sub.T
    np.fill_diagonal(on_cov, np.diag(rhos))
    on_cov_PD = posdefify(on_cov)
    
    on_cov_sqrt = scipy.linalg.cholesky(on_cov_PD).T

    return np.mean(X, 0), mus, on_cov_sqrt 


def generate_null(X, params, on_genes):
    lambdas = params[0].astype(np.float64)
    on_means = params[1].astype(np.float64)
    on_cov_sqrt = params[2].astype(np.float64)

    num_gen = min(X.shape[0], 1000)
    null = np.zeros((num_gen, X.shape[1]))

    idx = np.zeros(X.shape[1], dtype=bool)
    idx[on_genes] = True

    rng = np.random.default_rng()
    null[:, np.logical_not(idx)] = rng.poisson(lambdas[np.logical_not(idx)], (num_gen, np.sum(np.logical_not(idx))))

    num_on_genes = len(on_genes)
    rand_normals = rng.normal(size=(num_gen, num_on_genes))
    Y = rand_normals @ on_cov_sqrt.T
    Y += on_means
    Y = np.exp(Y)

    null[:, idx] = rng.poisson(Y)
    
    null = null[np.sum(null, 1) > 0][:, np.sum(null, 0) > 0]
    return null
    

def posdefify(m, method='someEVadd', symmetric=True, eigen_m=None, eps_ev=1e-7):

    if not isinstance(m, np.ndarray) or m.ndim != 2:
        raise ValueError("m은 2차원 numpy 배열이어야 합니다.")

    if eigen_m is None:
        lam, Q = np.linalg.eigh(m)
    else:
        lam, Q = eigen_m

    n = len(lam)
    Eps = eps_ev * abs(lam[-1])  # 가장 큰 고유값에 대한 상대적인 작은 수

    if lam[0] < Eps:
        if method == 'someEVadd':
            lam = np.maximum(lam, Eps)
        elif method == 'allEVadd':
            lam = lam + Eps - lam[0]
        else:
            raise ValueError("method는 'someEVadd' 또는 'allEVadd'이어야 합니다.")

        o_diag = np.diag(m)  # 원래 행렬의 대각 원소

        # 조정된 고유값으로 행렬 재구성
        m = Q @ np.diag(lam) @ Q.T
        D = np.sqrt(np.maximum(Eps, o_diag) / np.diag(m))
        m = np.outer(D, D) * m
    return m

            
def compute_ess(redduc):
    """
    Compute the ESS (sum of squared deviations from column means)
    """
    col_means = np.mean(redduc, axis=0)
    deviations = redduc - col_means
    return np.sum(np.sum(deviations ** 2, axis=1))

def ward_linkage(redduc, labels):
    """
    Compute the Ward linkage test statistic
    """
    u_arr = np.unique(labels)
    ess1 = compute_ess(redduc[labels == u_arr[0], :])
    ess2 = compute_ess(redduc[labels == u_arr[1], :])
    ess = compute_ess(redduc)
    return (ess - (ess1 + ess2)) / len(labels)

def cosine_distance(u, v):
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)

    if norm_u == 0 or norm_v == 0:
        return 1.0  # 최대 코사인 거리

    cosine_similarity = np.dot(u, v) / (norm_u * norm_v)
    cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)
    cosine_distance = 1 - cosine_similarity
    return cosine_distance

def average_linkage_cosine(data, labels):
    # 고유한 레이블 확인 (두 개의 클러스터)
    unique_labels = np.unique(labels)
    if len(unique_labels) != 2:
        raise ValueError("labels 배열은 반드시 두 개의 고유한 클러스터 레이블을 가져야 합니다.")

    cluster1_data = data[labels == unique_labels[0]]
    cluster2_data = data[labels == unique_labels[1]]

    def normalize_data(data):
        norms = np.linalg.norm(data, axis=1, keepdims=True)
        zero_norms = norms == 0
        norms[zero_norms] = 1  # 크기가 0인 경우 1로 설정하여 분모가 0이 되는 것 방지
        normalized_data = data / norms
        normalized_data[zero_norms.flatten()] = 0  # 원래 크기가 0인 벡터는 0으로 설정
        return normalized_data

    norm_cluster1 = normalize_data(cluster1_data)
    norm_cluster2 = normalize_data(cluster2_data)

    cosine_similarity_matrix = np.dot(norm_cluster1, norm_cluster2.T)
    cosine_similarity_matrix = np.clip(cosine_similarity_matrix, -1.0, 1.0)
    cosine_distance_matrix = 1 - cosine_similarity_matrix
    average_distance = np.mean(cosine_distance_matrix)

    return average_distance



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

def truncated_svd(X, nPC):
    # X = preprocess(X)
    svd = TruncatedSVD(n_components=nPC, n_iter=7)
    return svd.fit_transform(X)

def seurat_preprocessing(X):
    adata = scpy.AnnData(X=X,dtype=np.float64)
    scpy.pp.normalize_total(adata,target_sum = 10000)
    scpy.pp.log1p(adata)
    
    scpy.pp.highly_variable_genes(adata,n_top_genes=2000,flavor='seurat')
    adata.raw = adata
    adata = adata[:,adata.var.highly_variable]
    scpy.pp.scale(adata)
    return adata.X

def scshc_preprocess(X):
    n = np.sum(X, 1)
    pis = np.sum(X, 0) / np.sum(X)
    mu = pis.reshape(-1, 1) @ n.reshape(1, -1)
    mu = mu.T

    b_idx = X == 0
    d = np.empty_like(X)
    d[b_idx] = -2 * (X[b_idx] - mu[b_idx])
    d[~b_idx] = 2 * ((X[~b_idx] * np.log(X[~b_idx]/mu[~b_idx])) - (X[~b_idx] - mu[~b_idx]))
    d[d<0] = 0

    result = np.sqrt(d) * np.where(X > mu, 1, -1)
    return result

def truncated_pca(X,nPC=30):
    Y = np.transpose(X.T - np.mean(X, axis=1))
    return truncated_svd(Y,nPC=nPC)