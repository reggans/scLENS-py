from sklearn.neighbors import kneighbors_graph
import numpy as np
import igraph as ig
import leidenalg as la
import math
from joblib import Parallel, delayed
from numba import cuda

def snn(X, n_neighbors=20, min_weight=1/15, metric='cosine'):
    graph = kneighbors_graph(X, n_neighbors=n_neighbors, metric=metric).toarray()
    neighbors = np.array([set(graph[i].nonzero()[0]) for i in range(graph.shape[0])])
    
    dist = np.asarray([[get_snn_distance(neighbors[i], neighbors[j]) 
                        for j in range(graph.shape[0])] 
                        for i in range(graph.shape[0])])

    dist[dist > (1 - min_weight)] = 1

    return dist

def get_snn_distance(n1, n2):
    sim = len(n1.intersection(n2)) / len(n1.union(n2))
    dist = 1 - sim
    return dist
    
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

# Co-clustering score functions

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

def percent_match(x, reps):
    """
    Percentage of co-clustering
    """
    return np.divide(x.real, (reps - x.imag), where=x.imag!=reps)