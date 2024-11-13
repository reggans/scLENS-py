import numpy as np
import pandas as pd
import scipy
import torch
from scipy.cluster.hierarchy import linkage, to_tree
from sklearn.metrics import silhouette_samples
from resample.bootstrap import confidence_interval
from joblib import Parallel
import contextlib
import io

import matplotlib.pyplot as plt
from tqdm.auto import tqdm
# from tqdm_joblib import tqdm_joblib

from .cluster_utils import find_clusters, construct_sample_clusters, calculate_score, \
    group_silhouette, calculate_one_minus_rpac, truncated_svd, seurat_preprocessing, truncated_pca, \
    scshc_preprocess, test_split
from .scLENS import scLENS

from collections import Counter
import random

def chooseR(X,
            reps=100,
            size=0.8,
            resolutions=None,
            device='gpu',
            n_jobs=None,
            silent=False,
            batch_size = 20,
            metric='cosine'):
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
        # resolutions = [0.3, 0.5, 0.8, 1, 1.2, 1.6, 2, 4, 6, 8]
        resolutions = np.arange(0.05, 2, 0.05)
    resolutions = set(resolutions)
    stats = list()
    for res in tqdm(resolutions,
                    desc='ChooseR', 
                    total=len(resolutions), 
                    disable=silent):
        stats_row = [res]
        cls = find_clusters(X, res=res,metric=metric)
        stats_row.append(len(np.unique(cls)))
        
        clusters = construct_sample_clusters(X, 
                                            reps=reps, 
                                            size=size, 
                                            res=res, 
                                            n_jobs=n_jobs,
                                            metric=metric,
                                            batch_size=batch_size,
                                            disable=True)
        score = calculate_score(clusters, X.shape[0], reps, device=device)
        
        score = 1 - score
        np.fill_diagonal(score, 0)

        sil = silhouette_samples(score, cls, metric='precomputed')
        sil_grp = group_silhouette(sil, cls)

        stats_row.append(confidence_interval(np.median, sil_grp)[0])
        stats_row.append(np.median(sil_grp))

        stats.append(stats_row)
    
    stats = pd.DataFrame(stats, columns=['res', 'n_clusters', 'low_med', 'med']).sort_values(by=['n_clusters'], ascending=False)
    threshold = max(stats['low_med'])
    filtered_stats = stats[stats['med'] >= threshold]


    if len(filtered_stats) == 1:
        return filtered_stats['res'], stats
    return filtered_stats['res'].iloc[0], stats


def multiK(X,
           resolutions=None,
           reps=100,
           size=0.8,
           x1=0.1, x2=0.9,
           metric='cosine',
           reduce_func=None,
           nPC=None,
           device='gpu',
           n_jobs=None,
           old_preprocessing=False,
           batch_size=20,
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
        **
    device: One of ['cpu', 'gpu']
        Device to run the scoring on. 'cpu' will run scoring on CPU with n_jobs parallel jobs
    n_jobs: int or None
        joblib n_jobs; Number of CPU jobs to run in parallel
    
    Returns
    -------
    float
        The chosen best resolution
    """
    if reduce_func is not None and nPC is None:
        raise ValueError('nPC must be specified when using a custom reduce_func')
    if nPC is not None and reduce_func is None:
        raise ValueError('reduce_func must be specified when using a custom nPC')
    
    if resolutions is None:
        resolutions = np.arange(0.05, 2, 0.05)
    else:
        resolutions = np.array(resolutions)
    
    n = len(resolutions) * reps
    clusters = np.zeros((n, X.shape[0]))
    ks = np.zeros(n)
    f = io.StringIO()
    for i in tqdm(range(reps)):
        offset = i * len(resolutions)
        k = int(X.shape[0] * size)
        sample = random.sample(range(X.shape[0]), k)

        X_sample = X[sample]
        if reduce_func is not None:
            X_sample = reduce_func(X_sample, nPC)
        else:
            if nPC is not None:
                if old_preprocessing:
                    X_sample = truncated_svd(seurat_preprocessing(X_sample), nPC)
                else:
                    with contextlib.redirect_stdout(f):
                        scl.preprocess(X_sample)
                    X_sample = truncated_svd(scl.X, nPC)
            else:
                if old_preprocessing:
                    nPC = 30
                    X_sample = truncated_svd(seurat_preprocessing(X_sample), nPC)
                else:
                    scl = scLENS(device=torch.device('cuda') if device == 'gpu' else torch.device('cpu'))
                    scl.preprocess(X_sample)
                    X_sample = scl.fit_transform()
                    nPC = X_sample.shape[1]

        sample_cls = construct_sample_clusters(X_sample,
                                               reps=None, 
                                               size=size, 
                                               res=resolutions, 
                                               n_jobs=n_jobs,
                                               metric=metric,
                                               batch_size=batch_size,
                                               disable=True)

        full_cls = np.zeros((len(resolutions), X.shape[0])) - 1
        full_cls[:, sample] = sample_cls
        
        clusters[offset:offset+len(resolutions)] = full_cls
        ks[offset:offset+len(resolutions)] = [len(np.unique(cls)) - 1 for cls in full_cls] # accomodate for label of dropped data
    
    k_runs = [x[1] for x in sorted(Counter(ks).items())]
    k_unique = np.unique(ks)
    
    parallel = Parallel(n_jobs=n_jobs)
    one_minus_rpac = parallel(calculate_one_minus_rpac(clusters[ks==k], 
                                                        X.shape[0],
                                                        x1,
                                                        x2,
                                                        device=device)
                                                        for k in k_unique)

    points = np.array(list(zip(one_minus_rpac, k_runs)))

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
    plt.xlabel("1-rPAC")
    plt.ylabel("Number of clustering runs")
    plt.show()

    result = list()
    for k in opt_k:
        idx = np.nonzero(ks == k)[0] // reps
        res = Counter(list(resolutions[idx])).most_common(1)[0][0]
        result.append(res)
    
    print(f'Optimal resolutions: {result}')
    
    # if len(result) == 1:
    #     opt_res = result[0]
    # else:
    #     if metric is None:
    #         from sklearn.metrics import silhouette_score
    #         metric = silhouette_score

    #     best = 0
    #     opt_res = None
    #     for res in result:
    #         cls = find_clusters(X, res=res)
    #         if len(np.unique(cls)) == 1 and metric == silhouette_score:
    #             continue
    #         score = metric(X, cls, **kwargs)
    #         if score > best:
    #             best = score
    #             opt_res = res

    return result

def scSHC(X,
        nPC=30,
        alpha=0.05,
        device="gpu",
        old_preprocessing=False,
        n_jobs=None):
    """
    Daniel Müllner, fastcluster: Fast Hierarchical, 
    Agglomerative Clustering Routines for R and Python, 
    Journal of Statistical Software, 53 (2013), no. 9, 1-18,
    https://doi.org/10.18637/jss.v053.i09.
    """

    if old_preprocessing:
        metric='euclidean'
        X_pre = scshc_preprocess(X)
        X_trans = truncated_pca(X_pre,nPC)
    else:
        metric='cosine'
        scl = scLENS(device=torch.device('cuda') if device == 'gpu' else torch.device('cpu'))
        scl.preprocess(X)
        X_trans = scl.fit_transform()
        nPC = X_trans.shape[1]

    dist = scipy.spatial.distance.pdist(X_trans, metric)
    
    if metric=='euclidean':
        Z = linkage(dist, method='ward')
    elif metric=='cosine':
        Z = linkage(dist, method='weighted')
    
    tree, nodes = to_tree(Z, rd=True)

    dends_to_test = [tree]  # 분할할 덴드로그램 리스트
    clusters = []           # 최종 클러스터를 저장할 리스트
    node0 = None            # 트리의 루트 노드 (시각화를 위해)
    counter = 0             # 노드 번호 카운터
    parents = ["root"]      # 부모 노드 추적용 리스트
    alpha = 0.05            # 유의 수준 (예시 값)
    
    # test_queue = [dend.root]
    # clustering = np.zeros(X.shape[0]) - 1 # -1 for unassigned cluster
    # cluster_idx = 0

    while len(dends_to_test) > 0:
        current_node = dends_to_test[0]  # 현재 노드
        if current_node.is_leaf():
            clusters.append([current_node.id])
            dends_to_test.pop(0)
            continue  # 다음 노드로 이동
        
        left = current_node.left
        right = current_node.right
        
        ids1 = left.pre_order(lambda x: x.id)
        ids2 = right.pre_order(lambda x: x.id)
        leaves = current_node.pre_order(lambda x: x.id)
        alpha_level = alpha * ((len(leaves) - 1) / (X.shape[0] - 1))

        # 두 클러스터의 샘플 수 확인
        size_ids1 = len(ids1)
        size_ids2 = len(ids2)
        min_cluster_size = min(size_ids1, size_ids2)
        
        if min_cluster_size > 20:
            # 샘플 수가 충분하면 유의성 검정 수행
            test = test_split(X, ids1, ids2, nPC, alpha_level, n_jobs=n_jobs, old_preprocessing=old_preprocessing, device=device)
        else:
            # 샘플 수가 부족하면 유의성 검정을 수행하지 않고 유의하지 않다고 설정
            test = 1  # 유의하지 않음

        if test < alpha_level:
            # 유의미하면 자식 노드를 분할 리스트에 추가
            dends_to_test.append(left)
            dends_to_test.append(right)
            
            # 노드 정보 업데이트 (필요 시 시각화를 위해)
            if node0 is None:
                node0 = f"Node {counter}: p-value {round(test, 4)}"
            else:
                print(f"Node {counter}: p-value {round(test, 4)}")
            
            parents.append(f"Node {counter}")
            counter += 1
        else:
            # 유의하지 않으면 현재 노드의 잎 노드를 클러스터로 저장
            clusters.append(leaves)
            print(f"Cluster {len(clusters)}: p-value {round(test, 4)}")
        
        # 처리된 노드를 리스트에서 제거
        dends_to_test.pop(0)
    

    cluster_labels = np.zeros(X.shape[0], dtype=int)

    # 각 클러스터에 레이블 할당
    for i, cluster in enumerate(clusters):
        for idx in cluster:
            cluster_labels[idx] = i + 1  # 클러스터 번호는 1부터 시작

    print("최종 클러스터 레이블:", cluster_labels)
    return cluster_labels