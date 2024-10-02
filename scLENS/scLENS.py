import torch
import pandas as pd 
import numpy as np
import scipy
# from joblib import Parallel, delayed, wrap_non_picklable_objects
# import random, statistics
from .PCA import PCA

from tqdm.auto import tqdm
import matplotlib.pyplot as plt

class scLENS():
    def __init__(self, 
                 threshold=0.3420201433256688, 
                 sparsity='auto', 
                 n_rand_matrix=20, 
                 sparsity_step=0.001,
                 sparsity_threshold=0.9,
                 perturbed_n_scale = 2,
                 device=None):
        """
        Parameters
        ----------
        threshold: float
            Minimum average correlation threshold for robust components. Must be between 0 and 1
        sparsity: float or 'auto'
            Sparsity level of perturbation. If 'auto' calculated automatically. Else, must be between 0 and 1
        n_rand_matrix: int
            Number of perturbation matrices to calculate robust components
        sparsity_step: float
            Used in automatic sparsity calculation. The amount to reduce sparsity per iteration
        sparsity_threshold:
            Used in automatic sparsity calculation. Threshold of correlation between perturbed and original data,
            below which sparsity level is valid
        perturbed_n_scale: int
            Used in automatic sparsity calculation. Amount of components used for perturbed data 
            in relation to the amount of signal components in the original data
        device: torch.cuda.device or None
            Device to run the calculations on
        """
        self.threshold = threshold
        self.sparsity = sparsity
        if isinstance(sparsity, str) and sparsity != 'auto':
            raise Exception("sparsity must be between 0 and 1 or 'auto'")
        self.n_rand_matrix = n_rand_matrix
        self.sparsity_step = sparsity_step
        self.sparsity_threshold = sparsity_threshold
        self.preprocessed = False
        self._perturbed_n_scale = perturbed_n_scale
        self.device = device    
        if device is None:
            self.device = torch.device("cpu")
    
    def preprocess(self, 
                   data, 
                   min_tp=0, 
                   min_genes_per_cell=200, 
                   min_cells_per_gene=15,
                   plot=False):
        """
        Preprocesses the data

        Parameters
        ----------
        data: pandas.DataFrame, np.ndarray
            Data with shape (n_cells, n_genes) to preprocess
        min_tp: int
            Minimum total number of transcripts observed in every cell and gene
        min_genes_per_cell: int
            Minimum number of different genes observed in each cell
        min_cells_per_gene: int
            Minimum number of different cells each gene is observed in

        Returns
        -------
        pandas.DataFrame
            Preprocessed data
        """
        
        if isinstance(data, pd.DataFrame):
            if not data.index.is_unique:
                print("Cell names are not unique, resetting cell names")
                data.index =  range(len(data.index))

            if not data.columns.is_unique:
                print("Removing duplicate genes")
                data = data.loc[:, ~data.columns.duplicated()]
        
            self.normal_genes = np.where((np.sum(data.values, axis=0) > min_tp) &
                                    (np.count_nonzero(data.values, axis=0) >= min_cells_per_gene))[0]
            self.normal_cells = np.where((np.sum(data.values, axis=1) > min_tp) &
                                    (np.count_nonzero(data.values, axis=1) >= min_genes_per_cell))[0]
            self.min_tp = min_tp
            self.min_cells_per_gene = min_cells_per_gene
            self.min_genes_per_cell = min_genes_per_cell
            
            self._raw = data.iloc[self.normal_cells, self.normal_genes]

            print(f'Removed {data.shape[0] - len(self.normal_cells)} cells and {data.shape[1] - len(self.normal_genes)} genes in QC')
        else:
            self.normal_genes = np.where((np.sum(data, axis=0) > min_tp) &
                                    (np.count_nonzero(data, axis=0) >= min_cells_per_gene))[0]
            self.normal_cells = np.where((np.sum(data, axis=1) > min_tp) &
                                    (np.count_nonzero(data, axis=1) >= min_genes_per_cell))[0]
            
            self._raw = pd.DataFrame(data[self.normal_cells][:, self.normal_genes])

            self.min_tp = min_tp
            self.min_cells_per_gene = min_cells_per_gene
            self.min_genes_per_cell = min_genes_per_cell
            
            print(f'Removed {data.shape[0] - len(self.normal_cells)} cells and {data.shape[1] - len(self.normal_genes)} genes in QC')
        
        X = torch.tensor(self._raw.values).to(self.device, dtype=torch.double)
        
        # L1 and log normalization
        self.l1_norm = torch.linalg.vector_norm(X, ord=1, dim=1)
        X = X / self.l1_norm.unsqueeze(1)
        X = torch.log(1 + X)

        # Z-score normalization
        self.mean = torch.mean(X, dim=0)
        self.std = torch.std(X, dim=0)
        X = (X - self.mean) / self.std

        # L2 normalization
        self.l2_norm = torch.linalg.vector_norm(X, ord=2, dim=1)
        X = X / self.l2_norm.unsqueeze(1) * torch.mean(self.l2_norm)
        X = X - torch.mean(X, dim=0)

        self.X = X
        self.preprocessed = True

        if plot:
            self.plot_preprocessing()
        
        torch.cuda.empty_cache()
        return pd.DataFrame(X.cpu().numpy())
    
    def _preprocess_rand(self, X):
        """Preprocessing that does not save data statistics"""
        l1_norm = torch.linalg.vector_norm(X, ord=1, dim=1)
        X = X / l1_norm.unsqueeze(1)
        X = torch.log(1 + X)

        mean = torch.mean(X, dim=0)
        std = torch.std(X, dim=0)
        X = (X - mean) / std

        l2_norm = torch.linalg.vector_norm(X, ord=2, dim=1)
        X = X /l2_norm.unsqueeze(1) * torch.mean(l2_norm)
        X = X - torch.mean(X, dim=0)

        torch.cuda.empty_cache()
        return X

    def fit_transform(self, data=None, plot_mp=False):
        """
        Fits to the data by finding signal eigenvectors 
        and selecting robust eigenvectors from it, and
        projects the data to the robust eigenvectors

        Parameters
        ----------
        data: pandas.DataFrame, np.ndarray 
            Data to fit if preprocess is not called before. default = None

        Returns
        -------
        None
        """
        if data is None and not self.preprocessed:
            raise Exception('No data has been provided. Provide data directly or through the preprocess function')
        if not self.preprocessed:
            self._raw = data
            if isinstance(data, pd.DataFrame):
                self.X = torch.tensor(data.values).to(self.device, dtype=torch.double)
            else:
                self.X = torch.tensor(data).to(self.device, dtype=torch.double)
        
        pca_result = self._PCA(self.X, plot_mp=plot_mp)
        self._signal_components = torch.tensor(pca_result[1]).to(self.device, dtype=torch.double)

        if self.sparsity == 'auto':
            self._calculate_sparsity()
        
        if self.preprocessed:
            raw = torch.tensor(self._raw.values).to(self.device, dtype=torch.double)

        n = min(self._signal_components.shape[1] * self._perturbed_n_scale, self.X.shape[1])

        pert_vecs = list()
        for _ in tqdm(range(self.n_rand_matrix), total=self.n_rand_matrix):
            # Construct random matrix
            rand = scipy.sparse.rand(self._raw.shape[0], self._raw.shape[1], 
                                    density=1-self.sparsity, 
                                    format='csr')
            rand.data[:] = 1
            rand = torch.tensor(rand.toarray()).to(self.device)
        
            # Construct perturbed components
            rand = self._preprocess_rand(raw + rand)
            perturbed = self._PCA_rand(rand, n)
            pert_vecs.append(perturbed)

            del rand
            torch.cuda.empty_cache()

        # Select the most correlated components for each perturbation
        pert_select = [torch.argmax(torch.abs( \
            torch.transpose(self._signal_components, 0, 1) @ x), dim=1) \
            for x in pert_vecs]
        pert_vecs = [x[:, idx] for x, idx in zip(pert_vecs, pert_select)]
        
        # Calculate correlation between perturbed components
        pert_scores = list()
        for i in range(self.n_rand_matrix):
            for j in range(i+1, self.n_rand_matrix):
                dots = torch.transpose(pert_vecs[i], 0, 1) @ pert_vecs[j]
                corr = torch.max(torch.abs(dots), dim=1).values
                pert_scores.append(corr.cpu().numpy())
        
        pert_scores = np.array(pert_scores)
        pvals = np.sum(pert_scores < self.threshold, axis=0) / pert_scores.shape[0]
        robust = pvals < 0.01

        self.X_transform = pca_result[1][:, robust] * np.sqrt(pca_result[0][robust]).reshape(1, -1)
        self.robust_scores = pert_scores

        del raw, pert_scores, pert_vecs, pert_select
        torch.cuda.empty_cache()

        return self.X_transform

    def _calculate_sparsity(self):
        """Automatic sparsity level calculation"""
        sparse = 0.999
        zero_idx = np.nonzero(self._raw.values == 0)
        n_len = self.X.shape[0]*self.X.shape[1]
        n_zero = zero_idx[0].shape[0]
        
        rng = np.random.default_rng()
        # Calculate threshold for correlation
        n_sampling = min(self.X.shape)
        thresh = np.mean([max(np.abs(rng.normal(0, np.sqrt(1/n_sampling), n_sampling)))
                            for _ in range(5000)]).item()
        print(f'sparsity_th: {thresh}')

        # Construct binarized data matrix
        bin = scipy.sparse.csr_array(self._raw.values)
        bin.data[:] = 1
        bin = torch.tensor(bin.toarray()).to(self.device)
        Vb = self._PCA_rand(self._preprocess_rand(bin), bin.shape[0])
        n_vbp = Vb.shape[1]//2
        # n_vbp = self._signal_components.shape[1]

        n_buffer = 5
        buffer = [1] * n_buffer
        while sparse > self.sparsity_threshold:
            n_pert = int((1-sparse) * n_len)
            selection = np.random.choice(n_zero,n_pert,replace=False)
            idx = [x[selection] for x in zero_idx]

            # Construct perturbed data matrix
            pert = torch.zeros_like(bin, device=self.device)
            pert[idx] = 1
            pert += bin
            pert = self._preprocess_rand(pert)
            Vbp = self._PCA_rand(pert, pert.shape[0])
            Vbp = Vbp[:, :n_vbp]

            # Calculate correlation between perturbed and original data
            corr_arr = torch.max(torch.abs(torch.transpose(Vb, 0, 1) @ Vbp), dim=0).values.cpu().numpy()
            # corr = min(corr_arr[corr_arr > 1e-3])
            corr = np.sort(corr_arr)[1]

            buffer.pop(0)
            buffer.append(corr)

            print(f'Min(corr): {corr}, sparsity: {sparse}, add_ilen: {selection.shape}')
            if all([x < thresh for x in buffer]):
                self.sparsity = sparse + self.sparsity_step * (n_buffer - 1)
                break
            
            sparse -= self.sparsity_step
            
            del pert, Vbp
            torch.cuda.empty_cache()
        
        del bin
        torch.cuda.empty_cache()

    def _PCA(self, X, plot_mp=False):
        pca = PCA(device=self.device)
        pca.fit(X)

        if plot_mp:
            pca.plot_mp(comparison=False)
            plt.show()
        comp = pca.get_signal_components()

        del pca
        torch.cuda.empty_cache()
        return comp
    
    def _PCA_rand(self, X, n):
        W = (X @ torch.transpose(X, 0, 1)) / X.shape[1]
        _, V = torch.linalg.eigh(W)

        del W
        torch.cuda.empty_cache()
        return V[:, -n:]
    
    def plot_preprocessing(self):
        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        raw = self._raw
        clean = self.X.cpu().numpy()

        axs[0].hist(np.average(raw, axis=1), bins=100)
        axs[1].hist(np.average(clean, axis=1), bins=100)
        fig.suptitle('Mean of Gene Expression along Cells')
        plt.show()

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].hist(np.std(raw, axis=0), bins=100)
        axs[1].hist(np.std(clean, axis=0), bins=100)
        fig.suptitle('SD of Gene Expression for each Gene')
        plt.show()
    
    def plot_robust_score(self):
        n_fragile = self.robust_scores.shape[1] - self.X_transform.shape[1]
        for i in range(self.robust_scores.shape[1]):
            if i < n_fragile:
                plt.scatter(i*np.ones_like(self.robust_scores[:, i]), self.robust_scores[:, i], c='r', alpha=0.1)
            else:
                plt.scatter(i*np.ones_like(self.robust_scores[:, i]), self.robust_scores[:, i], c='g', alpha=0.1)
    
        plt.axhline(y=self.threshold, color='k', linestyle='--')
        plt.ylabel('Robustness Score')
        plt.title('Signal Component Robustness')
        plt.show()

    def cluster(self, 
                X,
                res=None,
                method='chooseR',
                n_neighbors=20, 
                min_weight=1/15, 
                n_iterations=-1,
                **kwargs):
        """"""
        from .cluster_utils import find_clusters

        X_transform = self.transform(X)

        if res is not None:
            self.resolution = res
        elif method == 'chooseR':
            from .clustering import chooseR

            self.resolution = chooseR(X_transform, **kwargs)
        elif method == 'multiK':
            from .clustering import multiK

            self.resolution = multiK(X_transform, **kwargs)
        elif method == 'scSHC':
            from .clustering import scSHC
            
            if isinstance(X, pd.DataFrame):
                normal_cells = np.where((np.sum(X.values, axis=1) > self.min_tp) &
                                (np.count_nonzero(X.values, axis=1) >= self.min_genes_per_cell))[0]
                X_clean = X.iloc[normal_cells, self.normal_genes].values
            else:
                normal_cells = np.where((np.sum(X, axis=1) > self.min_tp) &
                                (np.count_nonzero(X, axis=1) >= self.min_genes_per_cell))[0]
                X_clean = X[normal_cells, self.normal_genes]

            cluster = scSHC(X_clean, X_transform, **kwargs)
            return cluster
        else:
            raise Exception('Method not recognized')
        
        cluster = find_clusters(X_transform, 
                                n_neighbors=n_neighbors, 
                                min_weight=min_weight, 
                                res=self.resolution, 
                                n_iterations=n_iterations)
        
        return cluster