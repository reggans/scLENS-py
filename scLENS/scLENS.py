import torch
import pandas as pd 
import numpy as np
import scipy
from .PCA import PCA

from tqdm.auto import tqdm
import matplotlib.pyplot as plt

class scLENS():
    def __init__(self, 
                 threshold=0.5, 
                 sparsity='auto', 
                 n_rand_matrix=10, 
                 sparsity_step=0.002,
                 sparsity_threshold=0.9,
                 perturbed_n_scale = 2,
                 device=None, 
                 silent=False):
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
        self.silent = silent
    
    def preprocess(self, 
                   data, 
                   min_tp=0, 
                   min_genes_per_cell=200, 
                   min_cells_per_gene=15,
                   precomputed=False,
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
        if precomputed:
            if isinstance(data, pd.DataFrame):
                normal_cells = np.where((np.sum(data.values, axis=1) > self.min_tp) &
                                (np.count_nonzero(data.values, axis=1) >= self.min_genes_per_cell))[0]
                X_clean = data.iloc[normal_cells, self.normal_genes].values

                if not self.silent:
                    print(f'Removed {data.shape[0] - len(normal_cells)} cells and {data.shape[1] - len(self.normal_genes)} genes in QC')
                    print(f'Removed {data.shape[0] - len(normal_cells)} cells and {data.shape[1] - len(self.normal_genes)} genes in QC')
            else:
                normal_cells = np.where((np.sum(data, axis=1) > self.min_tp) &
                                (np.count_nonzero(data, axis=1) >= self.min_genes_per_cell))[0]
                X_clean = data[normal_cells, self.normal_genes]

                if not self.silent:
                    print(f'Removed {data.shape[0] - len(normal_cells)} cells and {data.shape[1] - len(self.normal_genes)} genes in QC')
                    print(f'Removed {data.shape[0] - len(normal_cells)} cells and {data.shape[1] - len(self.normal_genes)} genes in QC')

            X_clean = torch.tensor(X_clean, device=self.device, dtype=torch.double)
            X_clean = torch.transpose(torch.transpose(X_clean, 0, 1) / self.l1_norm, 0, 1)
            X_clean = torch.log(1 + X_clean)

            X_clean = (X_clean - self.mean) / self.std

            X_clean = torch.transpose(torch.transpose(X_clean, 0, 1) / self.l2_norm, 0, 1) * torch.mean(self.l2_norm)
            X_clean = torch.transpose(torch.transpose(X_clean, 0, 1) - torch.mean(X_clean, dim=1), 0, 1)

            return X_clean
        
        if isinstance(data, pd.DataFrame):
            if not data.index.is_unique and not self.silent:
                print("Cell names are not unique, resetting cell names")
                data.index =  range(len(data.index))

            if not data.columns.is_unique and not self.silent:
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

            if not self.silent:
                print(f'Removed {data.shape[0] - len(self.normal_cells)} cells and {data.shape[1] - len(self.normal_genes)} genes in QC')
        else:
            self.normal_genes = np.where((np.sum(data, axis=0) > min_tp) &
                                    (np.count_nonzero(data, axis=0) >= min_cells_per_gene))[0]
            self.normal_cells = np.where((np.sum(data, axis=1) > min_tp) &
                                    (np.count_nonzero(data, axis=1) >= min_genes_per_cell))[0]
            
            self._raw = pd.DataFrame(data[self.normal_cells][:, self.normal_genes])

            if not self.silent:
                print(f'Removed {data.shape[0] - len(self.normal_cells)} cells and {data.shape[1] - len(self.normal_genes)} genes in QC')
        
        X = torch.tensor(self._raw.values).to(self.device, dtype=torch.double)
        
        # L1 and log normalization
        self.l1_norm = torch.linalg.vector_norm(X, ord=1, dim=1)
        X = torch.transpose(torch.transpose(X, 0, 1) / self.l1_norm, 0, 1)
        X = torch.log(1 + X)

        # Z-score normalization
        self.mean = torch.mean(X, dim=0)
        self.std = torch.std(X, dim=0)
        X = (X - self.mean) / self.std

        # L2 normalization
        self.l2_norm = torch.linalg.vector_norm(X, ord=2, dim=1)
        X = torch.transpose(torch.transpose(X, 0, 1) / self.l2_norm, 0, 1) * torch.mean(self.l2_norm)
        X = torch.transpose(torch.transpose(X, 0, 1) - torch.mean(X, dim=1), 0, 1)

        self.X = X
        self.preprocessed = True

        if plot:
            self.plot_preprocessing()
        
        torch.cuda.empty_cache()
        return pd.DataFrame(X.cpu().numpy())
    
    def _preprocess_rand(self, X):
        """Preprocessing that does not save data statistics"""
        l1_norm = torch.linalg.vector_norm(X, ord=1, dim=1)
        X = torch.transpose(torch.transpose(X, 0, 1) / l1_norm, 0, 1)
        X = torch.log(1 + X)

        mean = torch.mean(X, dim=0)
        std = torch.std(X, dim=0)
        X = (X - mean) / std

        l2_norm = torch.linalg.vector_norm(X, ord=2, dim=1)
        X = torch.transpose(torch.transpose(X, 0, 1) / l2_norm, 0, 1) * torch.mean(l2_norm)
        X = torch.transpose(torch.transpose(X, 0, 1) - torch.mean(X, dim=1), 0, 1)

        torch.cuda.empty_cache()
        return X

    def fit(self, data=None, plot_mp=False):
        """
        Fits to the data by finding signal eigenvectors 
        and selecting robust eigenvectors from it

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

        self._signal_components = self._PCA(self.X, plot_mp=plot_mp)

        if self.sparsity == 'auto':
            self._calculate_sparsity()

        n = self._signal_components.shape[1] * self._perturbed_n_scale
        if self.preprocessed:
            raw = torch.tensor(self._raw.values).to(self.device, dtype=torch.double)

        score_all =  list()
        for _ in tqdm(range(self.n_rand_matrix), total=self.n_rand_matrix, desc='Calculating robust components', disable=self.silent):
            # Construct random matrix
            rand = scipy.sparse.rand(self._raw.shape[0], self._raw.shape[1], 
                                    density=1-self.sparsity, 
                                    format='csr')
            rand.data[:] = 1
            rand = torch.tensor(rand.toarray()).to(self.device)
        
            # Construct perturbed components
            if self.preprocessed:
                rand = self._preprocess_rand(raw + rand)
            else:
                rand = self.X + rand
            perturbed = self._PCA_rand(rand, n)

            # Find robust components
            pert_scores = self._calculate_correlation(self._signal_components, perturbed)
            score_all.append(pert_scores)

            del rand, perturbed
            torch.cuda.empty_cache()

        score_all = np.array(score_all)
        robust = np.average(score_all, axis=0) > self.threshold
        self.robust_components = self._signal_components[:, robust].cpu().numpy()
        self.robust_scores = score_all

        del raw, pert_scores
        torch.cuda.empty_cache()

    def _calculate_sparsity(self):
        """Automatic sparsity level calculation"""

        sparse = 0.999
        zero_idx = np.nonzero(self._raw.values == 0)
        n_zero = zero_idx[0].shape[0]

        # Calculate avg eigenvector correlation b/w 2 binary random matrices 
        data_sparsity = n_zero / (self._raw.shape[0] * self._raw.shape[1])
        Vr = list()
        for _ in range(2):
            rand = scipy.sparse.rand(self._raw.shape[0], self._raw.shape[1], 
                                    density=1-data_sparsity, 
                                    format='csr')
            rand.data[:] = 1
            rand = torch.tensor(rand.toarray()).to(self.device)

            V = self._PCA_rand(rand, rand.shape[0])
            Vr.append(V)

            del rand
            torch.cuda.empty_cache()
        
        thresh = torch.mean(torch.abs(torch.transpose(Vr[0], 0, 1) @ Vr[1]))

        bin = scipy.sparse.csr_array(self._raw.values)
        bin.data[:] = 1
        bin = torch.tensor(bin.toarray()).to(self.device)
        bin = self._preprocess_rand(bin)
        Vb = self._PCA_rand(bin, bin.shape[0])

        rng = np.random.default_rng()
        while sparse > self.sparsity_threshold:
            n_pert = int(sparse * n_zero)
            selection = rng.integers(n_zero, size=n_pert)
            idx = [x[selection] for x in zero_idx]

            pert = torch.zeros_like(bin, device=self.device)
            pert[idx] = 1
            pert += bin

            W = (pert @ torch.transpose(pert, 0, 1)) / pert.shape[1]
            eval, evec = torch.linalg.eigh(W)
            Vbp = evec[:, eval != 0]
            n_vbp = Vbp.shape[1]//2
            Vbp = Vbp[:, -n_vbp:]

            corr = torch.min(torch.abs(torch.transpose(Vb, 0, 1) @ Vbp))

            if corr < thresh:
                self.sparsity = sparse
                break
            else:
                sparse -= self.sparsity_step
        
            del pert, W, eval, evec, Vbp
            torch.cuda.empty_cache()
        
        del bin
        torch.cuda.empty_cache()
    
    def _calculate_correlation(self, X, Y):
        X = torch.transpose(X, 0, 1)
        dots = torch.abs(X @ Y)
        return [float(max(row)) for row in dots]
    
    def transform(self, X, preprocess=True):
        """
        Projects the input data X to the robust components

        Parameters
        ----------
        X: pandas.DataFrame
            Data to project. Should have the same genes as the training data. 
            If preprocess = False, assumed to already be preprocessed
        preprocess: bool, default = True
            Whether to preprocess X or not
        
        Returns
        -------
        numpy.ndarray
            Projected data
        """
        if preprocess:
            X_clean = self.preprocess(X, precomputed=True)
        else:
            X_clean = torch.tensor(X).to(self.device, dtype=torch.double)
        
        pcs = torch.tensor(self.robust_components).to(self.device, dtype=torch.double)
        transform = (X_clean @ X_clean.T @ pcs).cpu().numpy()

        del X_clean
        torch.cuda.empty_cache()
        return transform
    
    def fit_transform(self, data=None, plot_mp=False):
        """
        Fits to data and projects it to the robust components

        Parameters
        ----------
        data: pandas.DataFrame, np.ndarray
            Data to fit. If preprocess() has been called, will use the saved preprocessed data instead. default = None
        
        Returns
        -------
        numpy.ndarray
            Projected data
        """
        self.fit(data, plot_mp=plot_mp)
        return self.transform(self.X, preprocess=False)

    def _PCA(self, X, plot_mp=False):
        pca = PCA(device=self.device)
        pca.fit(X)

        if plot_mp:
            pca.plot_mp(comparison=False)
            plt.show()
        comp = pca.get_signal_components()[1]

        del pca
        torch.cuda.empty_cache()
        return torch.tensor(comp).to(self.device, dtype=torch.double)
    
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
        std = np.std(self.robust_scores, axis=0)
        avg = np.average(self.robust_scores, axis=0)
        plt.errorbar(np.arange(0, avg.shape[0]), avg, std, fmt='o', capsize=4)
        plt.axhline(y=self.threshold, color='r', linestyle='--')
        plt.ylabel('Robustness Score')
        plt.title('Signal Component Robustness')
        plt.show()

    def cluster(self, 
                X=None,
                res=None,
                method=None,
                n_neighbors=20, 
                min_weight=1/15, 
                n_iterations=-1,
                **kwargs):
        """"""
        from .clustering import find_clusters, chooseR, multiK, scSHC

        if X is None:
            X = self._raw
        elif isinstance(X, pd.DataFrame):
            normal_cells = np.where((np.sum(X.values, axis=1) > self.min_tp) &
                            (np.count_nonzero(X.values, axis=1) >= self.min_genes_per_cell))[0]
            X = X.iloc[normal_cells, self.normal_genes].values
        else:
            normal_cells = np.where((np.sum(X, axis=1) > self.min_tp) &
                            (np.count_nonzero(X, axis=1) >= self.min_genes_per_cell))[0]
            X = X[normal_cells, self.normal_genes]

        if method is None:
            if res is None:
                raise Exception('Provide a method or resolution')
            X_transform = self.transform(X, preprocess=False)
            cluster = find_clusters(X_transform,
                                    n_neighbors=n_neighbors, 
                                    min_weight=min_weight, 
                                    res=res, 
                                    n_iterations=n_iterations)
            
        elif method == 'chooseR':
            X_transform = self.transform(X, preprocess=False)
            cluster = chooseR(X_transform, **kwargs)
        
        elif method == 'multiK':
            sclens_kwargs = {"threshold": self.threshold,
                             "sparsity": 'auto',
                             "n_rand_matrix": self.n_rand_matrix,
                             "sparsity_step": self.sparsity_step,
                             "sparsity_threshold": self.sparsity_threshold,
                             "perturbed_n_scale": self._perturbed_n_scale,
                             "silent": True,
                             "device": self.device}
            
            X_transform = self.transform(X, preprocess=False)
            
            self.resolution = multiK(X, sclens_kwargs=sclens_kwargs, **kwargs)

            cluster = find_clusters(X_transform, 
                                n_neighbors=n_neighbors, 
                                min_weight=min_weight, 
                                res=self.resolution, 
                                n_iterations=n_iterations)
        elif method == 'scSHC':
            cluster = scSHC(X, X_transform, **kwargs)
            return cluster
        else:
            raise Exception('Method not recognized')
        return cluster