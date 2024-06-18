from .randomly_ import randomly
import torch
import pandas as pd
import numpy as np
from scipy import stats, linalg

import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.lines as mlines
import matplotlib.gridspec as gridspec
from matplotlib.offsetbox import AnchoredText
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

import seaborn as sns

class PCA(randomly.Rm):
    def __init__(self, tol=0.0, device=None, random_state=None) -> None:
        super(PCA, self).__init__(tol=tol, random_state=random_state)

        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            assert isinstance(device, torch.device)
            self.device = device
    
    def fit(self, X=None, eigen_solver='wishart'):
        """
        Fit RM model

        Parameters
        ----------
        X: torch.tensor, shape (n_cells, n_genes)
            where n_cells is the number of cells
            and n_genes is the number of genes
        ----------
        self: object
            Returns the instance itself
        """
        if not self._preprocessing_flag:
            self.X = X
            self.n_cells = X.shape[0]
            self.n_genes = X.shape[1]
            # self.X3 = pd.DataFrame(self.X, index=self.normal_cells, columns=self.normal_genes)

        self._fit()
        return
    
    def _fit(self):
        """Fit the model for the dataframe df and apply
           the dimensionality reduction by removing the eigenvalues
           that follow Marchenko - Pastur distribution
        """
        # self.mean_ = np.mean(self.X, axis=0)
        # self.std_ = np.std(self.X, axis=0, ddof=0)
        # self.X = (self.X-self.mean_) / (self.std_+0.0)

        """Dispatch to the right submethod depending on
           the chosen solver"""
        if self.eigen_solver == 'wishart':
            Y = self._wishart_matrix(self.X)
            (self.L, self.V) = self._get_eigen(Y)
            Xr = self._random_matrix(self.X)
            Yr = self._wishart_matrix(Xr)
            (self.Lr, self.Vr) = self._get_eigen(Yr)

            self.explained_variance_ = (self.L**2) / (self.n_cells)
            self.total_variance_ = self.explained_variance_.sum()

            self.L_mp = self._mp_calculation(self.L, self.Lr)
            self.lambda_c = self._tw()
            self.peak = self._mp_parameters(self.L_mp)['peak']

            del Y, Xr, Yr
            torch.cuda.empty_cache()
        else:
            print('''Solver is undefined, please use
                     Wishart Matrix as eigenvalue solver''')

        self.Ls = self.L[self.L > self.lambda_c]
        Vs = self.V[:, self.L > self.lambda_c]

        self.Vs = Vs
        noise_boolean = ((self.L < self.lambda_c) & (self.L > self.b_minus))
        Vn = self.V[:, noise_boolean]
        self.Ln = self.L[noise_boolean]
        self.n_components = len(self.Ls)
    
    def get_signal_components(self, n_components=0):
        """
        Extract signal components after fitting

        Parameters
        ----------
        n_components: int
            The number of signal components to be returned
        ----------
        Ls: numpy.ndarray
            Eigenvalues of signal components
        Vs: numpy.ndarray
            Signal component vectors
        """
        if n_components == 0:
            return self.Ls,  self.Vs
        elif n_components >= 1:
            return self.Ls[:n_components], self.Vs[:n_components]
        raise ValueError('n_components must be positive')
    
    def _wishart_matrix(self, X):
        """Compute Wishart Matrix of the cells"""
        return (X @ torch.transpose(X, 0, 1)) / X.shape[1]
    
    def _random_matrix(self, X):
        return torch.stack([
            row[torch.randperm(row.shape[0])] for row in torch.unbind(X, dim=0)
        ], dim=0)

    def _get_eigen(self, Y):
        """Compute Eigenvalues of the real symmetric matrix"""
        (L, V) = torch.linalg.eigh(Y)
        L = L.cpu().numpy()
        V = V.cpu().numpy()
        return (L, V)
    
    def plot_mp(self, comparison=True, path=False,
                info=True, bins=None, title=None):
        """Plot Eigenvalues,  Marchenko - Pastur distribution,
        randomized data and estimated Marchenko - Pastur for
        randomized data

        Parameters
        ----------
        path: string
                Path to save the plot
        fit: boolean
            The data.
        fdr_cut: float

        Returns
        -------
        object: plot
        """

        self.style_mp_stat()
        if bins is None:
            #bins = self.n_cells
            # bins = int(self.n_cells/3.0)
            bins = 300
        x = np.linspace(0, int(round(np.max(self.L_mp) + 0.5)), 2000)
        y = self._mp_pdf(x, self.L_mp)
        yr = self._mp_pdf(x, self.Lr)

        if info:
            fig = plt.figure(dpi=100)
            fig.set_tight_layout(False)

            ax = fig.add_subplot(111)
        else:
            plt.figure(dpi=100)

        plot = sns.distplot(self.L,
                            bins=bins,
                            norm_hist=True,
                            kde=False,
                            hist_kws={"alpha": 0.85,
                                      "color": sns.xkcd_rgb["cornflower blue"]
                                      }
                            )

        plot.set(xlabel='First cell eigenvalues normalized distribution')
        plt.plot(x, y,
                 sns.xkcd_rgb["pale red"],
                 lw=2)

        MP_data = mlines.Line2D([], [], color=sns.xkcd_rgb["pale red"],
                                label='MP for random part in data', linewidth=2)
        MP_rand = mlines.Line2D([], [], color=sns.xkcd_rgb["sap green"],
                                label='MP for randomized data', linewidth=1.5,
                                linestyle='--')
        randomized = mpatches.Patch(color=sns.xkcd_rgb["apple green"],
                                    label='Randomized data', alpha=0.75,
                                    linewidth=3, fill=False)
        data_real = mpatches.Patch(color=sns.xkcd_rgb["cornflower blue"],
                                   label='Real data', alpha=0.85)

        if comparison:
            sns.distplot(self.Lr, bins=30, norm_hist=True,
                         kde=False,
                         hist_kws={"histtype": "step", "linewidth": 3,
                                   "alpha": 0.75,
                                   "color": sns.xkcd_rgb["apple green"]}
                         )

            plt.plot(x, yr,
                     sns.xkcd_rgb["sap green"],
                     lw=1.5,
                     ls='--'
                     )

            plt.legend(handles=[data_real, MP_data, randomized, MP_rand],
                       loc="upper right", frameon=True)

        else:
            plt.legend(handles=[data_real, MP_data],
                       loc="upper right", frameon=True)

        plt.xlim([0, int(round(max(np.max(self.Lr), np.max(self.L_mp))
                               + 1.5))])
        # plt.grid(b=True, linestyle='--', lw=0.3)
        plt.grid(linestyle='--', lw=0.3)

        if title:
            plt.title(title)

        if info:
            dic = self._mp_parameters(self.L_mp)
            info1 = (r'$\bf{Data Parameters}$' + '\n{0} cells\n{1} genes'
                     .format(self.n_cells, self.n_genes))
            info2 = ('\n' + r'$\bf{MP\ distribution\ in\ data}$'
                     + '\n$\gamma={:0.2f}$ \n$\sigma^2={:1.2f}$\
                      \n$b_-={:2.2f}$\n$b_+={:3.2f}$'
                     .format(dic['gamma'], dic['s'], dic['b_minus'],
                             dic['b_plus']))
            info3 = ('\n' + r'$\bf{Analysis}$' +
                     '\n{0} eigenvalues > $\lambda_c (3 \sigma)$\
                     \n{1} noise eigenvalues'
                     .format(self.n_components, self.n_cells -
                             self.n_components))

            ks = stats.kstest(self.L_mp, self._call_mp_cdf(self.L_mp, dic))

            info4 = '\n'+r'$\bf{Statistics}$'+'\nKS distance ={0}'\
                .format(round(ks[0], 4))\
                + '\nKS test p-value={0}'\
                .format(round(ks[1], 2))

            infoT = info1+info2+info4+info3
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)

            at = AnchoredText(infoT, loc=2, prop=dict(size=10),
                              frameon=True,
                              bbox_to_anchor=(1., 1.024),
                              bbox_transform=ax.transAxes)
            at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
            lgd = ax.add_artist(at)

            if path:
                plt.savefig(path, bbox_extra_artists=(
                    lgd,), bbox_inches='tight')
        else:
            if path:
                plt.savefig(path)
        # plt.show()
        return fig
