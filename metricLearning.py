"""
this file contains methods of metric learning and prepares for usage in main.py
The following methods are included:

(1) LMNN: Large Margin Nearest Neighbor Metric Learning. learns a projection L, which reduces the dimension.
Consider using original 2048-dim data and reduce n_components to 49, or using the reduced 49-dim data by LDA.
Recommended to use original 2048-dim data.
params: k; n_components=49, or try different settings if using original 2048-dim data.

(2) NCA: Neighborhood Components Analysis. Learns a projection A, which reduces the dimension.
Roughly the same with LMNN. Recommended to use original 2048-dim data.
params: n_components=49, or try different settings if using original 2048-dim data.

(3) LFDA: Local Fisher Discriminant Analysis for Supervised Dimensionality Reduction. Also reduces dimension,
and is recommended to use original 2048-dim data. Should be fast, so no worries.
params: k, but maybe using default k=7 is better; n_components, recommended to experimented from 8, 16, ..., 256, 512.

(4) ITML_Supervised: Supervised version of Information Theoretic Metric Learning.
params: Nothing needs to be specified, recommend to use reduced dim=49.

(5) SDML_Supervised: Supervised version of Sparse Distance Metric Learning.
Like ITML, supervise in the same way (sampling).
params: balance_param=1e-5, sparsity_param=1e-5 is used in lmj's work, recommend to use reduced dim=49.

(6) RCA_Supervised: Supervised version of Relevant Components Analysis. lmj uses reduced 49 dim.
params: n_components=49, or default, then none.

(7) LSML_Supervised: Supervised version of Least Squared-residual Metric Learning.
params: Nothing to be specified, but recommend to use original dim=2048.

(8) MMC: Mahalanobis Metric for Clustering. Traditional!
params: Nothing to be specified, recommend to use reduced dim=49.

(9) MLKR: Metric Learning for Kernel Regression. Quite high computational cost, seems like PCA,
but why lmj uses 49 dim?
params: n_components=49, or default, then none.
"""


from metric_learn import LMNN, NCA, LFDA, ITML_Supervised, SDML_Supervised, RCA_Supervised, LSML_Supervised, MMC, MLKR


methods = {'lmnn': LMNN,
           'nca': NCA,
           'lfda': LFDA,
           'itml': ITML_Supervised,
           'sdml': SDML_Supervised,
           'rca': RCA_Supervised,
           'lsml': LSML_Supervised,
           'mmc': MMC,
           'mlkr': MLKR}


def run(X, y, method='lmnn', **kwargs):
    try:
        obj = methods[method](**kwargs)
    except KeyError:
        raise ValueError(f'Method {method} not implemented!')
    obj.fit(X, y)
    return obj.components_
