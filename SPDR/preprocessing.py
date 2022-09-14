import numpy as np
import scanpy as sc
from scipy.sparse import issparse
def normalize_scanpy(adata,
                     batch_key=None,
                     n_high_var=1000,
                     hvg_by_batch=False,
                     normalize_samples=True,
                     target_sum=None,
                     log_normalize=True,
                     scale_features=True,
                     scale_features_by_batch=True,
                     do_pca=True,
                     do_tsne=False,
                     do_umap=False):
    n, p = adata.shape
    sparsemode = issparse(adata.X)
    sc.pp.filter_genes(adata, min_cells=10)
    sc.pp.filter_cells(adata, min_counts=50)
    if not batch_key in adata.obs.columns:
        batch_key = None
    if batch_key is not None and scale_features_by_batch is True:
        batch = np.array(list(adata.obs[batch_key])).astype(str)
    else:
        batch = np.ones((n,), dtype='str')

    if normalize_samples:
        sc.pp.normalize_total(adata,target_sum=target_sum)#
    if log_normalize:
        sc.pp.log1p(adata)
    if n_high_var:
        sc.pp.highly_variable_genes(adata,inplace=True,subset=True,n_top_genes=n_high_var,batch_key=batch_key if hvg_by_batch else None)
    if scale_features:
        if batch_key is not None and scale_features_by_batch :
            batch_list = np.unique(batch)
            if sparsemode:
                adata.X = adata.X.toarray()

            for batch_ in batch_list:
                indices = [x == batch_ for x in batch]
                sub_adata = adata[indices]
                #sc.pp.scale(sub_adata)
                sc.pp.scale(sub_adata, max_value=6.0)
                adata[indices] = sub_adata.X
        else:
            #sc.pp.scale(adata)
            sc.pp.scale(adata, max_value=6.0)
    if do_pca:
        sc.pp.pca(adata)
    if do_tsne:
        sc.tl.tsne(adata)
    if do_umap:
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)
    return adata