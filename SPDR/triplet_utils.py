import scanpy as sc
import numpy as np
import pandas as pd
import hnswlib
from sklearn.neighbors import NearestNeighbors
import itertools
import networkx as nx
from annoy import AnnoyIndex
import os,csv
import time
from distance_utils import calculate_dist
from numba import types
from numba.typed import Dict
import numba

def nn_approx(ds1, ds2, names1, names2, knn=50, return_distance=False):
    dim = ds2.shape[1]
    num_elements = ds2.shape[0]
    tree = hnswlib.Index(space='l2', dim=dim)
    #square loss: 'l2' : d = sum((Ai - Bi) ^ 2)
    #Inner  product 'ip': d = 1.0 - sum(Ai * Bi)
    #Cosine similarity: 'cosine':d = 1.0 - sum(Ai * Bi) / sqrt(sum(Ai * Ai) * sum(Bi * Bi))
    tree.init_index(max_elements=num_elements, ef_construction=100, M=16) # refer to https://github.com/nmslib/hnswlib/blob/master/ALGO_PARAMS.md for detail
    tree.set_ef(10)
    tree.add_items(ds2)
    ind, distances = tree.knn_query(ds1, k=knn)
    if not return_distance:
        match = set()
        for a, b in zip(range(ds1.shape[0]), ind):
            for b_i in b:
                match.add((names1[a], names2[b_i]))
        return match
    else:
        match = {}
        for a, b in zip(range(ds1.shape[0]), ind):
            for b_ind, b_i in enumerate(b):
                match[(names1[a], names2[b_i])] = np.sqrt(distances[a, b_ind])  # not sure this is fast
                # match.add((names1[a], names2[b_i]))
        return match

def nn_approx_old(ds1, ds2, names1, names2, knn=20,save=False, return_distance=False):
    """ Assumes that Y is zero-indexed. """
    # Build index.
    tree = AnnoyIndex(ds2.shape[1], metric="euclidean")#metric
    if save:
        tree.on_disk_build('annoy.index')
    for i in range(ds2.shape[0]):
        tree.add_item(i, ds2[i, :])
    tree.build(60)#n_trees=50
    # Search index.
    ind = []
    for i in range(ds1.shape[0]):
        ind.append(tree.get_nns_by_vector(ds1[i, :], knn, search_k=-1)) #search_k=-1 means extract search neighbors
    ind = np.array(ind)
    # Match.

    if not return_distance:
        match = set()
        for a, b in zip(range(ds1.shape[0]), ind):
            for b_i in b:
                match.add((names1[a], names2[b_i]))
        return match
    else:
        # get distance
        match = {}
        for a, b in zip(range(ds1.shape[0]), ind):
            for b_i in b:
                #match[(names1[a], names2[b_i])] = tree.get_distance(a, b_i)
                match[(names1[a], names2[b_i])] = calculate_dist(ds1[a], ds2[b_i])
        return match



def nn(ds1, ds2, names1, names2, knn=50, metric_p=2, return_distance=False):
    # Find nearest neighbors of first dataset.
    nn_ = NearestNeighbors(n_neighbors=knn, p=metric_p)  # remove self
    nn_.fit(ds2)
    nn_distances, ind = nn_.kneighbors(ds1, return_distance=True)
    if not return_distance:
        match = set()
        for a, b in zip(range(ds1.shape[0]), ind):
            for b_i in b:
                match.add((names1[a], names2[b_i]))
        return match
    else:
        match = {}
        for a, b in zip(range(ds1.shape[0]), ind):
            for b_ind, b_i in enumerate(b):
                match[(names1[a], names2[b_i])] = nn_distances[a, b_ind]  # not sure this is fast
                # match.add((names1[a], names2[b_i]))
        return match

def mnn(ds1, ds2, names1, names2, knn=20, approx=True,return_distance=False):
    # Find nearest neighbors in first direction.

    if approx:
       #hnswlib
        match1 = nn_approx(ds1, ds2, names1, names2, knn=knn,return_distance=return_distance)  # save_on_disk = save_on_disk)
        # Find nearest neighbors in second direction.
        match2 = nn_approx(ds2, ds1, names2, names1, knn=knn,return_distance=return_distance)  # , save_on_disk = save_on_disk)
    else:
        match1 = nn(ds1, ds2, names1, names2, knn=knn, return_distance=return_distance)
        match2 = nn(ds2, ds1, names2, names1, knn=knn, return_distance=return_distance)
    # Compute mutual nearest neighbors.
    if not return_distance:
        # mutal are set
        mutual = match1 & set([(b, a) for a, b in match2])
        return mutual,None
    else:
        # mutal are set
        mutual = set([(a, b) for a, b in match1.keys()]) & set([(b, a) for a, b in match2.keys()])
        #distance list of numpy array
        distances = []
        for element_i in mutual:
            distances.append(match1[element_i])  # distance is sys so match1[element_i]=match2[element_2]
        return mutual, distances

def get_dict_mnn(adata, batch_key, dr="pca", k=50, approx=True, verbose=True, return_distance=False):
    assert type(adata) == sc.AnnData, "Please make sure `adata` is sc.AnnData"
    cell_names = adata.obs_names
    batch_list = adata.obs[batch_key] if batch_key in adata.obs.columns else np.ones(adata.shape[0], dtype=str)
    batch_unique = batch_list.unique()
    batch_unique.sort()
    # dataset_dr=[]
    cells_batch = []
    dr_name = "X_" + str(dr)
    for i in batch_unique:
        cells_batch.append(cell_names[batch_list == i])

    mnns = {}
    mnns_distance = {}
    for comb in list(itertools.combinations(range(len(cells_batch)), 2)):
        # comb=(2,3)
        i = comb[0]  # i batch
        j = comb[1]  # jth batch
        if verbose:
            i_batch = batch_unique[i]
            j_batch = batch_unique[j]
            print("Processing datasets: {} = {}".format((i, j), (i_batch, j_batch)))
        target = list(cells_batch[j])
        ref = list(cells_batch[i])
        ds1 = adata[target].obsm[dr_name]
        ds2 = adata[ref].obsm[dr_name]
        names1 = target
        names2 = ref
        match, distances = mnn(ds1, ds2, names1, names2, knn=k, approx=approx, return_distance=return_distance)
        if verbose:
            print("There are ({}) MNN pairs when processing {}={}".format(len(match),(i, j), (i_batch, j_batch)))
        if not return_distance:
            # G=nx.Graph()
            # G.add_edges_from(match)
            df = pd.DataFrame(list(match), columns=['source', 'target'])
            if nx.__version__ > "2.0":
                G = nx.from_pandas_edgelist(df=df, source='source', target='target')
            else:
                G=nx.from_pandas_dataframe(df=df, source='source', target='target')

            #node_names = np.array(G.nodes)
            adj = G.adj
            for i in adj:
                #remember that mnn are dict, the keys are cellname
                #mnns[i] = list(adj[i])#
                mnns[i] = list(adj[i]) if i not in mnns.keys() else mnns[i] + list(adj[i])  #
        else:
            df = pd.DataFrame(list(match), columns=['source', 'target'])
            df['distance'] = distances
            if nx.__version__ > "2.0":
                G = nx.from_pandas_edgelist(df=df, source='source', target='target', edge_attr=['distance'])
            else:
                G=nx.from_pandas_dataframe(df=df, source='source', target='target', edge_attr=['distance'])
            #node_names = np.array(G.nodes)
            adj = G.adj
            for ind, i in enumerate(adj):
                #mnns[i] = list(adj[i])
                #tmp0 = [adj[i][i0]['distance'] for i0 in adj[i]] # 这里使用dict还是list，想清楚一些
                #mnns_distance[i] = np.array(tmp0.copy())
                mnns[i] = list(adj[i]) if i not in mnns.keys() else mnns[i] + list(adj[i])
                tmp0 = [adj[i][i0]['distance'] for i0 in adj[i]]  # 这里使用dict还是list，想清楚一些
                mnns_distance[i] = np.array(tmp0.copy()) if i not in mnns_distance.keys() else np.array(
                    list(mnns_distance[i]) + list(tmp0))
    if not return_distance:
        return mnns, None
    else:
        return mnns, mnns_distance


def get_dict_knn(adata, cell_subset, dr="pca", k=50, save=True, approx=True, return_distance=False):
    #makesure cell_subset1 and cell_subset2 are the
    assert type(adata) == sc.AnnData, "Please make sure `adata` is sc.AnnData"
    dr_name = "X_" + str(dr)
    dr = adata.obsm[dr_name]
    dataset = adata[cell_subset]
    dr_query=dataset.obsm[dr_name]
    cell_subset_base=adata.obs_names
    if approx:
        num_elements, dim = dr.shape
        p = hnswlib.Index(space="l2", dim=dim)
        p.init_index(max_elements=num_elements, ef_construction=100, M=16)
        p.set_ef(10)
        p.add_items(dr)
        ind, distances = p.knn_query(dr_query, k=k+1)
        cell_subset_query = np.array(dataset.obs_names)
        #ind[:,1:] in order to exclude the current point x as the neighbors of x itself.
        names = list(map(lambda x: list(cell_subset_base[x]), ind[:,1:]))
        knns = dict(zip(cell_subset_query, names))
        if return_distance:
            knns_distance = dict(zip(cell_subset_query, distances[:,1:]))
            return knns, knns_distance
    else:
        nn_ = NearestNeighbors(n_neighbors=k+1, p=2,metric="minkowski")#euclidean,hamming,et al
        nn_.fit(dr)
        nn_distances, ind = nn_.kneighbors(dr_query, return_distance=True)
        cell_subset_query = np.array(dataset.obs_names)
        names = list(map(lambda x: list(cell_subset_base[x]), ind[:,1:]))
        knns = dict(zip(cell_subset_query, names))
        if return_distance:
            knns_distance = dict(zip(cell_subset_query, nn_distances[:,1:]))
            return knns, knns_distance
    return knns, None

def merge_dict(x, y, x_distance=None, y_distance=None):
    y_copy=y.copy()
    zz = time.time()
    #y_distance_copy=y_distance.copy() if y_distance is not None else {}
    for k, v in x.items():
        # k is key,
        # v, is the values of each
        if k in y.keys():
            y_copy[k] = list(set(v)|set(y[k]))
        else:
            y_copy[k] = v
    y_distance_copy=y_copy.copy()#can save time
    print("Step2:" + str(time.time() - zz))
    if x_distance is not None and y_distance is not None:
        for k, v in y_copy.items():
            distance_tmp=[]
            for j in v:
                if k in y.keys() and j in y[k]:
                    distance_tmp.append(y_distance[k][y[k].index(j)])
                else:
                    distance_tmp.append(x_distance[k][x[k].index(j)])
            y_distance_copy[k]=np.array(distance_tmp.copy())
        print("Step2:" + str(time.time() - zz))
        return y_copy,y_distance_copy
    print("Step2:" + str(time.time() - zz))
    return y_copy, None

def generate_nn_dict(adata, batch_key,dr="pca", k_mnn=20,k_mnn2=10,k_nn=20,
                     approx=True, verbose=True, return_distance=False):
    # step 1: calculate mnn pair between batches
    assert type(adata) == sc.AnnData, "Make sure adata is sc.AnnData"
    cells = adata.obs_names
    if verbose:
        print("Step1: calculating MNNs pairs between all batches.....")
    cells_for_knn = list(adata.obs_names)
    if batch_key in adata.obs.columns:
        mnn_dict, mnn_dict_distance = get_dict_mnn(adata, batch_key, dr=dr, k=k_mnn, approx=approx,
                                               verbose=verbose,
                                               return_distance=return_distance)

        if verbose:
            print("{}/{} cells are defined as MNNS".format(str(len(mnn_dict)), str(len(cells))))
        if k_mnn2>0 and k_mnn2!=k_nn:
            cells_for_knn = list(set(cells) - set(list(mnn_dict.keys())))
            # refind knn for mnn cells, based on full 5nn cells (sometimes mnn are very scare)
            cells_for_mnn_knn = list(mnn_dict.keys())
            if verbose:
                print("Recalculate knns globally for the cells which are detected for MNNs.")
            knn_dict_mnn, knn_dict_distance_mnn = get_dict_knn(adata, cells_for_mnn_knn, k=min(len(mnn_dict),k_mnn2), approx=approx,
                                                   return_distance=return_distance)
            # how to fast this step
            final_dict, final_dict_distance = merge_dict(mnn_dict, knn_dict_mnn, mnn_dict_distance, knn_dict_distance_mnn)
        else:
            final_dict,final_dict_distance=mnn_dict,mnn_dict_distance
    else:
        k_min = min(len(cells_for_knn), k_nn)
        if verbose:
            print("batch_key={} you provided is not exist in adata.obs.columns\n".format(str(batch_key)))
            print("So we calculate knns globally for all cells....")
        knn_dict, knn_dict_distance = get_dict_knn(adata, cells_for_knn, k=k_min,approx=approx,
                                                   return_distance=return_distance)

        return knn_dict,knn_dict_distance

    if len(cells_for_knn)>0:
        #
        k_min=min(len(cells_for_knn),k_nn)
        #remember adata is original data
        if verbose:
            if k_mnn==k_mnn2:
                print("Finding neighbors globally for the remaining non MNNs cells ({})!!!".format(len(cells_for_knn)))
            else:
                print("Finding neighbors globally for the all cells ({})!!!".format(len(cells_for_knn)))

        knn_dict, knn_dict_distance = get_dict_knn(adata, cells_for_knn, k=k_min,approx=approx,
                                                   return_distance=return_distance)
        if verbose:
            print("Merge mnn and knn .....!!!!!!")
        #how to fast this step
        final_dict, final_dict_distance = merge_dict(final_dict, knn_dict, final_dict_distance, knn_dict_distance)
        return final_dict,final_dict_distance
    else:
        if verbose:
            print("All cells has been matched by mnn pairs!!!!!.")
        return final_dict,final_dict_distance # mnn_dict_distance=None if return_distance=False

def get_sig(nn_dict,nn_dict_distance):
    sig={}
    for key,value in nn_dict.items():
        distance_tmp=nn_dict_distance[key].copy()
        dist_tmp = np.sort(distance_tmp)
        if len(dist_tmp)>5:
            sig[key] = np.maximum(np.mean(dist_tmp[3:8]), 1e-20)
            #sig[key] = np.maximum(np.mean(dist_tmp[start:end]), 1e-20)
        else:
            sig[key] = np.maximum(np.mean(dist_tmp[1:]), 1e-20)
    return sig

#@numba.njit
#参考https://www.cnpython.com/qa/157625
def get_p_nn(nn_dict,nn_dict_distance,sig):
    p={}
    for key, value in nn_dict.items():
        distance_tmp = nn_dict_distance[key].copy()
        sig_square=sig[key]*np.array([sig[i] for i in value])
        p[key]=np.exp(-distance_tmp**2/sig_square)
    return p

#@numba.njit()
def get_weight(nn_dict,p_sim,out_dict,p_out,out_method="replicate"):
    len_nn=np.array([len(v_nn) for k_nn,v_nn in nn_dict.items()],dtype=np.int32)
    len_out=np.array([len(v_nn) for k_nn,v_nn in out_dict.items()],dtype=np.int32)
    n_triplets=np.sum(len_nn*len_out) if out_method=="replicate" else np.sum(len_out)
    triplets=np.ones((n_triplets,3),dtype="U20")
    weights=np.empty(n_triplets)
    tmp0=np.ones(3,dtype=np.int32)
    count=0
    if out_method=="replicate":
        for k_nn,v_nn in nn_dict.items():
            #tmp0[1] = k_nn
            p_ij=p_sim[k_nn]
            p_ik=p_out[k_nn]
            for sim_id,j in enumerate(v_nn):
                #tmp0[2]=j
                p11=p_ij[sim_id]
                for out_id,m in enumerate(out_dict[k_nn]):
                    triplets[count]=[k_nn,j,m]
                    p12=p_ik[out_id]
                    weights[count]=p11/max(p12,1e-20)
                    count+=1
    else:
        for k_nn,v_nn in nn_dict.items():
            #tmp0[1] = k_nn
            p_ij=p_sim[k_nn]
            p_ik=p_out[k_nn]
            n_out=int(len(out_dict[k_nn])/len(v_nn))
            count_2 = 0
            for sim_id,j in enumerate(v_nn):
                #tmp0[2]=j
                p11=p_ij[sim_id]
                for out_id,m in enumerate(out_dict[k_nn][n_out*count_2:(n_out*count_2+n_out)]):
                    triplets[count]=[k_nn,j,m]
                    p12=p_ik[n_out*count_2+out_id]
                    weights[count]=p11/max(p12,1e-20)
                    count+=1
                count_2+=1
    return triplets,weights

#@numba.njit()
def get_out_triplets_replicate(dr,nbrs_dict,n_outliers=5,distance_index=0):
    all_cells=list(nbrs_dict)
    out_dict=nbrs_dict.copy()
    out_distance_dict=nbrs_dict.copy()
    for k,v in nbrs_dict.items():
        #set_diff_list=all_cells.copy()
        #for v_ind in v:
        #    set_diff_list.remove(v_ind)
        set_diff_list=list(set(all_cells)-set(v))
        out=np.random.default_rng().choice(set_diff_list,n_outliers,replace=False)
        out_dict[k]=list(out)
        #out_distance_dict[k]=np.array([calculate_dist(dr.loc[k].values,dr.loc[i].values,distance_index=distance_index) for i in out])
        out_distance_dict[k] = np.array([calculate_dist(dr[k], dr[i], distance_index=distance_index) for i in out])

    return out_dict,out_distance_dict

#@numba.njit()
def get_out_triplets_random(dr,nbrs_dict,n_outliers=5,distance_index=0):
    all_cells=list(nbrs_dict)
    out_dict=nbrs_dict.copy()
    out_distance_dict=nbrs_dict.copy()
    for k,v in nbrs_dict.items():
        set_diff_list_tmp=set(all_cells)-set(v)
        set_diff_list=list(set_diff_list_tmp-set([k]))
        out=[]
        for j in range(len(v)):
            out+=list(np.random.default_rng().choice(set_diff_list,n_outliers,replace=False))
        #out=np.random.choice(set_diff_list,n_outliers*len(v),replace=True)
        out_dict[k]=list(out.copy())
        tmp1=dr[k]
        #out_distance_dict[k]=np.array([calculate_dist(dr.loc[k].values,dr.loc[i].values,distance_index=distance_index) for i in out])
        out_distance_dict[k] = np.array([calculate_dist(tmp1, dr[i], distance_index=distance_index) for i in out])
    #
    return out_dict,out_distance_dict


def get_random_triplet(dr,sig,k_rand=4,distance_index=0):
    n=len(dr)
    all_cells = list(sig.keys())
    rand_triplets=np.ones((n*k_rand,3),dtype="U20")
    weights=np.empty(n*k_rand)
    count=0
    for k ,v in sig.items():
        set_diff=set(all_cells)-set([k])
        sim=np.random.default_rng().choice(list(set_diff),k_rand)
        out=np.random.default_rng().choice(list(set_diff-set(sim)),k_rand)
        for ind,(i1,i2) in enumerate(zip(sim,out)):
            #distance_tmp1=calculate_dist(dr.loc[k].values,dr.loc[i1].values,distance_index=distance_index)
            distance_tmp1=calculate_dist(dr[k], dr[i1], distance_index=distance_index)
            #distance_tmp2 = calculate_dist(dr.loc[k].values,dr.loc[i2].values, distance_index=distance_index)
            distance_tmp2 = calculate_dist(dr[k], dr[i2], distance_index=distance_index)
            p1=np.exp(-distance_tmp1**2/sig[k]/sig[i1])
            p2=np.exp(-distance_tmp2**2/sig[k]/sig[i2])
            sim,out=i1,i2
            if p1<p2:
                sim,out=i2,i1
                p1,p2=p2,p1
            if p2<1e-20:
                p2=1e-20
            rand_triplets[count]=[k,sim,out]
            weights[count]=p1/p2
            count+=1 #don't forget it.
    return rand_triplets,weights



def timer(start,end):
  hours, rem = divmod(end-start, 3600)
  minutes, seconds = divmod(rem, 60)
  return hours,minutes,seconds

def generate_triplets(adata,
                      batch_key,
                      dr="pca",
                      k_mnn=20,
                      k_mnn2=5,
                      k_nn=20,
                      k_out=5,
                      k_rand=5,
                      weight_adj=500.0,
                      out_pattern="random",# or replicate
                      approx=True,
                      verbose=True,
                      return_distance=True):
    #mnn&knn
    zz=time.time()
    final_dict,final_dict_distance=generate_nn_dict(adata, batch_key=batch_key, dr=dr, k_mnn=k_mnn,k_mnn2=k_mnn2, k_nn=k_nn, approx=approx,verbose=verbose, return_distance=return_distance)
    print("Step_get_nn_dict:"+str(time.time()-zz))
    sig=get_sig(final_dict, final_dict_distance)
    print("Step_get_sig:" + str(time.time() - zz))
    p_sim=get_p_nn(final_dict,final_dict_distance,sig)
    print("Step_get_p_nn:" + str(time.time() - zz))
    #dr=pd.DataFrame(adata.obsm["X_pca"],index=adata.obs_names)
    dr={v:adata.obsm['X_pca'][ind] for ind,v in enumerate(adata.obs_names)}
    if out_pattern=="replicate":
        out_dict,out_distance_dict=get_out_triplets_replicate(dr,final_dict,n_outliers=k_out)
    else:
        out_dict, out_distance_dict = get_out_triplets_random(dr, final_dict, n_outliers=k_out)
    print("Step_get_out_triplets_random:" + str(time.time() - zz))

    p_out=get_p_nn(out_dict,out_distance_dict,sig)
    print("Step_get_p_nn:" + str(time.time() - zz))


    triplets,weights=get_weight(final_dict,p_sim,out_dict,p_out,out_method=out_pattern)
    print("Step_get_weight:" + str(time.time() - zz))

    #random triplets
    if k_rand>0:
        triplet_random,weights_random=get_random_triplet(dr,sig,k_rand=k_rand)
        print("Step_get_random_triplet:" + str(time.time() - zz))
        triplets =np.vstack([triplets,triplet_random])
        weights =np.concatenate([weights,weights_random])
    weights[np.isnan(weights)] = 0.0
    weights /= np.max(weights)
    weights += 0.0001
    if True:
        weight_adj=weight_adj
        if not isinstance(weight_adj, (int, float)):
            weight_adj = 500.0
        weights = np.log(1 + weight_adj * weights)
        weights /= np.max(weights)

    return triplets,weights
