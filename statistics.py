import numpy as np 
import torch  
import networkx as nx 
import pickle
from sklearn.metrics.pairwise import pairwise_distances
from scipy.stats import wasserstein_distance
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os 
from torch_geometric.data import InMemoryDataset

from scipy.spatial import distance
import matplotlib.pyplot as plt
import random 
from utils import edge_index_removal
from dataset import remove_edge_index
from torch_geometric.datasets import TUDataset

import networkx as nx 
from torch_geometric.utils.convert import to_networkx
from ogb.graphproppred import PygGraphPropPredDataset
from libs.utils_new import SimulationDataset,SpectralDesign

from torch_geometric.utils import get_laplacian, to_dense_adj


def eigen_closeness(n, edges, args):
    A=np.zeros((n,n),dtype=np.float32)
    A[edges[0],edges[1]]=1
    A = np.eye(n) + A
    d = A.sum(axis=0) 
    dis=1/np.sqrt(d)
    dis[np.isinf(dis)]=0
    dis[np.isnan(dis)]=0
    D=np.diag(dis)
    nL=A.dot(D).T.dot(D)
    V,U = np.linalg.eigh(nL)
    hist_eigen, _ = np.histogram(V, bins=np.linspace(-1, 1, num=100), density=True)
    
    G = nx.from_numpy_array(nL)
    
    node_score = nx.closeness_centrality(G)
    node_score = list(node_score.values())
    # print(node_score)
    
    if args.dataset == "bbbp":
        max_v = 0.65
    elif args.dataset == "bace":
        max_v = 0.55
    elif args.dataset == "PROTEINS":
        max_v = 0.85
    hist_closeness, _ = np.histogram(node_score, bins=np.linspace(0, max_v, num=100))

    return hist_eigen, hist_closeness

def eigen_with_closeness(args):
    transform = SpectralDesign(recfield=1,dv=2,nfreq=5)
    if args.dataset == "bbbp" or args.dataset == "bace":
        pre_dataset = PygGraphPropPredDataset(name ="ogbg-mol"+ args.dataset, pre_transform=transform)
    else:
        pre_dataset = TUDataset(root='dataset',name=args.dataset, pre_transform=transform)

    sizes = []
    for sample in pre_dataset:
        sizes.append(sample.x.shape[0])
    sizes = np.array(sizes)
    sorted_index = np.argsort(sizes) 
    # max_v = -1000
    eigen_stats = []
    closeness_stats = []
    for sample in pre_dataset[sorted_index]:
        if sample.x.shape[0] >= 10:
            eigen_hist, closeness_hist = eigen_closeness(sample.x.shape[0], sample.edge_index, args)
            eigen_stats.append(eigen_hist)
            closeness_stats.append(closeness_hist)
            # max_v = max(max_v, closeness_hist)
    L = len(eigen_stats)
    
    SIMEIGEN, SIMCLOSE = np.zeros((L, L)), np.zeros((L, L))
    for i in range(L):
        for j in range(i, L):
            SIMEIGEN[i][j]=wasserstein_distance(eigen_stats[i], eigen_stats[j])
            SIMCLOSE[i][j]=wasserstein_distance(closeness_stats[i], closeness_stats[j]) 
            SIMEIGEN[j][i] = SIMEIGEN[i][j]
            SIMCLOSE[j][i] = SIMEIGEN[i][j]
    
    with open(f"{args.dataset}_SIMEIGEN.npy", 'wb') as f:
        np.save(f, SIMEIGEN)
    with open(f"{args.dataset}_SIMCLOSE.npy", 'wb') as f:
        np.save(f, SIMCLOSE)
        
            
def get_spectrum():
    pass 

EPS = 1e-16

def normalized_laplacian_matrix(G, nodelist=None, weight="weight"):
    r"""Returns the normalized Laplacian matrix of G.

    The normalized graph Laplacian is the matrix

    .. math::

        N = D^{-1/2} L D^{-1/2}

    where `L` is the graph Laplacian and `D` is the diagonal matrix of
    node degrees [1]_.

    Parameters
    ----------
    G : graph
       A NetworkX graph

    nodelist : list, optional
       The rows and columns are ordered according to the nodes in nodelist.
       If nodelist is None, then the ordering is produced by G.nodes().

    weight : string or None, optional (default='weight')
       The edge data key used to compute each value in the matrix.
       If None, then each edge has weight 1.

    Returns
    -------
    N : SciPy sparse array
      The normalized Laplacian matrix of G.

    Notes
    -----
    For MultiGraph, the edges weights are summed.
    See :func:`to_numpy_array` for other options.

    If the Graph contains selfloops, D is defined as ``diag(sum(A, 1))``, where A is
    the adjacency matrix [2]_.

    See Also
    --------
    laplacian_matrix
    normalized_laplacian_spectrum

    References
    ----------
    .. [1] Fan Chung-Graham, Spectral Graph Theory,
       CBMS Regional Conference Series in Mathematics, Number 92, 1997.
    .. [2] Steve Butler, Interlacing For Weighted Graphs Using The Normalized
       Laplacian, Electronic Journal of Linear Algebra, Volume 16, pp. 90-98,
       March 2007.
    """
    import numpy as np
    import scipy as sp
    import scipy.sparse  # call as sp.sparse

    if nodelist is None:
        nodelist = list(G)
    A = nx.to_scipy_sparse_array(G, nodelist=nodelist, weight=weight, format="csr")
    n, m = A.shape
    diags = A.sum(axis=1)
    # TODO: rm csr_array wrapper when spdiags can produce arrays
    D = sp.sparse.csr_array(sp.sparse.spdiags(diags, 0, m, n, format="csr"))
    L = D - A
    with sp.errstate(divide="ignore"):
        diags_sqrt = 1.0 / np.sqrt(diags)
    diags_sqrt[np.isinf(diags_sqrt)] = 0
    # TODO: rm csr_array wrapper when spdiags can produce arrays
    DH = sp.sparse.csr_array(sp.sparse.spdiags(diags_sqrt, 0, m, n, format="csr"))
    return DH @ (L @ DH)


def normalized_laplacian_spectrum(G, weight="weight"):
    """Return eigenvalues of the normalized Laplacian of G

    Parameters
    ----------
    G : graph
       A NetworkX graph

    weight : string or None, optional (default='weight')
       The edge data key used to compute each value in the matrix.
       If None, then each edge has weight 1.

    Returns
    -------
    evals : NumPy array
      Eigenvalues

    Notes
    -----
    For MultiGraph/MultiDiGraph, the edges weights are summed.
    See to_numpy_array for other options.

    See Also
    --------
    normalized_laplacian_matrix
    """
    import scipy as sp
    import scipy.linalg  # call as sp.linalg

    return sp.linalg.eigvalsh(
        normalized_laplacian_matrix(G, weight=weight).todense()
    )


def get_scalar_plot(pre_dataset, sorted_index, args):
    stats = []
    lens = 0
    for idx, sample in enumerate(pre_dataset[sorted_index]):
        curr_size = sample.x.shape[0]
        if curr_size >= 10:
            lens += 1 
            # print(sample)
            G = to_networkx(sample, to_undirected=True)  
            degree = G.degree() # [(node, degree), ...]
            degree = [j for (i, j) in degree]
            max_degree = np.max(degree)
            constrant = (1 / (2 *  curr_size)) **2 / (2 * max_degree)
            
            
            s = nx.normalized_laplacian_spectrum(G, weight=None)
            
            s = list(s)
            s = [0 if i <= constrant else i for i in s]
            smallest_nonzero = 1000000
            for idx, num in enumerate(s):
                if num > 0 and num < smallest_nonzero and idx >= 1:
                    smallest_nonzero = num 
            stats.append(smallest_nonzero)
    plt.plot(list(range(lens)), stats)
    plt.savefig(f"scalarplot_{args.dataset}_after100.png")


def get_cheeser_difference_plot(pre_dataset, sorted_index, args):
    
    stats = []
    lens = 0
    for idx, sample in enumerate(pre_dataset[sorted_index]):
        curr_size = sample.x.shape[0]
        # print(curr_size)
        if curr_size >= 10:
            lens += 1 
            # print(sample)
            G = to_networkx(sample, to_undirected=True)  
            degree = G.degree() # [(node, degree), ...]
            degree = [j for (i, j) in degree]
            max_degree = np.max(degree)
            constrant = (1 / (2 *  curr_size)) **2 / (2 * max_degree)
            
            
            s = nx.normalized_laplacian_spectrum(G, weight=None)
            
            s = list(s)
            s = [0 if i <= constrant else i for i in s]
            smallest_nonzero = 1000000
            for idx, num in enumerate(s):
                if num > 0 and num < smallest_nonzero and idx >= 1:
                    smallest_nonzero = num 
            stats.append(smallest_nonzero)
    L = len(stats)
    
    SIM = np.zeros((L, L))
    for i in range(L):
        for j in range(i, L):
            SIM[i][j] = np.log(np.absolute(stats[i] - stats[j]) + EPS)
            # print(SIM[i][j])
            SIM[j][i] = SIM[i][j]
    
    fig, ax = plt.subplots()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im = ax.imshow(SIM, cmap='gist_earth')
    fig.colorbar(im, cax=cax, orientation='vertical')
    plt.savefig(f"diagonal_Cheeser_difference_{args.dataset}.png")
    

def check_sparsity(loader, name="small_train"):
    # batch size = 1 
    sparsity_recorder = []
    for sample in loader:
        num_edge = sample.num_edges 
        num_possible_edge = sample.num_nodes * sample.num_nodes
        curr_sparsity  = num_edge / num_possible_edge
        sparsity_recorder.append(curr_sparsity)
    
    avg_sparsity = np.average(sparsity_recorder)
    std_sparsity = np.std(sparsity_recorder) 
    
    print(f"{name} average sparsity:{round(avg_sparsity * 100, 2)}, std_sparsity: {round(std_sparsity * 100, 2)}")
        

from torch_geometric.utils import dropout_edge, dropout_node

class Removed_edgeDataset(InMemoryDataset):
    def __init__(self, args, pre_dataset, remove_idx, root_exp='dataset_remove_edge_plot', transform= None, pre_transform= None, 
                 pre_filter= None):
        
        temp_name = f"{args.dataset}_{args.egonet_measure}_{args.dataset}_rad{args.ego_rad}_topk{args.ego_top_k}"
        if args.ego_sub_percent is not None:
            temp_name = temp_name + f"_subPercent{args.ego_sub_percent}"
        if args.ego_sub_num is not None:
            temp_name = temp_name + f"_subNum{args.ego_sub_num}"
        if args.use_edge_betweeness:
            temp_name = temp_name + "_edgeBetweeness"
        root = os.path.join(root_exp, temp_name)
        self.pre_dataset = pre_dataset
        self.args = args 
        self.remove_idx = remove_idx

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['graph_data.pt']

    # def __len__(self) -> int:
    #     return len(self.label)

    def process(self):
        data_list = []
        print("dropout edge")
        for i in range(len(self.pre_dataset)):
            sample = self.pre_dataset[i]
            # if i in self.remove_idx:
            # sample.edge_index = remove_edge_index(sample.x.shape[0], sample.edge_index.cpu(),measure=self.args.egonet_measure, args=self.args)
            sample.edge_index, edge_id = dropout_edge(edge_index=sample.edge_index, p=0.2, force_undirected=True) 
                # sample.edge_index, _, _ = dropout_node(edge_index=sample.edge_index, p=0.5, num_nodes=sample.num_nodes)
            data_list.append(sample)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def process_predataset(args, pre_dataset):
    # sizes = []
    # for sample in pre_dataset:
    #     sizes.append(sample.x.shape[0])
    # sizes = np.array(sizes)
    # sorted_index = np.argsort(sizes) 
    
    small_index = np.load(f"/home/ubuntu/Generalization/split_index/{args.dataset}/small_test_idx.npy")
    
    # target_length = int(len(sorted_index) * 0.5)
    # target_length = len(sorted_index)
    # small_graph_index = sorted_index[:target_length]
    
    return Removed_edgeDataset(args=args, pre_dataset=pre_dataset, remove_idx=small_index)

    
    
    
        

def get_deg_dist(n,edges):
    A=np.zeros((n,n),dtype=np.float32)
    A[edges[0],edges[1]]=1
    d = A.sum(axis=0)
    # if args.dataset == "COLLAB":
    #     # print("Bin 8")
    #     # 1 - 491
    #     hist, _ = np.histogram(d, bins=np.linspace(0, 500, 100), density=True)
    # else:
    hist, _ = np.histogram(d, bins=np.arange(9), density=False)
    hist = hist / np.sum(hist)
    
    return hist


def eigen(n,edges):
    A=np.zeros((n,n),dtype=np.float32)
    A[edges[0],edges[1]]=1
    A = np.eye(n) + A
    d = A.sum(axis=0) 
    dis=1/np.sqrt(d)
    dis[np.isinf(dis)]=0
    dis[np.isnan(dis)]=0
    D=np.diag(dis)
    nL=A.dot(D).T.dot(D)
    V,U = np.linalg.eigh(nL)
    hist, _ = np.histogram(V, bins=np.linspace(-1, 1, num=100), density=False)
    hist = hist / np.sum(hist)

    return hist


def normalized_Laplacian_eigen(n, edges):
    lapla = get_laplacian(edge_index=edges, edge_weight=None, num_nodes=n, normalization="sym")
    edge_index, edge_weight = lapla
    dense_adj = to_dense_adj(edge_index=edge_index, edge_attr=edge_weight, max_num_nodes=n, batch=None)[0] 
    V,U = np.linalg.eigh(dense_adj.numpy())
    hist, _ = np.histogram(V, bins=np.linspace(-1, 1, num=100), density=True) 
    return hist 


def eigen_vec(n,edges, direc=1):
    A=np.zeros((n,n),dtype=np.float32)
    A[edges[0],edges[1]]=1
    A = np.eye(n) + A
    d = A.sum(axis=0) 
    dis=1/np.sqrt(d)
    dis[np.isinf(dis)]=0
    dis[np.isnan(dis)]=0
    D=np.diag(dis)
    nL=A.dot(D).T.dot(D)
    V,U = np.linalg.eigh(nL)
    # print(U[-1])
    # print(U[0])
    if direc:
        samp_vec = U[-1]
        if samp_vec[0]<0:
            samp_vec = -samp_vec
        # print(samp_vec)
        samp_vec = samp_vec*np.sqrt(n)
        # print(max(samp_vec))
        hist, _ = np.histogram(samp_vec
        , bins=np.linspace(-10, 10, num=100), density=True)
    else:
        samp_vec = U[0]
        if samp_vec[0]<0:
            samp_vec = -samp_vec
        samp_vec = samp_vec*np.sqrt(n)
        hist, _ = np.histogram(samp_vec, bins=np.linspace(-10, 10, num=100), density=True)
    return hist


# def eigen_versus

def average_connectivity(n, edges, measure="closeness_centrality"):
    A=np.zeros((n,n),dtype=np.float32)
    A[edges[0],edges[1]]=1
    A = np.eye(n) + A
    d = A.sum(axis=0) 
    dis=1/np.sqrt(d)
    dis[np.isinf(dis)]=0
    dis[np.isnan(dis)]=0
    D=np.diag(dis)
    nL=A.dot(D).T.dot(D)

    G = nx.from_numpy_matrix(nL)

    if measure == "closeness_centrality":
        closeness_centra = nx.closeness_centrality(G)
        res = np.average(list(closeness_centra.values()))
        # record.extend(list(closeness_centra.values()))
        # hist, _ = np.histogram(list(closeness_centra.values()), bins=np.linspace(0, 0.70, num=100), density=True)
        # print(max(record)) # 0.642857 
    elif measure == "current_flow_closeness_centrality":
        # This contains bugs
        output = nx.current_flow_betweenness_centrality(G)
        # print("current flow!")
        print(output)
        res = np.average(list(output.values()))
    elif measure == "clustering":
        output = nx.clustering(G)
        # print(output)
        res = np.average(list(output.values()))
    elif measure == "triangles":
        output = nx.triangles(G)
        # print(output)
        res = np.average(list(output.values()))
    elif measure == "average_node_connectivity":
        # workable
        output = nx.average_node_connectivity(G)
        # print(output)
        res = output 
    elif measure == "average_degree_connectivity":
        output = nx.average_degree_connectivity(G)
        # print(output)
        res = np.average(list(output.values()))
    elif measure == "degree_centrality":
        output = nx.degree_centrality(G)
        # print(output)
        res = np.average(list(output.values()))
    elif measure == "eigenvector_centrality":
        output = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-4)
        # print(output)
        res = np.average(list(output.values()))
    elif measure == "betweenness_centrality":
        output = nx.betweenness_centrality(G)
        # print(output)
        res = np.average(list(output.values()))
    elif measure == "edge_betweenness_centrality":
        output = nx.edge_betweenness_centrality(G)
        # print(output)
        res = np.average(list(output.values()))
    elif measure == "communicability":
        output = nx.communicability(G) 
        avg_communica = {ky: np.average(list(vals.values())) for (ky, vals) in output.items()}
        # print(avg_communica)
        # avg of avg
        res = np.average(list(avg_communica.values()))
        print("Communicability: ", res)
    else:
        raise NotImplementedError("This measure not Implemented!")
    # print(closeness_centra)
    return res 

recorder = []


def local_closeness_cheeser(sample, args):
    # A=np.zeros((n,n),dtype=np.float32)
    # A[edges[0],edges[1]]=1
    # A = np.eye(n) + A
    # d = A.sum(axis=0) 
    # dis=1/np.sqrt(d)
    # dis[np.isinf(dis)]=0
    # dis[np.isnan(dis)]=0
    # D=np.diag(dis)
    # nL=A.dot(D).T.dot(D)

    # # G = nx.from_numpy_matrix(nL)
    # G = nx
    
    G = to_networkx(sample, to_undirected=True)

    nodes = G.nodes()
    node_spectral_gap = []
    for node in nodes:
        subgraph = nx.ego_graph(G, node, undirected=True, radius=args.numHops)
        curr_spectral_gap = nx.normalized_laplacian_spectrum(subgraph)
        # print(curr_spectral_gap)
        if len(curr_spectral_gap) <= 1:
            score = 0
        else:
            score = curr_spectral_gap[1]
        node_spectral_gap.append(score) # Second smallest eigenvalue 
    hist, _ = np.histogram(node_spectral_gap, bins=np.linspace(0, 1, num=100), density=True)
    return hist
    

def local_closeness(n, edges, args):
    A=np.zeros((n,n),dtype=np.float32)
    A[edges[0],edges[1]]=1
    A = np.eye(n) + A
    d = A.sum(axis=0) 
    dis=1/np.sqrt(d)
    dis[np.isinf(dis)]=0
    dis[np.isnan(dis)]=0
    D=np.diag(dis)
    nL=A.dot(D).T.dot(D)

    G = nx.from_numpy_array(nL)

    nodes = G.nodes()
    node_score = []
    for node in nodes:
        subgraph = nx.ego_graph(G, node, undirected=True, radius=args.numHops)
        closeness_score = nx.closeness_centrality(subgraph)[node]
        node_score.append(closeness_score)
    
    hist, _ = np.histogram(node_score, bins=np.linspace(0, 1, num=1000), density=True)
    return hist 

def distribution_connectivity(n, edges, measure="closeness_centrality"):
    A=np.zeros((n,n),dtype=np.float32)
    A[edges[0],edges[1]]=1
    A = np.eye(n) + A
    d = A.sum(axis=0) 
    dis=1/np.sqrt(d)
    dis[np.isinf(dis)]=0
    dis[np.isnan(dis)]=0
    D=np.diag(dis)
    nL=A.dot(D).T.dot(D)

    G = nx.from_numpy_matrix(nL)

    if measure == "closeness_centrality":
        closeness_centra = nx.closeness_centrality(G)
        samp_vec = list(closeness_centra.values())
        # recorder.extend(samp_vec)
        # print("max: ", max(recorder))
        # print("min: ", min(recorder))
        # 0.648 - 0.0
        hist, _ = np.histogram(samp_vec, bins=np.linspace(0, 0.75, num=1000), density=True)
    elif measure == "betweenness_centrality":
        output = nx.betweenness_centrality(G)
        samp_vec = list(output.values())
        # recorder.extend(samp_vec)
        # print("max: ", max(recorder))
        # print("min: ", min(recorder))
        hist, _ = np.histogram(samp_vec, bins=np.linspace(0, 1, num=1000), density=True)
    elif measure == "eigenvector_centrality":
        output = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-4)
        samp_vec = list(output.values())
        # recorder.extend(samp_vec)
        # print("max: ", max(recorder))
        # print("min: ", min(recorder))
        hist, _ = np.histogram(samp_vec, bins=np.linspace(0, 1, num=1000), density=True)
    elif measure == "communicability":
        output = nx.communicability(G) 
        # print(output)
        # print("\n"*2)
        avg_communica = {ky: np.average(list(vals.values())) for (ky, vals) in output.items()}
        samp_vec = list(avg_communica.values())
        # recorder.extend(samp_vec)
        # print("max: ", max(recorder))
        # print("min: ", min(recorder))
        #  0.1 5.95 bbbp  0.2 - 3.52 bace  0.2 - 5.15 toxcast 0.18 - 5.7
        hist, _ = np.histogram(samp_vec, bins=np.linspace(0, 6.50, num=1000), density=True)
    elif measure == "communicability_betweeness":
        output = nx.communicability_betweenness_centrality(G) 
        print(output)
        samp_vec = list(output.values())
        # recorder.extend(samp_vec)
        # print("max: ", max(recorder))
        # print("min: ", min(recorder))
        #  0.1 5.95 bbbp  0.2 - 3.52 bace  0.2 - 5.15 toxcast 0.18 - 5.7
        hist, _ = np.histogram(samp_vec, bins=np.linspace(0, 0.90, num=1000), density=True)
    elif measure == "load_centrality":
        output = nx.load_centrality(G) 
        samp_vec = list(output.values())
        # recorder.extend(samp_vec)
        # print("max: ", max(recorder))
        # print("min: ", min(recorder))
        #  0.1 5.95 bbbp  0.2 - 3.52 bace  0.2 - 5.15 toxcast 0.18 - 5.7
        hist, _ = np.histogram(samp_vec, bins=np.linspace(0, 1, num=1000), density=True)
    elif measure == "edge_betweenness_centrality":
        output = nx.edge_betweenness_centrality(G)
        samp_vec = list(output.values())
        # recorder.extend(samp_vec)
        # print("max: ", max(recorder))
        # print("min: ", min(recorder))
        hist, _ = np.histogram(samp_vec, bins=np.linspace(0, 1, num=1000), density=True)
    # elif measure == "average_degree_connectivity":
    #     output = nx.average_degree_connectivity(G)
    #     samp_vec = list(output.values())
    #     print("max: ", max(recorder))
    #     print("min: ", min(recorder))
    #     hist, _ = np.histogram(samp_vec, bins=np.linspace(0, 6.50, num=100), density=True)
    # elif measure == "average_node_connectivity":
    #     # workable
    #     output = nx.average_node_connectivity(G)
    #     print(output)
    #     samp_vec = list(output.values())
    #     print("max: ", max(recorder))
    #     print("min: ", min(recorder))
    #     hist, _ = np.histogram(samp_vec, bins=np.linspace(0, 6.50, num=100), density=True)
    # elif measure == "clustering":
    #     output = nx.clustering(G)
    #     print(output)
    #     samp_vec = list(output.values())
    #     print("max: ", max(recorder))
    #     print("min: ", min(recorder))
    #     hist, _ = np.histogram(samp_vec, bins=np.linspace(0, 6.50, num=100), density=True)
    else:
        raise NotImplementedError("This Connecitivity Measure not Implemented!")

    return hist



def get_sim_matrix(func, name, pre_dataset, sorted_index, args=None, sample_size=None, 
                   cmp_statistics=True, is_distribution=False, 
                   connect_measure="closeness_centrality",
                   save_name="distribution_09", egonet_measure="load_centrality"):
    # print(name)
    # exit()
    if cmp_statistics:
        stats = []
        mins = 1000,
        maxs = -1000
        curr_min, curr_max = [], []
        kept_sizes = []
        if sample_size is None:
            length_dataset = len(pre_dataset[sorted_index]) 
            for idx, sample in enumerate(pre_dataset[sorted_index]):
            # for idx in range(length_dataset):
                # sample = pre_dataset[idx]
                # if sample.x.shape[0] >=10:
                if sample.num_nodes >= 10:
                    # kept_sizes.append(sample.x.shape[0])
                    # kept_sizes.append(sample.x.sum())
                    # print(sample)
                    # exit()
                    kept_sizes.append(sample.x.sum())
                    if name[-6:] == "connec":
                        temp_output = func(sample.num_nodes, sample.edge_index, measure=connect_measure)
                    elif name[:13] == "remove_egonet":
                        temp_output = func(sample.num_nodes, sample.edge_index, measure=egonet_measure,
                                            args=args)
                        # large = True if idx > length_dataset / 2 else False  
                        # if large:
                            # temp_output = func(sample.x.shape[0], sample.edge_index, measure=egonet_measure,
                            #                args=args)
                        # else:
                        #     temp_output = eigen(sample.x.shape[0], sample.edge_index)
                    elif name[:10] == "edge_score":
                        temp_output = func(sample.num_nodes, sample.edge_index, measure=args.remove_edge_score_measure,
                                           args=args)
                    elif name[:23] == "local_closeness_cheeser":
                        temp_output = func(sample,
                                            args=args)
                        # curr_min.append(temp_output2)
                        # curr_max.append(temp_output1)
                        # print(temp_output1)
                        # exit()
                    elif name[:15] == "local_closeness":
                        temp_output = func(sample.num_nodes, sample.edge_index,
                                            args=args)
                    else:
                        temp_output = func(sample.num_nodes, sample.edge_index)
                    # print(temp_output)
                    stats.append(temp_output)
            kept_sizes = np.array(kept_sizes)
            # with open(f"./SIM_dir/kept_size_{args.dataset}.npy", 'wb') as f:
            with open(f"./SIM_dir/kept_size_{name}.npy", 'wb') as f:
                np.save(f, kept_sizes)
            # print("saved!")
            # exit()
            L=len(stats)
            # print("max: ", np.max(curr_max))
            # print("min: ", np.min(curr_min))
            # exit()
        else:
            samp_portion = np.random.permutation(sorted_index.shape[0])[:sample_size]
            samp_portion_sorted = np.sort(samp_portion)
            for sample in pre_dataset[sorted_index[samp_portion_sorted]]:
                if sample.x.shape[0] >=10:
                    # size_list.append(sample.x.shape[0])
                    stats.append(func(sample.x.shape[0], sample.edge_index))
            
            L=len(stats)
        save_var = (L,stats)
        if not os.path.exists("./pickle_save"):
            os.makedirs("./pickle_save")
        with open(f"./pickle_save/{name}.pickle", 'wb') as handle:
            pickle.dump(save_var, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open(f"./pickle_save/{name}.pickle", 'rb') as handle:
            save_var = pickle.load(handle)
        L,stats = save_var
    if is_distribution:
        SIM = np.zeros((L, L))
        for i in range(L):
            for j in range(i, L):
                SIM[i][j]=wasserstein_distance(stats[i], stats[j])
                SIM[j][i]= SIM[i][j]
    # SIM = pairwise_distances(stats_sorted, metric=distance.jensenshannon, n_jobs=-1)
        # SIM = pairwise_distances(stats_sorted, metric=wasserstein_distance)
    else:
        stats = np.array(stats)
        # if stats.ndim == 1:
        #     stats = stats[:, np.newaxis]
        SIM = pairwise_distances(stats, metric=distance.euclidean, n_jobs=-1)
    # print SIM 
    sim_dir = "./SIM_dir"
    if not os.path.exists(sim_dir):
        os.makedirs(sim_dir)
    with open(f"{sim_dir}/{name}_SIM.npy", 'wb') as f:
        np.save(f, SIM)
    # print(name)
    # large_index = np.load(f"/home/ubuntu/Generalization/split_index/{args.dataset}/large_test_idx.npy")
    # small_index = np.load(f"/home/ubuntu/Generalization/split_index/{args.dataset}/small_test_idx.npy")
    # large_small_distance = SIM[large_index, small_index]
    # distance = np.average(large_small_distance)
    # print("current average distance: ", distance)
    # print("current std distance: ", np.std(large_small_distance))
    # print("current median distance: ", np.median(large_small_distance))
    fig, ax = plt.subplots()
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    im = ax.imshow(SIM, cmap='gist_earth')
    fig.colorbar(im, cax=cax, orientation='vertical')

    if not os.path.exists(f"./{save_name}_png"):
        os.makedirs(f"./{save_name}_png")
    if not os.path.exists(f"./{save_name}_pdf"):
        os.makedirs(f"./{save_name}_pdf")
    plt.savefig(f"./{save_name}_png/"+name+".png")
    plt.savefig(f"./{save_name}_pdf/"+name+".pdf")
    # print(f"./{save_name}_png/"+name+".png")

    with open(f"{name}_sim.pickle", 'wb') as handle:
        pickle.dump(SIM, handle, protocol=pickle.HIGHEST_PROTOCOL)





def remove_egonet_based_on_measure(n, edges, measure="load_centrality", args=None):
    # print(edges)
    A=np.zeros((n,n),dtype=np.float32)
    A[edges[0],edges[1]]=1
    A = np.eye(n) + A
    d = A.sum(axis=0) 
    dis=1/np.sqrt(d)
    dis[np.isinf(dis)]=0
    dis[np.isnan(dis)]=0
    D=np.diag(dis)
    nL=A.dot(D).T.dot(D)

    G = nx.from_numpy_matrix(nL)

    if measure == "load_centrality":
        output = nx.load_centrality(G) 
        samp_vec = list(output.values()) 
        ego_center = np.argmax(samp_vec)
        ego_center_value = samp_vec[ego_center]
        # print(ego_center)'
    elif measure == "betweenness_centrality":
        output = nx.betweenness_centrality(G)
        samp_vec = list(output.values())
        ego_center = np.argmax(samp_vec)
        ego_center_value = samp_vec[ego_center]
    elif measure == "closeness_centrality":
        output = nx.closeness_centrality(G)
        samp_vec = np.array(list(output.values()))
        # print(samp_vec)
        # print(np.argsort(-np.array(samp_vec)))
        # print(np.argmax(samp_vec))
        ego_centers = np.argsort(-samp_vec)[:args.ego_top_k]
        ego_center_values_before = samp_vec[ego_centers]
        # print(ego_centers)
        # print(ego_center_values)
        # exit()
        # ego_center = np.argmax(samp_vec) 
        # ego_center_value = samp_vec[ego_center]
    elif measure == "eigenvector_centrality":
        output = nx.eigenvector_centrality(G, max_iter=1000, tol=1e-4)
        samp_vec = list(output.values())
        ego_center = np.argmax(samp_vec) 
        ego_center_value = samp_vec[ego_center]
    else:
        raise NotImplementedError("This Measure not Implemented!")
    
    # remove all the egonet edges

    ego_nets = []

    for ego_center in ego_centers:
        ego_nets.append(nx.ego_graph(G, ego_center, undirected=True, radius=args.ego_rad))
    
    # ego_net = nx.ego_graph(G, ego_center, undirected=True, radius=args.ego_rad)
    # Remove all edges in the ego network of node x
    # G.remove_edges_from(list(ego_net.edges(ego_center)))


    # remove_edges = list(ego_net.edges(ego_center))
    remove_edges = []
    for ego_net in ego_nets:
        remove_edges.extend(list(ego_net.edges()))
    # exit()
    # remove_edges = list(ego_net.edges())
    remove_edges = [(i, j) for (i, j) in remove_edges if i != j]
    # print(remove_edges)

    # sampling do here 
    if args.ego_sub_percent is not None:
        remove_edges = np.array(remove_edges)
        shuffle = np.random.permutation(len(remove_edges))
        kept_part = int(len(remove_edges) * args.ego_sub_percent)
        remove_edges = remove_edges[shuffle]
        remove_edges =  list(remove_edges[:kept_part])
        remove_edges = [(i, j) for i,j in remove_edges]
        # print(remove_edges)
    # exit()

    remove_edges = [*set(remove_edges)]
    # print(remove_edges)
    # Get Numers do here 
    if args.ego_sub_num is not None and len(remove_edges) > args.ego_sub_num:
        remove_edges = random.sample(remove_edges, k=args.ego_sub_num)
    # print(remove_edges)
    # exit()
    # print(remove_edges)
    # exit()

    # print(f"egocenter: {ego_center}, value: {ego_center_value}")

    # print("to-be-removed edges: ", remove_edges)
    
    delete_edge = torch.zeros(2, 2*len(remove_edges), dtype=torch.long)
    idx = 0 
    for (i, j) in remove_edges: 
        small = min(i, j) 
        big = max(i, j)
        delete_edge[0, idx] =  small 
        delete_edge[1, idx] = big 
        delete_edge[0, idx+1] = big 
        delete_edge[1, idx+1] = small 
        idx += 2 

    # print("delete edges: \n", delete_edge)
    # exit()
    

    def contains_column(A, c):
        # tensor A, column c 
        ans = -1 
        assert A.ndim == 2 and A.shape[0] == 2 
        for j in range(A.shape[1]):
            if torch.allclose(A[:, j:j+1], c):
                ans = j 
        return ans  
                

    # Masked Out the overlapped columns
    mask = torch.ones(edges.shape[1], dtype=torch.bool)
    for c_del_index in range(delete_edge.shape[1]):
        c_del = delete_edge[:, c_del_index].unsqueeze(-1)
        idx_contain = contains_column(edges, c_del)
        if idx_contain >= 0:
            mask[idx_contain] = False 
        # print(idx_contain) 
        # print("c_del: ", c_del)
    edges_new = edges[:, mask] 
    # print(edges_new)
    # print("\n"*2)
    # exit()


    A=np.zeros((n,n),dtype=np.float32)
    A[edges_new[0],edges_new[1]]=1
    A = np.eye(n) + A
    d = A.sum(axis=0) 
    dis=1/np.sqrt(d)
    dis[np.isinf(dis)]=0
    dis[np.isnan(dis)]=0
    D=np.diag(dis)
    nL=A.dot(D).T.dot(D)

    V,U = np.linalg.eigh(nL)
    hist, _ = np.histogram(V, bins=np.linspace(-1, 1, num=100), density=True)

    new_G = nx.from_numpy_matrix(nL)

    if measure == "load_centrality":
        output = nx.load_centrality(new_G) 
        samp_vec = list(output.values()) 
        ego_center_value_after = samp_vec[ego_center]
        # print(ego_center)'
    elif measure == "betweenness_centrality":
        output = nx.betweenness_centrality(new_G)
        samp_vec = list(output.values())
        ego_center_value_after = samp_vec[ego_center]
    elif measure == "closeness_centrality":
        output = nx.closeness_centrality(new_G)
        samp_vec = np.array(list(output.values()))

        # ego_centers = np.argsort(-samp_vec)[:args.ego_top_k]
        ego_center_values_after = samp_vec[ego_centers]
    elif measure == "eigenvector_centrality":
        output = nx.eigenvector_centrality(new_G, max_iter=1000, tol=1e-4)
        samp_vec = list(output.values())
        ego_center_value_after = samp_vec[ego_center]

    print("egocenters: ", ego_centers, "\t", "value_before: ", ego_center_values_before, "\t", "value after: ", ego_center_values_after)    
    # print(f"egocenter: {ego_center}, value before: {ego_center_value}, value after: {ego_center_value_after}")
    # print("\n")
    
    return hist 
    

def statistics_remove_edge_based_on_edgeScores(n, edges, args):
    A=np.zeros((n,n),dtype=np.float32)
    A[edges[0],edges[1]]=1
    A = np.eye(n) + A
    d = A.sum(axis=0) 
    dis=1/np.sqrt(d)
    dis[np.isinf(dis)]=0
    dis[np.isnan(dis)]=0
    D=np.diag(dis)
    nL=A.dot(D).T.dot(D)

    G = nx.from_numpy_matrix(nL)

    if args.remove_edge_score_measure == "adamic_adar_index":
        result_gene = nx.adamic_adar_index(G, G.edges())
    elif args.remove_edge_score_measure == "jaccard_coefficient":
        result_gene = nx.adamic_adar_index(G, G.edges())
    
    edges, scores = [] 
    for (u, v, score) in result_gene:
        edges.append((u, v))
        scores.append(score)
    
    