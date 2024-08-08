from libs.utils_new import SimulationDataset,SpectralDesign
from torch_geometric.data import DataLoader, Data 
from torch_geometric.datasets import TUDataset
from ogb.graphproppred import PygGraphPropPredDataset
import numpy as np 
import torch 
from torch_geometric.data import InMemoryDataset
import os 
import networkx as nx 
import random 
from utils import edge_index_removal
from torch_geometric.utils.convert import to_networkx, from_networkx

# This is based on egonet measrues 
def remove_edge_index(n, edges, measure="load_centrality", args=None):
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
        output = nx.closeness_centrality(G)
        samp_vec = np.array(list(output.values()))
        # print(samp_vec)
        # print(np.argsort(-np.array(samp_vec)))
        # print(np.argmax(samp_vec))
        ego_centers = np.argsort(-samp_vec)[:args.ego_top_k]
        ego_center_values_before = samp_vec[ego_centers]
    else:
        raise NotImplementedError("This Measure not Implemented!")
    
    ego_nets = []
    

    for ego_center in ego_centers:
        ego_nets.append(nx.ego_graph(G, ego_center, undirected=True, radius=args.ego_rad))
    
    remove_edges = []
    for ego_net in ego_nets:
        remove_edges.extend(list(ego_net.edges()))
    remove_edges = [(i, j) for (i, j) in remove_edges if i != j] # Do not consider self_connections
    
    if args.use_edge_betweeness:
        edge_betweness = nx.edge_betweenness_centrality(G)
        
        remove_edge_btwness = [ edge_betweness[(i,j)] if i <= j else edge_betweness[(j,i)] for (i,j) in remove_edges ]    
        highest_index_by_betw = np.argsort(- np.array(remove_edge_btwness))

    # sampling do here 
    if args.ego_sub_percent is not None:
        remove_edges = np.array(remove_edges)
        remove_part = int(len(remove_edges) * args.ego_sub_percent)
        if args.use_edge_betweeness:
            remove_edges = remove_edges[highest_index_by_betw[:remove_part]]
        else:
            shuffle = np.random.permutation(len(remove_edges))
            remove_edges = remove_edges[shuffle]
            remove_edges =  list(remove_edges[:remove_part])
        remove_edges = [(i, j) for i,j in remove_edges]

    remove_edges = [*set(remove_edges)]

    if args.ego_sub_num is not None and len(remove_edges) > args.ego_sub_num:
        remove_edges = random.sample(remove_edges, k=args.ego_sub_num)
    
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
    
    return edges_new

def remove_edge_based_on_edge_scores(input_data, args=None):
    edges = input_data.edge_index
    n = input_data.x.shape[0] 

    data = Data(x=input_data.x, edge_index=edges) 
    # data = Data
    

    G = to_networkx(data, to_undirected=True)
    print(G)
    print("G edges: ", G.edges())
    
    # This measure is intended to be used when no self-loops are PRESENT 
    if args.remove_edge_score_measure == "adamic_adar_index":
        result_gene = nx.adamic_adar_index(G, G.edges())
    elif args.remove_edge_score_measure == "jaccard_coefficient":
        result_gene = nx.jaccard_coefficient(G, G.edges())
    
    # The above result is a generator 
    existing_edges, scores = [], []
    for (u, v, score) in result_gene:
        # if u != v: # Note: Do not consider self-connections, resulting from np.eye computation
        existing_edges.append((u, v))
        scores.append(score)
    
    print("edges: ", existing_edges)
    print("scores: ", scores)

    remove_threshold = 0.10
    num_to_remove = int(len(existing_edges) * remove_threshold)

    # Remove the top percent 
    sorted_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True) # Equivalent to np.argsort(-np.array(scores))
    removing_indices = sorted_indices[:num_to_remove] 
    remove_edges = [existing_edges[i] for i in removing_indices] 
    print("remove_edges: ", remove_edges) 
    new_edge_index = edge_index_removal(old_edge_index=edges, remove_edge_list_representation=remove_edges)
    print(edges)
    print(new_edge_index)





# This is for egonet-based measures 
class ModifiedDataset(InMemoryDataset):
    def __init__(self, args, pre_dataset, root_exp='dataset_adjusted',split_name = "train", transform= None, pre_transform= None, 
                 pre_filter= None):
        
        temp_name = f"{args.dataset}_{args.egonet_measure}_{args.dataset}_rad{args.ego_rad}_topk{args.ego_top_k}"
        if args.ego_sub_percent is not None:
            temp_name = temp_name + f"_subPercent{args.ego_sub_percent}"
        if args.ego_sub_num is not None:
            temp_name = temp_name + f"_subNum{args.ego_sub_num}"
        if args.not_remove_val_test:
            temp_name = temp_name + "_notremoveValtest"
        if args.use_edge_betweeness:
            temp_name = temp_name + "_edgeBetweeness"
        root = os.path.join(root_exp, temp_name, split_name)
        self.pre_dataset = pre_dataset
        self.args = args 
        self.transforms = SpectralDesign(recfield=1, dv=2, nfreq=5)

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['graph_data.pt']

    # def __len__(self) -> int:
    #     return len(self.label)

    def process(self):
        data_list = []

        for sample in self.pre_dataset:
            # print(sample)
            sample.edge_index = remove_edge_index(sample.x.shape[0], sample.edge_index.cpu(),measure=self.args.egonet_measure, args=self.args)
            sample = self.transforms(sample)            
            data_list.append(sample)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

def get_centrality_rank_high(data, threshold=0.5):
    n, edges = data.x.shape[0], data.edge_index.cpu()
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

    nodes = G.nodes()
    node_score = []
    for node in nodes:
        subgraph = nx.ego_graph(G, node, undirected=True, radius=3)
        closeness_score = nx.closeness_centrality(subgraph)[node]
        node_score.append(closeness_score)

    high_closenss_mask = np.array(node_score) > 0.50
    low_closenss_mask = np.array(node_score) < 0.45
    if np.all(low_closenss_mask) or np.all(high_closenss_mask) or np.all(~low_closenss_mask) or np.all(~high_closenss_mask):
        if len(node_score) == 1:
            low_closenss_mask = np.array([True])
            high_closenss_mask = np.array([True])
        else:
            max_element_index = np.argmax(node_score)
            high_closenss_mask = np.zeros(len(node_score), dtype=bool)
            high_closenss_mask[max_element_index] = True  
            low_closenss_mask = ~ high_closenss_mask 
    low_closenss_mask = torch.from_numpy(low_closenss_mask)
    high_closenss_mask = torch.from_numpy(high_closenss_mask)
    return low_closenss_mask, high_closenss_mask, node_score

def get_centrality_rank_extreme(data, threshold=0.32):
    n, edges = data.x.shape[0], data.edge_index.cpu()
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

    nodes = G.nodes()
    node_score = []
    for node in nodes:
        subgraph = nx.ego_graph(G, node, undirected=True, radius=3)
        closeness_score = nx.closeness_centrality(subgraph)[node]
        node_score.append(closeness_score)
    node_score = np.array(node_score)
    min_value = np.min(node_score)
    max_value = np.max(node_score)
    low_closenss_mask = node_score == min_value
    high_closenss_mask = node_score == max_value

    if np.all(~low_closenss_mask):
        if len(node_score) == 1:
            low_closenss_mask = np.array([True])
            high_closenss_mask = np.array([True])
        else:
            min_element_index = np.argmin(node_score)
            low_closenss_mask = np.zeros(len(node_score), dtype=bool)
            low_closenss_mask[min_element_index] = True  
    if np.all(~high_closenss_mask):
        if len(node_score) == 1:
            low_closenss_mask = np.array([True])
            high_closenss_mask = np.array([True])
        else:
            max_element_index = np.argmax(node_score)
            high_closenss_mask = np.zeros(len(node_score), dtype=bool)
            high_closenss_mask[max_element_index] = True 

    great_q1 = ~low_closenss_mask 
    less_q3 = ~high_closenss_mask
    middle = np.logical_and(great_q1, less_q3)
    if np.all(~middle):
        middle = np.ones(low_closenss_mask.shape, dtype=bool)
    
    low_closenss_mask = torch.from_numpy(low_closenss_mask)
    high_closenss_mask = torch.from_numpy(high_closenss_mask)
    middle = torch.from_numpy(middle)
    return low_closenss_mask, high_closenss_mask, middle 

def get_centrality_rank(data, threshold=0.32):
    n, edges = data.x.shape[0], data.edge_index.cpu()
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

    nodes = G.nodes()
    node_score = []
    for node in nodes:
        subgraph = nx.ego_graph(G, node, undirected=True, radius=3)
        closeness_score = nx.closeness_centrality(subgraph)[node]
        node_score.append(closeness_score)
    # print(node_score)
    node_score = np.array(node_score)
    # median_score = np.median(node_score)
    low_closenss_mask = node_score < threshold
    high_closenss_mask = ~low_closenss_mask 
    if np.all(low_closenss_mask) or np.all(high_closenss_mask):
        if len(node_score) == 1:
            low_closenss_mask = np.array([True])
            high_closenss_mask = np.array([True])
        else:
            min_element_index = np.argmin(node_score)
            low_closenss_mask = np.zeros(len(node_score), dtype=bool)
            low_closenss_mask[min_element_index] = True  
            high_closenss_mask = ~ low_closenss_mask 
    low_closenss_mask = torch.from_numpy(low_closenss_mask)
    high_closenss_mask = torch.from_numpy(high_closenss_mask)
    return low_closenss_mask, high_closenss_mask

def get_bins(args, curr_node_scores, num_bins=5):
    if args.dataset == "bbbp":
        min_val = 0.15
        max_val = 1.0 
        log_min = np.log10(min_val)
        log_max = np.log10(max_val)
        bins = np.logspace(log_min, log_max, num_bins-1)
        bins = np.insert(bins, 0, 0)
    elif args.dataset == "bace":
        min_val = 0.24
        max_val = 0.84 
        log_min = np.log10(min_val)
        log_max = np.log10(max_val)
        bins = np.logspace(log_min, log_max, num_bins+1)
    hist, _ = np.histogram(curr_node_scores, bins=bins, density=True)
    return hist 


def get_n_hop_local_closeness(G, node, radius):
    subgraph = nx.ego_graph(G, node, undirected=True, radius=radius) 
    subgraph_closeness = nx.closeness_centrality(subgraph)
    return subgraph_closeness[node]

def create_closeness_feature(data, args):
    """
    Feature Enignerring on the closeness
    1. closeness
    2. 3-hop local closeness max value 
    3. 3-hop local closeness min value 
    4. closeness average 
    5. closeness std
    """
    n, edges = data.x.shape[0], data.edge_index.cpu()
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
    features = []
    node_score = []
    for node in nodes:
        curr_node_feature = []
       
        subgraph = nx.ego_graph(G, node, undirected=True, radius=args.numHops)
        subgraph_closenss =  nx.closeness_centrality(subgraph)
        curr_closeness_scores = list(subgraph_closenss.values())
        closeness_score = subgraph_closenss[node]
        node_score.append(closeness_score)

        curr_node_feature.append(closeness_score)

        if args.use_entire_features:
            curr_node_feature.append(np.max(curr_closeness_scores)) # max 
            curr_node_feature.append(np.min(curr_closeness_scores)) # min 
            curr_node_feature.append(np.average(curr_closeness_scores)) # avg 
            curr_node_feature.append(np.std(curr_closeness_scores)) # std 
        elif args.use_one_hop_features:
            one_hop_nodes = nx.ego_graph(G, node, undirected=True, radius=1).nodes()
            one_hop_closeness = [subgraph_closenss[i] for i in one_hop_nodes]
            curr_node_feature.append(np.max(one_hop_closeness)) 
            curr_node_feature.append(np.min(one_hop_closeness))
            curr_node_feature.append(np.average(one_hop_closeness)) 
            curr_node_feature.append(np.std(one_hop_closeness))

        features.append(curr_node_feature)
    features = torch.tensor(features, dtype=torch.float32)
    return features

def create_closeness_feature_bins(data, args):
    """
    Feature Enignerring on the closeness
    based on bins
    """
    n, edges = data.x.shape[0], data.edge_index.cpu()
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

    nodes = G.nodes()
    features = []
    for node in nodes:
        subgraph = nx.ego_graph(G, node, undirected=True, radius=3)
        subgraph_closenss =  nx.closeness_centrality(subgraph)
        curr_node_scores = list(subgraph_closenss.values())

        curr_node_scores = np.array(curr_node_scores)
        if args.dataset == "bbbp":
            min_val = 0.005
            max_val = 1.0 
            log_min = np.log10(min_val)
            log_max = np.log10(max_val)
            bins = np.logspace(log_min, log_max, 64)
            bins = np.insert(bins, 0, 0)
            hist, _ = np.histogram(curr_node_scores, bins=bins, density=True)
        elif args.dataset == "bace":
            min_val = 0.24
            max_val = 0.84 
            log_min = np.log10(min_val)
            log_max = np.log10(max_val)
            bins = np.logspace(log_min, log_max, 65)
            hist, _ = np.histogram(curr_node_scores, bins=bins, density=True)
        features.append(list(hist))
        
    features = torch.tensor(features, dtype=torch.float32)
    return features


def append_local_closeness(data, args):
    """
    local closeness returned
    """
    n, edges = data.x.shape[0], data.edge_index.cpu()
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

    nodes = G.nodes()
    features = []
    for node in nodes:
        subgraph = nx.ego_graph(G, node, undirected=True, radius=3)
        curr_node_score =  nx.closeness_centrality(subgraph)[node]
        features.append(curr_node_score)
    
    features = torch.tensor(features, dtype=torch.float32)
    features = torch.unsqueeze(features, dim=-1)
    return features


class CloseFeatureDataset(InMemoryDataset):
    def __init__(self, args, pre_dataset, root_exp='dataset_Handcrafted', split_name="train", transform= None, pre_transform= None, 
                 pre_filter= None):
        
        self.pre_dataset = pre_dataset
        self.args = args 
        dataset = args.dataset
        
        root_name = f"closeness_{args.numHops}Hops"
        
        if args.use_entire_features:
            root_name += "_entireFeatures"
        elif args.use_one_hop_features:
            root_name += "_onehopFeatures"
        else:
            root_name += "_localClosenessOnly"
        
        root = os.path.join(root_exp, dataset, root_name, split_name)

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['graph_data.pt']


    def process(self):
        data_list = []

        # statistics = []
        for sample in self.pre_dataset:
            
            feature = create_closeness_feature(sample, self.args)
            data_list.append(Data(x=sample.x, edge_index=sample.edge_index, y = sample.y,
                                  num_nodes = sample.num_nodes, 
                                  closenes_feature=feature,
                                  edge_attr2=sample.edge_attr2,
                                  edge_index2=sample.edge_index2,
                                  lmax=sample.lmax))
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
    

class MetaDataset(InMemoryDataset):
    def __init__(self, args, pre_dataset, root_exp='dataset_Combined', split_name="train", transform= None, pre_transform= None, 
                 pre_filter= None):
        

        self.pre_dataset = pre_dataset
        self.args = args 
        dataset = args.dataset
        
        root_name = ""
        if args.closeness_engieering:
            root_name += f"closeness_{args.numHops}Hops"
            print("Using Closeness as features!")
        if args.remove_edge and not args.statistics_mode:
            print("Removing Edges based on closeness!")
            self.transforms = SpectralDesign(recfield=1, dv=2, nfreq=5)
            temp_name = f"{args.egonet_measure}_{args.dataset}_rad{args.ego_rad}_topk{args.ego_top_k}"
            if args.ego_sub_percent is not None:
                temp_name = temp_name + f"_subPercent{args.ego_sub_percent}"
            if args.ego_sub_num is not None:
                temp_name = temp_name + f"_subNum{args.ego_sub_num}"
            if args.not_remove_val_test:
                temp_name = temp_name + "_notremoveValtest"
            if args.use_edge_betweeness:
                temp_name = temp_name + "_edgeBetweeness"
            root_name += temp_name
       
        
        root = os.path.join(root_exp, dataset, root_name, split_name)

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['graph_data.pt']


    def process(self):
        data_list = []

        for sample in self.pre_dataset:
            
            feature = create_closeness_feature(sample, self.args)
            removed_edge_index = remove_edge_index(sample.x.shape[0], sample.edge_index.cpu(),measure=self.args.egonet_measure, args=self.args)
            sample.edge_index = removed_edge_index 
            sample = self.transforms(sample) 
            data_list.append(Data(x=sample.x, edge_index=removed_edge_index, y = sample.y,
                                num_nodes = sample.num_nodes, 
                                closenes_feature=feature,
                                edge_attr2=sample.edge_attr2,
                                edge_index2=sample.edge_index2,
                                lmax=sample.lmax))
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])






def modifie_centrality_rank(data, args):
    """
    Include extreme, median mode 
    """
    n, edges = data.x.shape[0], data.edge_index.cpu()
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

    nodes = G.nodes()
    node_score = []
    for node in nodes:
        subgraph = nx.ego_graph(G, node, undirected=True, radius=3)
        closeness_score = nx.closeness_centrality(subgraph)[node]
        node_score.append(closeness_score)
    node_score = np.array(node_score)
    if args.centrality_median_mode:
        low_value = np.median(node_score)
        high_value = np.median(node_score)
    elif args.centrality_25_quantile_mode:
        low_value = np.percentile(node_score, 25)
        high_value = np.percentile(node_score, 75)
    elif args.centrality_fixed_threshold_mode:
        low_value = args.centrality_low_value
        high_value = args.centrality_high_value
    elif args.extreme_centrality_mode:
        low_value = np.min(node_score) 
        high_value = np.max(node_score)
    else:
        raise NotImplementedError("Not Implemented for centrality mode")
   
    low_closenss_mask = node_score <= low_value
    high_closenss_mask = node_score >= high_value

    if np.all(~low_closenss_mask):
        if len(node_score) == 1:
            low_closenss_mask = np.array([True])
            high_closenss_mask = np.array([True])
        else:
            min_element_index = np.argmin(node_score)
            low_closenss_mask = np.zeros(len(node_score), dtype=bool)
            low_closenss_mask[min_element_index] = True  
    if np.all(~high_closenss_mask):
        if len(node_score) == 1:
            low_closenss_mask = np.array([True])
            high_closenss_mask = np.array([True])
        else:
            max_element_index = np.argmax(node_score)
            high_closenss_mask = np.zeros(len(node_score), dtype=bool)
            high_closenss_mask[max_element_index] = True 

    low_closenss_mask = torch.from_numpy(low_closenss_mask)
    high_closenss_mask = torch.from_numpy(high_closenss_mask)
    return low_closenss_mask, high_closenss_mask

# This is for Attention on node embeddings
class AttentionDataset(InMemoryDataset):
    def __init__(self, args, pre_dataset, root_exp='dataset_attention', split_name="train", transform= None, pre_transform= None, 
                 pre_filter= None):

        self.pre_dataset = pre_dataset
        self.args = args 
        dataset = args.dataset
        if args.centrality_median_mode:
            print(f"Doing Median Mode Centrality for {dataset}")
            root_name = "median"
        elif args.centrality_25_quantile_mode:
            print(f"Doing Quantile Mode Centrality for {dataset}!")
            root_name = "25_quantile"
        elif args.centrality_fixed_threshold_mode:
            print(f"Doing Fixed threshold Centrality for {dataset}!")
            root_name = f"{args.centrality_low_value}_{args.centrality_high_value}"
        elif args.extreme_centrality_mode:
            print(f"Doing Extreme Centrality for {dataset}!")
            root_name = "extreme"
        else:
            raise NotImplementedError("Not Implemented for centrality mode")
        
        root = os.path.join(root_exp, dataset, root_name, split_name)

        super().__init__(root, transform, pre_transform, pre_filter, split_name)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['graph_data.pt']

    def process(self):
        data_list = []

        statistics = []
        for sample in self.pre_dataset:

            low_centrality_mask, high_centrality_mask = modifie_centrality_rank(sample, args=self.args)
            data_list.append(Data(x=sample.x, edge_index=sample.edge_index, y = sample.y,
                                  num_nodes = sample.num_nodes, 
                                  low_centrality_mask=low_centrality_mask,
                                  high_centrality_mask=high_centrality_mask,
                                  edge_attr2=sample.edge_attr2,
                                  edge_index2=sample.edge_index2,
                                  lmax=sample.lmax))
            
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

class UpsampledDataset(InMemoryDataset):
    """
    Address the data imbalance issues
    """
    def __init__(self, args, pre_dataset, root_exp='train_dataset_upsampled', transform= None, pre_transform= None, 
                 pre_filter= None):
        
        
        root = os.path.join(root_exp, args.dataset)
        self.pre_dataset = pre_dataset
        self.args = args 

        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return ['graph_data.pt']

   
    def process(self):
        data_list = []

        for idx, sample in enumerate(self.pre_dataset):
            
            if self.args.dataset == "bbbp":
                curr_y = sample.y.item()
                if curr_y == 1:
                    data_list.append(sample)
                elif curr_y == 0:
                    for _ in range(6):
                        data_list.append(sample) 
                else:
                    raise NotImplementedError("Check your dataset!")
            elif self.args.dataset == "bace":
                curr_y = sample.y.item()
                # print(curr_y)
                if curr_y == 0:
                    data_list.append(sample)
                elif curr_y == 1:
                    for _ in range(2):
                        data_list.append(sample) 
                else:
                    raise NotImplementedError("Check your dataset!")
            elif self.args.dataset == "PROTEINS":
                curr_y = sample.y.item()
                if curr_y == 1:
                    data_list.append(sample)
                elif curr_y == 0:
                    if idx % 2 == 0:
                        for _ in range(2):
                            data_list.append(sample)
                    else:
                        data_list.append(sample)
            else:
                raise NotImplementedError("Not Implemented yet!")

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]
        
        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])



from torch_geometric.data import Batch
        

def approximate_list(args, 
                    pre_dataset, 
                    root_exp='approximate', 
                    method='erodos_renyi',
                    p_up=1,
                    largest_size=132,):
    data_list = []
    for sample in pre_dataset:
    
        n_node = sample.num_nodes
        n_generated_node = largest_size - n_node 
        
        if n_generated_node == 0:
            data = sample 
        else:
            G = nx.erdos_renyi_graph(n=n_generated_node, p= p_up/n_generated_node, seed=0,
                                        directed=False)
            data2 = from_networkx(G)
            data2.edge_index += n_node 
            # x = torch.cat([sample.x, data2.x], dim=0)
            edge_index = torch.cat([sample.edge_index, data2.edge_index], dim=1)
            # print("sample: ",sample)
            # print(n_node)
            data = Data(edge_index=edge_index, 
                        num_nodes=largest_size, 
                        origin=n_node
                        )
            # print(data)
            # exit()
        data_list.append(data)
    return data_list

def prepare_dataset(args):
    transform = SpectralDesign(recfield=1,dv=2,nfreq=5)
    if args.dataset == 'syn':
        pre_dataset = SimulationDataset(root=args.train_path,pre_transform=transform)
        train = range(args.train_size)
        val = range(args.train_size, args.train_size+args.val_size)
        val_loader  = DataLoader(pre_dataset[val], batch_size=args.batch_size, shuffle=False)
        train_loader = DataLoader(pre_dataset[train], batch_size=args.batch_size, shuffle=True)

        dataset = SimulationDataset(root=args.test_path,pre_transform=transform)
        test = range(args.test_size)
        test_loader  = DataLoader(dataset[test], batch_size=args.batch_size, shuffle=False)
        # y_pre_s = 1
        y_s = 1
        x_s = 1
    else: #### toxcat pretrain, clintox train
        

        if args.dataset == "PROTEINS":
            pre_dataset = TUDataset(root='dataset',name=args.dataset, pre_transform=transform)
        else:
            pre_dataset = PygGraphPropPredDataset(name ="ogbg-mol"+args.dataset, pre_transform=transform) 

        # typical_sample = pre_dataset[0].y 
        # print(typical_sample)
        # exit()

        sizes = []
        for sample in pre_dataset:
            sizes.append(sample.x.shape[0])
        sizes = np.array(sizes)
        if args.mode!=1:
            print("small_to_large")
            sorted_index = np.argsort(sizes)
        else:
            print("large_to_small")
            sorted_index = np.argsort(sizes)[::-1]
        
        largest_size = np.max(sizes)
        
        split_exp = os.path.join("split_index", args.dataset)
        train_idx = np.load(os.path.join(split_exp, "train_idx.npy"))
        val_idx = np.load(os.path.join(split_exp, "val_idx.npy"))
        small_test_idx = np.load(os.path.join(split_exp, "small_test_idx.npy"))
        large_test_idx = np.load(os.path.join(split_exp, "large_test_idx.npy"))

        train_dataset = pre_dataset[train_idx]
        train_dataset = UpsampledDataset(args=args, pre_dataset=train_dataset) # Upsample
    
        val_dataset = pre_dataset[val_idx]
        small_test_dataset = pre_dataset[small_test_idx]
        large_test_dataset = pre_dataset[large_test_idx]
    
        if args.remove_edge and not args.statistics_mode and args.closeness_engieering:
            print("Using MetaClass")
            train_dataset = MetaDataset(args=args, pre_dataset=train_dataset, split_name="train")
            val_dataset = MetaDataset(args=args, pre_dataset=val_dataset, split_name="val")
            small_test_dataset = MetaDataset(args=args, pre_dataset=small_test_dataset, split_name="small_test")
            large_test_dataset = MetaDataset(args=args, pre_dataset=large_test_dataset, split_name="large_test")
        else:
            if args.remove_edge and not args.statistics_mode: # if it is in statistics, we don't want to modify the orignal dataset
                print("Using Modified Dataset!")
                train_dataset = ModifiedDataset(args=args, pre_dataset=train_dataset, split_name="train")
                if not args.not_remove_val_test:
                    val_dataset = ModifiedDataset(args=args, pre_dataset=val_dataset, split_name="val")
                    small_test_dataset = ModifiedDataset(args=args, pre_dataset=small_test_dataset, split_name="small_test")
                    large_test_dataset = ModifiedDataset(args=args, pre_dataset=large_test_dataset, split_name="large_test")
            elif args.attention_centrality_mode: 
                print("Using Centrality for attention on node embeddings!")
                train_dataset = AttentionDataset(args=args, pre_dataset=train_dataset, split_name="train")
                val_dataset = AttentionDataset(args=args, pre_dataset=val_dataset, split_name="val")
                small_test_dataset = AttentionDataset(args=args, pre_dataset=small_test_dataset, split_name="small_test")
                large_test_dataset = AttentionDataset(args=args, pre_dataset=large_test_dataset, split_name="large_test")
            elif args.closeness_engieering:
                print("HumanCrafted Features of closeness!")
                train_dataset = CloseFeatureDataset(args=args, pre_dataset=train_dataset, split_name="train")
                val_dataset = CloseFeatureDataset(args=args, pre_dataset=val_dataset, split_name="val")
                small_test_dataset = CloseFeatureDataset(args=args, pre_dataset=small_test_dataset, split_name="small_test")
                large_test_dataset = CloseFeatureDataset(args=args, pre_dataset=large_test_dataset, split_name="large_test")
            else:
                print("Not using Modified Dataset!")

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        x_s = train_dataset[0].x.shape[-1]
        y_s = train_dataset[0].y.shape[-1]
        test_loader_1 = DataLoader(small_test_dataset, batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(large_test_dataset, batch_size=args.batch_size, shuffle=False)
        

    return pre_dataset, sorted_index, train_loader, val_loader, test_loader_1, test_loader, x_s, y_s 