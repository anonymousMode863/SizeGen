import torch
from torch_geometric.data import InMemoryDataset
from torch_geometric.data.data import Data
from torch_geometric.utils import to_networkx
from torch_geometric.utils import to_undirected
import numpy as np
import networkx as nx
import pickle
import os
import scipy.io as sio
from math import comb
from itertools import combinations, groupby
import random
import argparse


#### same way of generation but different parameters #########
#### method = ER, BA ##########
class graph_gen(object):
    def __init__(self, seed, path, method, num_train, num_test, graph_size, graph_size_range, para_train, para_test, is_large):
        # self.seed = 1234567
        self.seed = seed
        self.path = path
        self.method = method
        self.num_train = num_train
        self.num_test = num_test
        self.graph_size = graph_size
        self.graph_size_range = graph_size_range
        self.para_train = para_train
        self.para_test = para_test
        self.is_large = is_large
    
    def connect_graph(self, G):
        components = dict(enumerate(nx.connected_components(G)))
        components_combs = combinations(components.keys(), r=2)

        for _, node_edges in groupby(components_combs, key=lambda x: x[0]):
            node_edges = list(node_edges)
            # print(node_edges)
            # break
            random_comps = random.choice(node_edges)
            source = random.choice(list(components[random_comps[0]]))
            target = random.choice(list(components[random_comps[1]]))
            G.add_edge(source, target)
        return G


    def generate(self, num, para):
        data_list = []
        y = np.random.binomial(size=num, n=1, p= 0.5)
        small = not self.is_large
        for i in range(num):
            if small:
                graph_size = random.choice(range(self.graph_size-self.graph_size_range, self.graph_size+1))
            else:
                graph_size = random.choice(range(self.graph_size*2, self.graph_size*2+1+self.graph_size_range))
            if self.method == "ER":
                p = para/(graph_size*1.0)
                g = nx.fast_gnp_random_graph(graph_size, p, seed=self.seed)
            else:
                g = nx.watts_strogatz_graph(graph_size, para, 0.5, seed=self.seed)
            ######### insert subgraphs #########
            if y[i] < 0.5:
                h = nx.cubical_graph()
            else:
                h = nx.octahedral_graph()
            g_new = nx.disjoint_union(g, h)
            # for j in range(self.graph_size, self.graph_size+h.number_of_nodes()):
            #     if np.random.rand()>p:
            #         g_new.add_edge(j, np.random.randint(0,self.graph_size))
            g_new = self.connect_graph(g_new)
            # node_in_old = random.choice(range(graph_size))
            # node_in_new = random.choice(range(graph_size, graph_size+h.number_of_nodes()))
            # g_new.add_edge(node_in_old, node_in_new)
            # print(g_new)
            # degree_sequence = [[d] for n, d in g_new.degree()]
            # x = torch.FloatTensor(degree_sequence)
            x = torch.FloatTensor([[1.0]]*g_new.number_of_nodes())
            # print(x.shape)
            edge_index=torch.transpose(torch.LongTensor([list(e) for e in g_new.edges]), 1,0)
            edge_index2=torch.flip(edge_index,[0,1])
            edge_index=torch.cat((edge_index, edge_index2), 1)
            # print(edge_index.shape)
            yi=torch.FloatTensor([[y[i]]])
            # print(yi.shape)
            data = Data(x=x, edge_index=edge_index, y=yi)
            data_list.append(data)
        return data_list
    
    def save(self):
        train_data = self.generate(self.num_train, self.para_train)
        print(len(train_data))
        test_data = self.generate(self.num_test, self.para_test)
        data = train_data + test_data
        with open(os.path.join(self.path, "raw/simulation.pkl"), 'wb') as handle:
            pickle.dump(data, handle)
       




class SimulationDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super(SimulationDataset, self).__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["simulation.pkl"]

    @property
    def processed_file_names(self):
        return 'data.pt'

    def download(self):
        pass  

    def process(self):
        # Read data into huge `Data` list.
        data_list = pickle.load(open(os.path.join(self.root, "raw/simulation.pkl"), "rb"))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])



class SpectralDesign(object):   

    def __init__(self,recfield=1,dv=5,nfreq=5,adddegree=False,laplacien=True,addadj=False,vmax=None):
        # receptive field. 0: adj, 1; adj+I, n: n-hop area 
        self.recfield=recfield  
        # b parameter
        self.dv=dv
        # number of sampled point of spectrum
        self.nfreq=  nfreq
        # if degree is added to node feature
        self.adddegree=adddegree
        # use laplacian or adjacency for spectrum
        self.laplacien=laplacien
        # add adjacecny as edge feature
        self.addadj=addadj
        # use given max eigenvalue
        self.vmax=vmax

        # # max node for PPGN algorithm, set 0 if you do not use PPGN
        # self.nmax=nmax 


    def __call__(self, data):

        # data = Data(**data.__dict__)
        # print(data.y.shape)
        if data.x is None:
            data.x=torch.FloatTensor([[1.0]]*data.num_nodes)
        n =data.x.shape[0]     
        nf=data.x.shape[1]  

        data.x=data.x.type(torch.float32)  

        data.y = data.y.type(torch.float32)
        if len(data.y.shape)==1:
            data.y = data.y[:, None]
               
        nsup=self.nfreq+1
        if self.addadj:
            nsup+=1
            
        A=np.zeros((n,n),dtype=np.float32)
        SP=np.zeros((nsup,n,n),dtype=np.float32) 
        # print(data.edge_index)
        A[data.edge_index[0],data.edge_index[1]]=1
        
        if self.adddegree:
            data.x=torch.cat([data.x,torch.tensor(A.sum(0)).unsqueeze(-1)],1)

        # calculate receptive field. 0: adj, 1; adj+I, n: n-hop area
        if self.recfield==0:
            M=A
        else:
            M=(A+np.eye(n))
            for i in range(1,self.recfield):
                M=M.dot(M) 

        M=(M>0)

        
        d = A.sum(axis=0) 
        # normalized Laplacian matrix.
        dis=1/np.sqrt(d)
        dis[np.isinf(dis)]=0
        dis[np.isnan(dis)]=0
        D=np.diag(dis)
        nL=np.eye(D.shape[0])-(A.dot(D)).T.dot(D)
        V,U = np.linalg.eigh(nL) 
        V[V<0]=0
        # keep maximum eigenvalue for Chebnet if it is needed
        data.lmax=V.max().astype(np.float32)
        

        if not self.laplacien:        
            V,U = np.linalg.eigh(A)

        # design convolution supports
        vmax=self.vmax
        if vmax is None:
            vmax=V.max()

        freqcenter=np.linspace(V.min(),vmax,self.nfreq)
        
        # design convolution supports (aka edge features)         
        for i in range(0,len(freqcenter)): 
            SP[i,:,:]=M* (U.dot(np.diag(np.exp(-(self.dv*(V-freqcenter[i])**2))).dot(U.T))) 
        # add identity
        SP[len(freqcenter),:,:]=np.eye(n)
        # add adjacency if it is desired
        if self.addadj:
            SP[len(freqcenter)+1,:,:]=A
           
        # set convolution support weigths as an edge feature
        E=np.where(M>0)
        data.edge_index2=torch.Tensor(np.vstack((E[0],E[1]))).type(torch.int64)
        data.edge_attr2 = torch.Tensor(SP[:,E[0],E[1]].T).type(torch.float32)  

        return data





# class Degdesign(object):   

#     def __init__(self, k_hop=3, dim=8):
#         self.k_hop = k_hop
#         self.dim = dim
    
#     def __call__(self, data):
#         n =data.x.shape[0]     
#         A=np.zeros((n,n),dtype=np.float32)
#         A[data.edge_index[0],data.edge_index[1]]=1
#         SP=np.zeros((self.k_hop,n,self.dim),dtype=np.float32) 
#         d = A.sum(axis=0, keepdims=True) 

#         M = A
#         k = 0
#         while True:
#             h1 = M*d
#             for r in range(n):
#                 hr = h1[r,:]
#                 hr = hr[hr>0]
#                 SP[k,r,:]=np.histogram(hr, bins=self.dim)[0]
#             M=M.dot(M) 
#             M = (M>0)
#             k += 1
#             if k==self.k_hop:
#                 break
#         SP = torch.Tensor(SP).type(torch.float32)
#         SP = torch.transpose(SP, 0, 1)
#         SP = torch.reshape(SP, (n, -1))
#         data.x = torch.cat([data.x, SP],dim=-1)
#         # print(data.x.shape)
#         # print(data.x.shape)
#         # print(data.SP.shape)
#         return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=1234567, type=int, help='random seed')
    parser.add_argument('--folder', default="dataset", type=str, help='dataset folder')
    parser.add_argument('--method', default="WS", choices=["ER", "WS"], help='background graph type')
    parser.add_argument('--num_train', default=2000, type=int, help='number of training sample')
    parser.add_argument('--num_test', default=100, type=int, help='number of test sample')
    parser.add_argument('--graph_size', default=30, type=int, help='graph size')
    parser.add_argument('--graph_range', default=20, type=int, help='graph size range')
    parser.add_argument('--para_train', default=4, type=float, help='para_train')
    parser.add_argument('--para_test', default=4, type=float, help='para_test')
    parser.add_argument('--large', action='store_true')

    args = parser.parse_args()
    seed = args.seed
    np.random.seed(seed)
    path = args.folder + "/" + args.method
    method = args.method
    num_train = args.num_train
    num_test = args.num_test
    graph_size = args.graph_size
    para_train = args.para_train
    para_test = args.para_test
    graph_size_range = args.graph_range
    path += "_" + str(para_train).replace('.', '') + "_" + str(para_test).replace('.', '') + args.large*"_large"+"/"
    if not os.path.exists(path):
        os.mkdir(path)
        os.mkdir(path+"raw/")
    
    g_gen = graph_gen(seed, path, method, num_train, num_test, graph_size, graph_size_range, para_train, para_test, args.large)
    g_gen.save()



if __name__ == "__main__":
    main()
    

