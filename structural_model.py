from torch_geometric.nn import (GINConv,global_mean_pool,global_max_pool, TopKPooling,SAGPooling, ASAPooling, GATConv,ChebConv,GCNConv, FAConv, APPNP, SAGEConv)
from torch.nn import Sequential, Linear, ReLU
from torch.nn import Parameter
import torch 
from torch_scatter import scatter_add, scatter_min
import torch.nn as nn
import numpy as np 
import torch.nn.functional as F
from libs.spect_conv import SpectConv,ML3Layer


# def structural
class GcnNet_StructuralFeatures(nn.Module):
    def __init__(self, args, x_s,  ys=1):
        super(GcnNet_StructuralFeatures, self).__init__()

        print("You are doing GCNNet with Closeness Structural features")
        self.args = args
        neuron=64
        self.conv1 = GCNConv(x_s, neuron, cached=False)
        self.conv2 = GCNConv(neuron, neuron, cached=False)
        self.conv3 = GCNConv(neuron, neuron, cached=False) 
        

        self.closeness_layer = torch.nn.Linear(args.attention_num_feature, 1)
        self.attention1 = torch.nn.Linear(neuron, 16)
        self.attention2 = torch.nn.Linear(16, ys)

        if args.aggregator == "size_max" or args.aggregator == "size_average":
            self.weight = Parameter(torch.Tensor(neuron))
            self.alpha = Parameter(torch.Tensor(1))
            self.beta = Parameter(torch.Tensor(1))
            self.gamma = Parameter(torch.Tensor(1))
            self.reset_parameters()
        
        if args.aggregator == "SAG_max":
            self.pool1 = SAGPooling(in_channels=neuron, min_score=args.baseline_threshold, GNN=GCNConv)
            self.pool2 = SAGPooling(in_channels=neuron, min_score=args.baseline_threshold, GNN=GCNConv)
            self.pool3 = SAGPooling(in_channels=neuron, min_score=args.baseline_threshold, GNN=GCNConv)
        elif args.aggregator == "TOPK_max":
            self.pool1 = TopKPooling(in_channels=neuron, min_score=args.baseline_threshold)
            self.pool2 = TopKPooling(in_channels=neuron, min_score=args.baseline_threshold)
            self.pool3 = TopKPooling(in_channels=neuron, min_score=args.baseline_threshold)
            
    
    def reset_parameters(self):
        torch.nn.init.zeros_(self.weight)
        torch.nn.init.constant_(self.alpha, self.args.alpha)
        torch.nn.init.zeros_(self.beta)
        torch.nn.init.constant_(self.gamma, self.args.gamma)   

    def forward(self, data, edge_weight=None):

        x=data.x
        edge_index=data.edge_index   
        
        if self.args.aggregator == "SAG_max" or self.args.aggregator == "TOPK_max":

            batch = data.batch
            x = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
            x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
                        
            x = F.relu(self.conv2(x, edge_index, edge_weight=edge_weight))
            x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
            
            x = F.relu(self.conv3(x, edge_index, edge_weight=edge_weight))
            x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
            x = global_max_pool(x, batch)
                                  
            x = F.relu(self.attention1(x)) 
            return self.attention2(x)
        else:
            x = F.relu(self.conv1(x, edge_index, edge_weight=edge_weight))
            x = F.relu(self.conv2(x, edge_index, edge_weight=edge_weight))
            x = F.relu(self.conv3(x, edge_index, edge_weight=edge_weight)) 
        

        if self.args.aggregator == "average_pool":
            x = global_mean_pool(x, data.batch)
            x = F.relu(self.attention1(x))
            return self.attention2(x)
        elif self.args.aggregator == "max_pool":
            x = global_max_pool(x, data.batch)
            x = F.relu(self.attention1(x))
            return self.attention2(x)
        elif self.args.aggregator == "size_max":
            num_nodes = scatter_add(data.batch.new_ones(x.size(0)), data.batch, dim=0)
            x = global_max_pool(x, data.batch)
            x = x-torch.outer(self.alpha*(torch.pow(num_nodes, self.gamma)-self.beta), self.weight)
            x = F.relu(self.attention1(x))
            return self.attention2(x)
        elif self.args.aggregator == "size_average":
            num_nodes = scatter_add(data.batch.new_ones(x.size(0)), data.batch, dim=0)
            x = global_mean_pool(x, data.batch)
            x = x-torch.outer(self.alpha*(torch.pow(num_nodes, self.gamma)-self.beta), self.weight)
            x = F.relu(self.attention1(x))
            return self.attention2(x)
        
        # Attention on each input graph
        closeness_hand_feature = data.closenes_feature
        size = int(data.batch.max().item() + 1)
        batch = data.batch 
        tensor_cat_helper = [ ]
        for i in range(size):
            mask = torch.eq(batch, i) 
            input_tensor =  x[mask]
            input_closeness_feature = closeness_hand_feature[mask]
            curr_size = input_tensor.shape[0]
            input_closeness_feature = self.closeness_layer(input_closeness_feature) 
            input_closeness_feature = F.softmax(input_closeness_feature, dim=0)

            if self.args.aggregator == "self_max":
                input_closeness_feature = input_closeness_feature * curr_size 
                attentioned, _= torch.max(input_closeness_feature * input_tensor, dim=0)
            elif self.args.aggregator == "self_average":
                input_closeness_feature = input_closeness_feature * curr_size #* self.alpha
                attentioned = torch.mean(input_closeness_feature * input_tensor, dim=0)
            else:
                raise NotImplementedError("Error in aggregator") 
            attentioned = F.relu(self.attention1(attentioned))
            output = self.attention2(attentioned)
            output = torch.unsqueeze_copy(output, dim=0)
            tensor_cat_helper.append(output)

        x = torch.vstack(tensor_cat_helper)
        return x

class GraphSAGE_Structural(nn.Module):
    def __init__(self, args, x_s, ys=1):
        super(GraphSAGE_Structural, self).__init__()
        self.args = args
        neuron=64
        
        
        self.conv1 = SAGEConv(x_s, neuron, aggr="max")
        self.conv2 = SAGEConv(neuron, neuron, aggr="max")
        self.conv3 = SAGEConv(neuron, neuron, aggr="max") 
        

        self.closeness_layer = torch.nn.Linear(args.attention_num_feature, 1)
        self.attention1 = torch.nn.Linear(neuron, 16)
        self.attention2 = torch.nn.Linear(16, ys)

        if args.aggregator == "size_max" or args.aggregator == "size_average":
            self.weight = Parameter(torch.Tensor(neuron))
            self.alpha = Parameter(torch.Tensor(1))
            self.beta = Parameter(torch.Tensor(1))
            self.gamma = Parameter(torch.Tensor(1))
            self.reset_parameters()
        
        if args.aggregator == "SAG_max":
            self.pool1 = SAGPooling(in_channels=neuron, min_score=args.baseline_threshold, GNN=GCNConv)
            self.pool2 = SAGPooling(in_channels=neuron, min_score=args.baseline_threshold, GNN=GCNConv)
            self.pool3 = SAGPooling(in_channels=neuron, min_score=args.baseline_threshold, GNN=GCNConv)
        elif args.aggregator == "TOPK_max":
            self.pool1 = TopKPooling(in_channels=neuron, min_score=args.baseline_threshold)
            self.pool2 = TopKPooling(in_channels=neuron, min_score=args.baseline_threshold)
            self.pool3 = TopKPooling(in_channels=neuron, min_score=args.baseline_threshold)
    
    def reset_parameters(self):
        torch.nn.init.zeros_(self.weight)
        torch.nn.init.constant_(self.alpha, self.args.alpha)
        torch.nn.init.zeros_(self.beta)
        torch.nn.init.constant_(self.gamma, self.args.gamma)

    def forward(self, data):

        x=data.x 
        edge_index=data.edge_index

        if self.args.aggregator == "SAG_max" or self.args.aggregator == "TOPK_max":

            batch = data.batch
            x = F.relu(self.conv1(x, edge_index))
            x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
                        
            x = F.relu(self.conv2(x, edge_index))
            x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
            
            x = F.relu(self.conv3(x, edge_index))
            x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
            x = global_max_pool(x, batch)
                                  
            x = F.relu(self.attention1(x)) 
            return self.attention2(x)
        else:
            x = F.relu(self.conv1(x, edge_index))
            x = F.relu(self.conv2(x, edge_index))
            x = F.relu(self.conv3(x, edge_index))   

        if self.args.aggregator == "average_pool":
            x = global_mean_pool(x, data.batch)
            x = F.relu(self.attention1(x))
            return self.attention2(x)
        elif self.args.aggregator == "max_pool":
            x = global_max_pool(x, data.batch)
            x = F.relu(self.attention1(x))
            return self.attention2(x)       
        elif self.args.aggregator == "size_max":
            num_nodes = scatter_add(data.batch.new_ones(x.size(0)), data.batch, dim=0)
            x = global_max_pool(x, data.batch)
            x = x-torch.outer(self.alpha*(torch.pow(num_nodes, self.gamma)-self.beta), self.weight)
            x = F.relu(self.attention1(x))
            return self.attention2(x)
        elif self.args.aggregator == "size_average":
            num_nodes = scatter_add(data.batch.new_ones(x.size(0)), data.batch, dim=0)
            x = global_mean_pool(x, data.batch)
            x = x-torch.outer(self.alpha*(torch.pow(num_nodes, self.gamma)-self.beta), self.weight)
            x = F.relu(self.attention1(x))
            return self.attention2(x)  

        closeness_hand_feature = data.closenes_feature
        size = int(data.batch.max().item() + 1)
        batch = data.batch 
        tensor_cat_helper = [ ]

        for i in range(size):
            mask = torch.eq(batch, i) 
            input_tensor =  x[mask]
            input_closeness_feature = closeness_hand_feature[mask]
            curr_size = input_tensor.shape[0]
                        
            input_closeness_feature = self.closeness_layer(input_closeness_feature) 
            input_closeness_feature = F.softmax(input_closeness_feature, dim=0)

            if self.args.aggregator == "self_max":
                input_closeness_feature = input_closeness_feature * curr_size #* self.alpha 
                attentioned, _= torch.max(input_closeness_feature * input_tensor, dim=0)
            elif self.args.aggregator == "self_average":
                input_closeness_feature = input_closeness_feature * curr_size #* self.alpha
                attentioned = torch.mean(input_closeness_feature * input_tensor, dim=0)
            else:
                raise NotImplementedError("Error in aggregator!")      
                  
            attentioned = F.relu(self.attention1(attentioned))
            output = self.attention2(attentioned)
            output = torch.unsqueeze_copy(output, dim=0)
            tensor_cat_helper.append(output)

        x = torch.vstack(tensor_cat_helper)
        return x

class GinNet_Structural(nn.Module):
    def __init__(self, args, x_s, ys=1):
        super(GinNet_Structural, self).__init__()
        self.args = args
        neuron=64
        r1=np.random.uniform()
        r2=np.random.uniform()
        r3=np.random.uniform()

        nn1 = Sequential(Linear(x_s, neuron))
        self.conv1 = GINConv(nn1,eps=r1,train_eps=True)        

        nn2 = Sequential(Linear(neuron, neuron))
        self.conv2 = GINConv(nn2,eps=r2,train_eps=True)        

        nn3 = Sequential(Linear(neuron, neuron))
        self.conv3 = GINConv(nn3,eps=r3,train_eps=True) 
        
        
        
        self.closeness_layer = torch.nn.Linear(args.attention_num_feature, 1)
        self.attention1 = torch.nn.Linear(neuron, 16)
        self.attention2 = torch.nn.Linear(16, ys)
        
        if args.aggregator == "SAG_max":
            self.pool1 = SAGPooling(in_channels=neuron, min_score=args.baseline_threshold, GNN=GCNConv)
            self.pool2 = SAGPooling(in_channels=neuron, min_score=args.baseline_threshold, GNN=GCNConv)
            self.pool3 = SAGPooling(in_channels=neuron, min_score=args.baseline_threshold, GNN=GCNConv)
        elif args.aggregator == "TOPK_max":
            self.pool1 = TopKPooling(in_channels=neuron, min_score=args.baseline_threshold)
            self.pool2 = TopKPooling(in_channels=neuron, min_score=args.baseline_threshold)
            self.pool3 = TopKPooling(in_channels=neuron, min_score=args.baseline_threshold)

        if args.aggregator == "size_max" or args.aggregator == "size_average":
            self.weight = Parameter(torch.Tensor(neuron))
            self.alpha = Parameter(torch.Tensor(1))
            self.beta = Parameter(torch.Tensor(1))
            self.gamma = Parameter(torch.Tensor(1))
            self.reset_parameters()
    
    def reset_parameters(self):
        torch.nn.init.zeros_(self.weight)
        torch.nn.init.constant_(self.alpha, self.args.alpha)
        torch.nn.init.zeros_(self.beta)
        torch.nn.init.constant_(self.gamma, self.args.gamma)

    def forward(self, data):

        x=data.x 
        edge_index=data.edge_index

        if self.args.aggregator == "SAG_max" or self.args.aggregator == "TOPK_max":

            batch = data.batch
            x = F.relu(self.conv1(x, edge_index))
            x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
                        
            x = F.relu(self.conv2(x, edge_index))
            x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
            
            x = F.relu(self.conv3(x, edge_index))
            x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
            x = global_max_pool(x, batch)
                                  
            x = F.relu(self.attention1(x)) 
            return self.attention2(x)
        else:
            x = F.relu(self.conv1(x, edge_index))
            x = F.relu(self.conv2(x, edge_index))
            x = F.relu(self.conv3(x, edge_index))   

        if self.args.aggregator == "average_pool":
            x = global_mean_pool(x, data.batch)
            x = F.relu(self.attention1(x))
            return self.attention2(x)
        elif self.args.aggregator == "max_pool":
            x = global_max_pool(x, data.batch)
            x = F.relu(self.attention1(x))
            return self.attention2(x)       
        elif self.args.aggregator == "size_max":
            num_nodes = scatter_add(data.batch.new_ones(x.size(0)), data.batch, dim=0)
            x = global_max_pool(x, data.batch)
            x = x-torch.outer(self.alpha*(torch.pow(num_nodes, self.gamma)-self.beta), self.weight)
            x = F.relu(self.attention1(x))
            return self.attention2(x)
        elif self.args.aggregator == "size_average":
            num_nodes = scatter_add(data.batch.new_ones(x.size(0)), data.batch, dim=0)
            x = global_mean_pool(x, data.batch)
            x = x-torch.outer(self.alpha*(torch.pow(num_nodes, self.gamma)-self.beta), self.weight)
            x = F.relu(self.attention1(x))
            return self.attention2(x)  

        closeness_hand_feature = data.closenes_feature
        size = int(data.batch.max().item() + 1)
        batch = data.batch 
        tensor_cat_helper = [ ]

        for i in range(size):
            mask = torch.eq(batch, i) 
            input_tensor =  x[mask]
            input_closeness_feature = closeness_hand_feature[mask]
            curr_size = input_tensor.shape[0]
                        
            input_closeness_feature = self.closeness_layer(input_closeness_feature) 
            input_closeness_feature = F.softmax(input_closeness_feature, dim=0)

            if self.args.aggregator == "self_max":
                input_closeness_feature = input_closeness_feature * curr_size #* self.alpha 
                attentioned, _= torch.max(input_closeness_feature * input_tensor, dim=0)
            elif self.args.aggregator == "self_average":
                input_closeness_feature = input_closeness_feature * curr_size #* self.alpha
                attentioned = torch.mean(input_closeness_feature * input_tensor, dim=0)
            else:
                raise NotImplementedError("Error in aggregator!")      
                  
            attentioned = F.relu(self.attention1(attentioned))
            output = self.attention2(attentioned)
            output = torch.unsqueeze_copy(output, dim=0)
            tensor_cat_helper.append(output)

        x = torch.vstack(tensor_cat_helper)
        return x


class ChebNet_Structural(nn.Module):
    def __init__(self, args,x_s, S=5, ys=1):
        super(ChebNet_Structural, self).__init__()
        neuron=64 
        self.args = args
        self.conv1 = ChebConv(x_s, 32,S)
        self.conv2 = ChebConv(32, 64, S)
        self.conv3 = ChebConv(64, 64, S)        

        self.closeness_layer = torch.nn.Linear(args.attention_num_feature, 1)
        self.attention1 = torch.nn.Linear(neuron, 16)
        self.attention2 = torch.nn.Linear(16, ys)

        if args.aggregator == "size_max" or args.aggregator == "size_average":
            self.weight = Parameter(torch.Tensor(neuron))
            self.alpha = Parameter(torch.Tensor(1))
            self.beta = Parameter(torch.Tensor(1))
            self.gamma = Parameter(torch.Tensor(1))
            self.reset_parameters()

        if args.aggregator == "SAG_max":
            self.pool1 = SAGPooling(in_channels=32, min_score=args.baseline_threshold, GNN=GCNConv)
            self.pool2 = SAGPooling(in_channels=neuron, min_score=args.baseline_threshold, GNN=GCNConv)
            self.pool3 = SAGPooling(in_channels=neuron, min_score=args.baseline_threshold, GNN=GCNConv)
        elif args.aggregator == "TOPK_max":
            self.pool1 = TopKPooling(in_channels=32, min_score=args.baseline_threshold)
            self.pool2 = TopKPooling(in_channels=neuron, min_score=args.baseline_threshold)
            self.pool3 = TopKPooling(in_channels=neuron, min_score=args.baseline_threshold)
            
    def reset_parameters(self):
        torch.nn.init.zeros_(self.weight)
        torch.nn.init.constant_(self.alpha, self.args.alpha)
        torch.nn.init.zeros_(self.beta)
        torch.nn.init.constant_(self.gamma, self.args.gamma)
        
    def forward(self, data):
        x=data.x
        edge_index=data.edge_index
        
        
        if self.args.aggregator == "SAG_max" or self.args.aggregator == "TOPK_max":

            batch = data.batch
            x = F.relu(self.conv1(x, edge_index,lambda_max=data.lmax,batch=data.batch))
            x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
                        
            x = F.relu(self.conv2(x, edge_index,lambda_max=data.lmax,batch=data.batch))
            x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
            
            x = F.relu(self.conv3(x, edge_index,lambda_max=data.lmax,batch=data.batch))
            x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
            x = global_max_pool(x, batch)
                                  
            x = F.relu(self.attention1(x)) 
            return self.attention2(x)
        else:
            x = F.relu(self.conv1(x, edge_index,lambda_max=data.lmax,batch=data.batch))              
            x = F.relu(self.conv2(x, edge_index,lambda_max=data.lmax,batch=data.batch))        
            x = F.relu(self.conv3(x, edge_index,lambda_max=data.lmax,batch=data.batch))
            
        

        if self.args.aggregator == "average_pool":
            x = global_mean_pool(x, data.batch)
            x = F.relu(self.attention1(x))
            return self.attention2(x)
        elif self.args.aggregator == "max_pool":
            x = global_max_pool(x, data.batch)
            x = F.relu(self.attention1(x))
            return self.attention2(x)
        elif self.args.aggregator == "size_max":
            num_nodes = scatter_add(data.batch.new_ones(x.size(0)), data.batch, dim=0)
            x = global_max_pool(x, data.batch)
            x = x-torch.outer(self.alpha*(torch.pow(num_nodes, self.gamma)-self.beta), self.weight)
            x = F.relu(self.attention1(x))
            return self.attention2(x)
        elif self.args.aggregator == "size_average":
            num_nodes = scatter_add(data.batch.new_ones(x.size(0)), data.batch, dim=0)
            x = global_mean_pool(x, data.batch)
            x = x-torch.outer(self.alpha*(torch.pow(num_nodes, self.gamma)-self.beta), self.weight)
            x = F.relu(self.attention1(x))
            return self.attention2(x)

        closeness_hand_feature = data.closenes_feature
        size = int(data.batch.max().item() + 1)
        batch = data.batch 
        tensor_cat_helper = [ ]

        for i in range(size):
            mask = torch.eq(batch, i) 
            input_tensor =  x[mask]
            input_closeness_feature = closeness_hand_feature[mask]
            curr_size = input_tensor.shape[0]
            
            input_closeness_feature = self.closeness_layer(input_closeness_feature) 
            input_closeness_feature = F.softmax(input_closeness_feature, dim=0)

            if self.args.aggregator == "self_max":
                input_closeness_feature = input_closeness_feature * curr_size #* self.alpha 
                attentioned, _= torch.max(input_closeness_feature * input_tensor, dim=0)
            elif self.args.aggregator == "self_average":
                input_closeness_feature = input_closeness_feature * curr_size #* self.alpha
                attentioned = torch.mean(input_closeness_feature * input_tensor, dim=0)
            else:
                raise NotImplementedError("Error in aggregator")
            
            attentioned = F.relu(self.attention1(attentioned))
            output = self.attention2(attentioned)
            output = torch.unsqueeze_copy(output, dim=0)
            tensor_cat_helper.append(output)

        x = torch.vstack(tensor_cat_helper)
        return x

    
class FANet_Structural(nn.Module):
    def __init__(self, args, x_s, ys=1):
        super(FANet_Structural, self).__init__()
        self.args = args
        neuron=64
        self.nn1 = Linear(x_s, neuron)
        self.conv1 = FAConv(neuron)
        self.conv2 = FAConv(neuron)
        self.conv3 = FAConv(neuron)
        
        self.closeness_layer = torch.nn.Linear(args.attention_num_feature, 1)
        self.attention1 = torch.nn.Linear(neuron, 16)
        self.attention2 = torch.nn.Linear(16, ys)
        
        if args.aggregator == "size_max" or args.aggregator == "size_average":
            self.weight = Parameter(torch.Tensor(neuron))
            self.alpha = Parameter(torch.Tensor(1))
            self.beta = Parameter(torch.Tensor(1))
            self.gamma = Parameter(torch.Tensor(1))
            self.reset_parameters()
        
        if args.aggregator == "SAG_max":
            self.pool1 = SAGPooling(in_channels=neuron, min_score=args.baseline_threshold, GNN=GCNConv)
            self.pool2 = SAGPooling(in_channels=neuron, min_score=args.baseline_threshold, GNN=GCNConv)
            self.pool3 = SAGPooling(in_channels=neuron, min_score=args.baseline_threshold, GNN=GCNConv)
        elif args.aggregator == "TOPK_max":
            self.pool1 = TopKPooling(in_channels=neuron, min_score=args.baseline_threshold)
            self.pool2 = TopKPooling(in_channels=neuron, min_score=args.baseline_threshold)
            self.pool3 = TopKPooling(in_channels=neuron, min_score=args.baseline_threshold)
            
    
    def reset_parameters(self):
        torch.nn.init.zeros_(self.weight)
        torch.nn.init.constant_(self.alpha, self.args.alpha)
        torch.nn.init.zeros_(self.beta)
        torch.nn.init.constant_(self.gamma, self.args.gamma)

    def forward(self, data):
        x=data.x 
        edge_index=data.edge_index
        
        x0 = F.relu(self.nn1(x))
        
        if self.args.aggregator == "SAG_max" or self.args.aggregator == "TOPK_max":

            batch = data.batch
            x = self.conv1(x0, x0, edge_index)
            x, edge_index, _, batch, perm, score_perm = self.pool1(x, edge_index, None, batch)
            x0 = x0[perm] * score_perm.view(-1, 1) # Follows from original implementaion
                      
            x = self.conv2(x, x0, edge_index)
            x, edge_index, _, batch, perm, score_perm = self.pool2(x, edge_index, None, batch)
            x0 = x0[perm] * score_perm.view(-1, 1) # Follows from original implementaion
            
            x = self.conv3(x, x0, edge_index)
            x, edge_index, _, batch, perm, score_perm = self.pool3(x, edge_index, None, batch)
            x0 = x0[perm] * score_perm.view(-1, 1) # Follows from original implementaion
            
            x = global_max_pool(x, batch)
                                  
            x = F.relu(self.attention1(x)) 
            return self.attention2(x)
        else:
            x = self.conv1(x0, x0, edge_index)
            x = self.conv2(x, x0, edge_index)
            x = self.conv3(x, x0, edge_index)
        
        if self.args.aggregator == "average_pool":
            x = global_mean_pool(x, data.batch)
            x = F.relu(self.attention1(x))
            return self.attention2(x)
        elif self.args.aggregator == "max_pool":
            x = global_max_pool(x, data.batch)
            x = F.relu(self.attention1(x))
            return self.attention2(x)
        elif self.args.aggregator == "size_max":
            num_nodes = scatter_add(data.batch.new_ones(x.size(0)), data.batch, dim=0)
            x = global_max_pool(x, data.batch)
            x = x-torch.outer(self.alpha*(torch.pow(num_nodes, self.gamma)-self.beta), self.weight)
            x = F.relu(self.attention1(x))
            return self.attention2(x)
        elif self.args.aggregator == "size_average":
            num_nodes = scatter_add(data.batch.new_ones(x.size(0)), data.batch, dim=0)
            x = global_mean_pool(x, data.batch)
            x = x-torch.outer(self.alpha*(torch.pow(num_nodes, self.gamma)-self.beta), self.weight)
            x = F.relu(self.attention1(x))
            return self.attention2(x)
        
        closeness_hand_feature = data.closenes_feature
        size = int(data.batch.max().item() + 1)
        batch = data.batch 
        tensor_cat_helper = [ ]

        for i in range(size):
            mask = torch.eq(batch, i) 
            input_tensor =  x[mask]
            input_closeness_feature = closeness_hand_feature[mask]
            curr_size = input_tensor.shape[0]
                        
            input_closeness_feature = self.closeness_layer(input_closeness_feature) 
            input_closeness_feature = F.softmax(input_closeness_feature, dim=0)

            if self.args.aggregator == "self_max":
                input_closeness_feature = input_closeness_feature * curr_size #* self.alpha 
                attentioned, _= torch.max(input_closeness_feature * input_tensor, dim=0)
            elif self.args.aggregator == "self_average":
                input_closeness_feature = input_closeness_feature * curr_size #* self.alpha
                attentioned = torch.mean(input_closeness_feature * input_tensor, dim=0)
            else:
                raise NotImplementedError("Error in aggregator!")      
                  
            attentioned = F.relu(self.attention1(attentioned))
            output = self.attention2(attentioned)
            output = torch.unsqueeze_copy(output, dim=0)
            tensor_cat_helper.append(output)

        x = torch.vstack(tensor_cat_helper)
        return x


class MlpNet_Stuctural(nn.Module):
    def __init__(self, args, x_s, ys=1):
        super(MlpNet_Stuctural, self).__init__()
        self.args = args
        neuron=64
        self.conv1 = torch.nn.Linear(x_s, neuron)
        self.conv2 = torch.nn.Linear(neuron, neuron)
        self.conv3 = torch.nn.Linear(neuron, neuron) 
        self.last = torch.nn.Linear(10, ys)
        
        self.closeness_layer = torch.nn.Linear(args.attention_num_feature, 1)
        self.attention1 = torch.nn.Linear(neuron, 16)
        self.attention2 = torch.nn.Linear(16, ys)
        
        if args.aggregator == "size_max" or args.aggregator == "size_average":
            self.weight = Parameter(torch.Tensor(neuron))
            self.alpha = Parameter(torch.Tensor(1))
            self.beta = Parameter(torch.Tensor(1))
            self.gamma = Parameter(torch.Tensor(1))
            self.reset_parameters()
        
        
        if args.aggregator == "SAG_max":
            self.pool1 = SAGPooling(in_channels=neuron, min_score=args.baseline_threshold, GNN=GCNConv)
            self.pool2 = SAGPooling(in_channels=neuron, min_score=args.baseline_threshold, GNN=GCNConv)
            self.pool3 = SAGPooling(in_channels=neuron, min_score=args.baseline_threshold, GNN=GCNConv)
        elif args.aggregator == "TOPK_max":
            self.pool1 = TopKPooling(in_channels=neuron, min_score=args.baseline_threshold)
            self.pool2 = TopKPooling(in_channels=neuron, min_score=args.baseline_threshold)
            self.pool3 = TopKPooling(in_channels=neuron, min_score=args.baseline_threshold)
    
    def reset_parameters(self):
        torch.nn.init.zeros_(self.weight)
        torch.nn.init.constant_(self.alpha, self.args.alpha)
        torch.nn.init.zeros_(self.beta)
        torch.nn.init.constant_(self.gamma, self.args.gamma)

    def forward(self, data):

        x=data.x 
        edge_index=data.edge_index

        if self.args.aggregator == "SAG_max" or self.args.aggregator == "TOPK_max":

            batch = data.batch
            x = F.relu(self.conv1(x))
            x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
                        
            x = F.relu(self.conv2(x))
            x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
            
            x = F.relu(self.conv3(x))
            x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
            x = global_max_pool(x, batch)
                                 
            x = F.relu(self.attention1(x)) 
            return self.attention2(x)
        else:
            x = F.relu(self.conv1(x))                
            x = F.relu(self.conv2(x))        
            x = F.relu(self.conv3(x))  

        if self.args.aggregator == "average_pool":
            x = global_mean_pool(x, data.batch)
            x = F.relu(self.attention1(x))
            return self.attention2(x)
        elif self.args.aggregator == "max_pool":
            x = global_max_pool(x, data.batch)
            x = F.relu(self.attention1(x))
            return self.attention2(x)   
        elif self.args.aggregator == "size_max":
            num_nodes = scatter_add(data.batch.new_ones(x.size(0)), data.batch, dim=0)
            x = global_max_pool(x, data.batch)
            x = x-torch.outer(self.alpha*(torch.pow(num_nodes, self.gamma)-self.beta), self.weight)
            x = F.relu(self.attention1(x))
            return self.attention2(x)
        elif self.args.aggregator == "size_average":
            num_nodes = scatter_add(data.batch.new_ones(x.size(0)), data.batch, dim=0)
            x = global_mean_pool(x, data.batch)
            x = x-torch.outer(self.alpha*(torch.pow(num_nodes, self.gamma)-self.beta), self.weight)
            x = F.relu(self.attention1(x))
            return self.attention2(x)       

        closeness_hand_feature = data.closenes_feature
        size = int(data.batch.max().item() + 1)
        batch = data.batch 
        tensor_cat_helper = [ ]

        for i in range(size):
            mask = torch.eq(batch, i) 
            input_tensor =  x[mask]
            input_closeness_feature = closeness_hand_feature[mask]
            curr_size = input_tensor.shape[0]
                        
            input_closeness_feature = self.closeness_layer(input_closeness_feature) 
            input_closeness_feature = F.softmax(input_closeness_feature, dim=0)

            if self.args.aggregator == "self_max":
                input_closeness_feature = input_closeness_feature * curr_size #* self.alpha 
                attentioned, _= torch.max(input_closeness_feature * input_tensor, dim=0)
            elif self.args.aggregator == "self_average":
                input_closeness_feature = input_closeness_feature * curr_size #* self.alpha
                attentioned = torch.mean(input_closeness_feature * input_tensor, dim=0)
            else:
                raise NotImplementedError("Error in aggregator!")      
                  
            attentioned = F.relu(self.attention1(attentioned))
            output = self.attention2(attentioned)
            output = torch.unsqueeze_copy(output, dim=0)
            tensor_cat_helper.append(output)

        x = torch.vstack(tensor_cat_helper)
        return x


class GatNet_Structural(nn.Module):
    def __init__(self, args, x_s, ys=1):
        super(GatNet_Structural, self).__init__()
        self.args = args

        '''number of param (in+3)*head*out
        '''
        print("Gat Num Heads: ", args.gat_numhead)
        print("Gat Hidden Size: ", args.gat_hiddensize)
        print("Gat OutChannel: ", args.gat_outchannel)
        
        self.conv1 = GATConv(x_s, args.gat_outchannel, heads=args.gat_numhead,concat=True, dropout=0.0)        
        self.conv2 = GATConv(args.gat_hiddensize, args.gat_outchannel, heads=args.gat_numhead, concat=True, dropout=0.0)
        self.conv3 = GATConv(args.gat_hiddensize, args.gat_outchannel, heads=args.gat_numhead, concat=True, dropout=0.0)
        
        # self.conv1 = GATConv(x_s, 16, heads=8,concat=True, dropout=0.0)        
        # self.conv2 = GATConv(128, 16, heads=8, concat=True, dropout=0.0)
        # self.conv3 = GATConv(128, 16, heads=8, concat=True, dropout=0.0)

        neuron = args.gat_hiddensize
        self.closeness_layer = torch.nn.Linear(args.attention_num_feature, 1)
        self.attention1 = torch.nn.Linear(neuron, 16)
        self.attention2 = torch.nn.Linear(16, ys)
        
        if args.aggregator == "size_max" or args.aggregator == "size_average":
            self.weight = Parameter(torch.Tensor(neuron))
            self.alpha = Parameter(torch.Tensor(1))
            self.beta = Parameter(torch.Tensor(1))
            self.gamma = Parameter(torch.Tensor(1))
            self.reset_parameters()
        
        if args.aggregator == "SAG_max":
            self.pool1 = SAGPooling(in_channels=neuron, min_score=args.baseline_threshold, GNN=GCNConv)
            self.pool2 = SAGPooling(in_channels=neuron, min_score=args.baseline_threshold, GNN=GCNConv)
            self.pool3 = SAGPooling(in_channels=neuron, min_score=args.baseline_threshold, GNN=GCNConv)
        elif args.aggregator == "TOPK_max":
            self.pool1 = TopKPooling(in_channels=neuron, min_score=args.baseline_threshold)
            self.pool2 = TopKPooling(in_channels=neuron, min_score=args.baseline_threshold)
            self.pool3 = TopKPooling(in_channels=neuron, min_score=args.baseline_threshold)
    
    def reset_parameters(self):
        torch.nn.init.zeros_(self.weight)
        torch.nn.init.constant_(self.alpha, self.args.alpha)
        torch.nn.init.zeros_(self.beta)
        torch.nn.init.constant_(self.gamma, self.args.gamma)
           
    def forward(self, data):
        x=data.x 
        edge_index=data.edge_index
        
        if self.args.aggregator == "SAG_max" or self.args.aggregator == "TOPK_max":

            batch = data.batch
            x = F.relu(self.conv1(x, edge_index))
            x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
                        
            x = F.relu(self.conv2(x, edge_index))
            x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
            
            x = F.relu(self.conv3(x, edge_index))
            x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
            x = global_max_pool(x, batch)
                                  
            x = F.relu(self.attention1(x)) 
            return self.attention2(x)
        else: 
            x = F.relu(self.conv1(x, edge_index))
            x = F.relu(self.conv2(x, edge_index))        
            x = F.relu(self.conv3(x, edge_index))        

        if self.args.aggregator == "average_pool":
            x = global_mean_pool(x, data.batch)
            x = F.relu(self.attention1(x))
            return self.attention2(x)
        elif self.args.aggregator == "max_pool":
            x = global_max_pool(x, data.batch)
            x = F.relu(self.attention1(x))
            return self.attention2(x) 
        elif self.args.aggregator == "size_max":
            num_nodes = scatter_add(data.batch.new_ones(x.size(0)), data.batch, dim=0)
            x = global_max_pool(x, data.batch)
            x = x-torch.outer(self.alpha*(torch.pow(num_nodes, self.gamma)-self.beta), self.weight)
            x = F.relu(self.attention1(x))
            return self.attention2(x)
        elif self.args.aggregator == "size_average":
            num_nodes = scatter_add(data.batch.new_ones(x.size(0)), data.batch, dim=0)
            x = global_mean_pool(x, data.batch)
            x = x-torch.outer(self.alpha*(torch.pow(num_nodes, self.gamma)-self.beta), self.weight)
            x = F.relu(self.attention1(x))
            return self.attention2(x)    

        closeness_hand_feature = data.closenes_feature
        size = int(data.batch.max().item() + 1)
        batch = data.batch 
        tensor_cat_helper = [ ]

        for i in range(size):
            mask = torch.eq(batch, i) 
            input_tensor =  x[mask]
            input_closeness_feature = closeness_hand_feature[mask]
            curr_size = input_tensor.shape[0]
                        
            input_closeness_feature = self.closeness_layer(input_closeness_feature) 
            input_closeness_feature = F.softmax(input_closeness_feature, dim=0)

            if self.args.aggregator == "self_max":
                input_closeness_feature = input_closeness_feature * curr_size #* self.alpha 
                attentioned, _= torch.max(input_closeness_feature * input_tensor, dim=0)
            elif self.args.aggregator == "self_average":
                input_closeness_feature = input_closeness_feature * curr_size #* self.alpha
                attentioned = torch.mean(input_closeness_feature * input_tensor, dim=0)
            else:
                raise NotImplementedError("Error in aggregator!")      
                  
            attentioned = F.relu(self.attention1(attentioned))
            output = self.attention2(attentioned)
            output = torch.unsqueeze_copy(output, dim=0)
            tensor_cat_helper.append(output)

        x = torch.vstack(tensor_cat_helper)
        return x
    


class APPNPNet_Structural(nn.Module):
    def __init__(self, args, x_s, ys=1):
        super(APPNPNet_Structural, self).__init__()
        self.args = args
        neuron=64
        self.args = self.args 
        self.nn1 = Linear(x_s, neuron)
        self.conv1 = APPNP(K=3, alpha=0.1, dropout=0.2)

        self.closeness_layer = torch.nn.Linear(args.attention_num_feature, 1)
        self.attention1 = torch.nn.Linear(neuron, 16)
        self.attention2 = torch.nn.Linear(16, ys)

        if args.aggregator == "size_max" or args.aggregator == "size_average":
            self.weight = Parameter(torch.Tensor(neuron))
            self.alpha = Parameter(torch.Tensor(1))
            self.beta = Parameter(torch.Tensor(1))
            self.gamma = Parameter(torch.Tensor(1))
            self.reset_parameters()
    
    def reset_parameters(self):
        torch.nn.init.zeros_(self.weight)
        torch.nn.init.constant_(self.alpha, self.args.alpha)
        torch.nn.init.zeros_(self.beta)
        torch.nn.init.constant_(self.gamma, self.args.gamma)
    
    def forward(self, data):
        x=data.x 
        edge_index=data.edge_index
        
        
        x = self.conv1(x, edge_index)
        x = F.relu(self.nn1(x))
        
        

        if self.args.aggregator == "average_pool":
            x = global_mean_pool(x, data.batch)
            x = F.relu(self.attention1(x))
            return self.attention2(x)
        elif self.args.aggregator == "max_pool":
            x = global_max_pool(x, data.batch)
            x = F.relu(self.attention1(x))
            return self.attention2(x)
        elif self.args.aggregator == "size_max":
            num_nodes = scatter_add(data.batch.new_ones(x.size(0)), data.batch, dim=0)
            x = global_max_pool(x, data.batch)
            x = x-torch.outer(self.alpha*(torch.pow(num_nodes, self.gamma)-self.beta), self.weight)
            x = F.relu(self.attention1(x))
            return self.attention2(x)
        elif self.args.aggregator == "size_average":
            num_nodes = scatter_add(data.batch.new_ones(x.size(0)), data.batch, dim=0)
            x = global_mean_pool(x, data.batch)
            x = x-torch.outer(self.alpha*(torch.pow(num_nodes, self.gamma)-self.beta), self.weight)
            x = F.relu(self.attention1(x))
            return self.attention2(x)


        closeness_hand_feature = data.closenes_feature
        size = int(data.batch.max().item() + 1)
        batch = data.batch 
        tensor_cat_helper = [ ]

        for i in range(size):
            mask = torch.eq(batch, i) 
            input_tensor =  x[mask]
            input_closeness_feature = closeness_hand_feature[mask]
            curr_size = input_tensor.shape[0]
                        
            input_closeness_feature = self.closeness_layer(input_closeness_feature)
            # input_closeness_feature = self.batchnorm(input_closeness_feature) 
            input_closeness_feature = F.softmax(input_closeness_feature, dim=0)
            # print(input_closeness_feature.shape)
            if self.args.aggregator == "self_max":
                input_closeness_feature = input_closeness_feature * curr_size #* self.alpha 
                attentioned, _= torch.max(input_closeness_feature * input_tensor, dim=0)
            elif self.args.aggregator == "self_average":
                input_closeness_feature = input_closeness_feature * curr_size #* self.alpha
                attentioned = torch.mean(input_closeness_feature * input_tensor, dim=0)
            else:
                raise NotImplementedError("Error in aggregator!")      
                  
            attentioned = F.relu(self.attention1(attentioned))
            output = self.attention2(attentioned)
            output = torch.unsqueeze_copy(output, dim=0)
            tensor_cat_helper.append(output)

        x = torch.vstack(tensor_cat_helper)
        return x


class GNNML3_Structural(nn.Module):
    def __init__(self, args, x_s, pre_dataset=None, ys=1):
        super(GNNML3_Structural, self).__init__()
        self.args = args

        # number of neuron for for part1 and part2
        nout1=32
        nout2=16

        nin=nout1+nout2
        ne=pre_dataset.data.edge_attr2.shape[1]
        ninp=x_s

        self.conv1=ML3Layer(learnedge=True,nedgeinput=ne,nedgeoutput=ne,ninp=ninp,nout1=nout1,nout2=nout2)
        self.conv2=ML3Layer(learnedge=True,nedgeinput=ne,nedgeoutput=ne,ninp=nin ,nout1=nout1,nout2=nout2)
        self.conv3=ML3Layer(learnedge=True,nedgeinput=ne,nedgeoutput=ne,ninp=nin ,nout1=nout1,nout2=nout2) 
        
        neuron = nin
        self.closeness_layer = torch.nn.Linear(args.attention_num_feature, 1)
        self.attention1 = torch.nn.Linear(neuron, 16)
        self.attention2 = torch.nn.Linear(16, ys)

        if args.aggregator == "size_max" or args.aggregator == "size_average":
            self.weight = Parameter(torch.Tensor(neuron))
            self.alpha = Parameter(torch.Tensor(1))
            self.beta = Parameter(torch.Tensor(1))
            self.gamma = Parameter(torch.Tensor(1))
            self.reset_parameters()
        
        if args.aggregator == "SAG_max":
            self.pool1 = SAGPooling(in_channels=neuron, min_score=args.baseline_threshold, GNN=GCNConv)
            self.pool2 = SAGPooling(in_channels=neuron, min_score=args.baseline_threshold, GNN=GCNConv)
            self.pool3 = SAGPooling(in_channels=neuron, min_score=args.baseline_threshold, GNN=GCNConv)
        elif args.aggregator == "TOPK_max":
            self.pool1 = TopKPooling(in_channels=neuron, min_score=args.baseline_threshold)
            self.pool2 = TopKPooling(in_channels=neuron, min_score=args.baseline_threshold)
            self.pool3 = TopKPooling(in_channels=neuron, min_score=args.baseline_threshold)
    
    def reset_parameters(self):
        torch.nn.init.zeros_(self.weight)
        torch.nn.init.constant_(self.alpha, self.args.alpha)
        torch.nn.init.zeros_(self.beta)
        torch.nn.init.constant_(self.gamma, self.args.gamma)
           
    def forward(self, data):
        x=data.x
        edge_index=data.edge_index2
        edge_attr=data.edge_attr2
        
        if self.args.aggregator == "SAG_max" or self.args.aggregator == "TOPK_max":

            batch = data.batch
            x=(self.conv1(x, edge_index,edge_attr))
            x, edge_index, edge_attr, batch, _, _ = self.pool1(x, edge_index, edge_attr, batch)

            
            x = self.conv2(x, edge_index,edge_attr)
            x, edge_index, edge_attr, batch, _, _ = self.pool2(x, edge_index, edge_attr, batch)
        
            x = (self.conv3(x, edge_index,edge_attr)) 
            x, edge_index, edge_attr, batch, _, _ = self.pool3(x, edge_index, edge_attr, batch)
            x = global_max_pool(x, batch)
                                  
            x = F.relu(self.attention1(x)) 
            return self.attention2(x)
        else:
            x=(self.conv1(x, edge_index,edge_attr))
            x=(self.conv2(x, edge_index,edge_attr))
            x=(self.conv3(x, edge_index,edge_attr))  

        if self.args.aggregator == "average_pool":
            x = global_mean_pool(x, data.batch)
            x = F.relu(self.attention1(x))
            return self.attention2(x)
        elif self.args.aggregator == "max_pool":
            x = global_max_pool(x, data.batch)
            x = F.relu(self.attention1(x))
            return self.attention2(x)
        elif self.args.aggregator == "size_max":
            num_nodes = scatter_add(data.batch.new_ones(x.size(0)), data.batch, dim=0)
            x = global_max_pool(x, data.batch)
            x = x-torch.outer(self.alpha*(torch.pow(num_nodes, self.gamma)-self.beta), self.weight)
            x = F.relu(self.attention1(x))
            return self.attention2(x)
        elif self.args.aggregator == "size_average":
            num_nodes = scatter_add(data.batch.new_ones(x.size(0)), data.batch, dim=0)
            x = global_mean_pool(x, data.batch)
            x = x-torch.outer(self.alpha*(torch.pow(num_nodes, self.gamma)-self.beta), self.weight)
            x = F.relu(self.attention1(x))
            return self.attention2(x)

        closeness_hand_feature = data.closenes_feature
        size = int(data.batch.max().item() + 1)
        batch = data.batch 
        tensor_cat_helper = [ ]

        for i in range(size):
            mask = torch.eq(batch, i) 
            input_tensor =  x[mask]
            input_closeness_feature = closeness_hand_feature[mask]
            curr_size = input_tensor.shape[0]
                        
            input_closeness_feature = self.closeness_layer(input_closeness_feature) 
            input_closeness_feature = F.softmax(input_closeness_feature, dim=0)

            if self.args.aggregator == "self_max":
                input_closeness_feature = input_closeness_feature * curr_size #* self.alpha 
                attentioned, _= torch.max(input_closeness_feature * input_tensor, dim=0)
            elif self.args.aggregator == "self_average":
                input_closeness_feature = input_closeness_feature * curr_size #* self.alpha
                attentioned = torch.mean(input_closeness_feature * input_tensor, dim=0)
            else:
                raise NotImplementedError("Error in aggregator!")      
                  
            attentioned = F.relu(self.attention1(attentioned))
            output = self.attention2(attentioned)
            output = torch.unsqueeze_copy(output, dim=0)
            tensor_cat_helper.append(output)

        x = torch.vstack(tensor_cat_helper)
        return x


# class APPNPNet_Structural_batch1(nn.Module):
#     def __init__(self, args, x_s, ys=1):
#         super(APPNPNet_Structural_batch1, self).__init__()
#         self.args = args
#         neuron=64
#         self.args = self.args 
#         self.nn1 = Linear(x_s, neuron)
#         self.conv1 = APPNP(K=3, alpha=0.1, dropout=0.5)
        

    #     # self.closeness_layer = torch.nn.Linear(5, 1)
    #     # torch.nn.init.xavier_uniform_(self.closeness_layer.weight)
    #     # torch.nn.init.constant_(self.closeness_layer.bias, 0.0)
    #     self.attention1 = torch.nn.Linear(neuron, 16)
    #     self.attention2 = torch.nn.Linear(16, ys)
    #     self.alpha = Parameter(torch.tensor(1.0))
    
    # def forward(self, data):
    #     x=data.x 
    #     edge_index=data.edge_index
    #     closeness_hand_feature = data.closenes_feature
        
    #     curr_size = x.shape[0]
    #     input_closeness_feature = closeness_hand_feature 
    #     # input_closeness_feature = self.closeness_layer(input_closeness_feature) 
    #     input_closeness_feature = F.softmax(input_closeness_feature, dim=0)
    #     input_closeness_feature = input_closeness_feature 
    

    #     x = input_closeness_feature * x 
    #     x = self.conv1(x, edge_index)
    #     x = F.relu(self.nn1(x))

    #     if self.args.aggregator == "self_max":

    #         # attentioned, _= torch.max(x, dim=0)
    #         attentioned= torch.sum(x, dim=0)
    #     elif self.args.aggregator == "self_average":
    #         attentioned = torch.mean(x, dim=0)
    #     else:
    #         raise NotImplementedError("Error in aggregator!")
        
    #     attentioned = F.relu(self.attention1(attentioned))
    #     output = self.attention2(attentioned)
    #     output = torch.unsqueeze_copy(output, dim=-1)
    #     return output 