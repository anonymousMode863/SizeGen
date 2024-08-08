import torch 
import torch.nn.functional as F
def edge_index_removal(old_edge_index, remove_edge_list_representation):
    # Requirement: Old_edge_index is tensor input; remove_edge is in list representation
    # e.g Old edge index: [ [0, 1], [1, 0], [3, 4], [4, 3]  ]  remove_edge : [(0,1)]
    # The first one is undirected while the second one aint so 

    remove_edges = remove_edge_list_representation 

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
    mask = torch.ones(old_edge_index.shape[1], dtype=torch.bool)
    for c_del_index in range(delete_edge.shape[1]):
        c_del = delete_edge[:, c_del_index].unsqueeze(-1)
        idx_contain = contains_column(old_edge_index, c_del)
        if idx_contain >= 0:
            mask[idx_contain] = False 
        # print(idx_contain) 
        # print("c_del: ", c_del)
    edges_new =  old_edge_index[:, mask] 

    return edges_new

def get_top_k_by_sorted_list(sorted_index_list, k, data_list):
    """
    Obtaining top k elements of data_list given sorted_index_list 
    Requirement:
        sorted_index_list - a well-defined sorted index of the data_list wrt. to some metric
        k - number of outputs you want 
        data_list - original data list 
    Output:
        a list containing the top k elements 
    """

    output = []

    if k >= len(sorted_index_list):
        keep_index_list = sorted_index_list 
    else:
        keep_index_list = sorted_index_list[:k] 
    
    for item in keep_index_list:
        output.append(data_list[item])
    return output 
    

def get_top_k_percent_by_sorted_list(sorted_index_list, k, data_list):
    """
    Obtaining top k percent elements of data_list given sorted_index_list 
    Requirement:
        sorted_index_list - a well-defined sorted index of the data_list wrt. to some metric
        k - percentage of outputs you want 
        data_list - original data list 
    Output:
        a list containing the top k elements 
    """
    assert 0 <= k  and k <= 1 
    output = []
    orignal_len = len(sorted_index_list) 
    num_keep = int(k * orignal_len)

    keep_index_list = sorted_index_list[:num_keep]
    
    for item in keep_index_list:
        output.append(data_list[item])
    return output 





leaky_relu = torch.nn.LeakyReLU()
def final_attention(input_feature, W1, W2):
    """
    Apply attention on the output layer 
    Input:
        W1 - learnable matrix with shape (output_shape, attention_size)
        W2 - learnable matrix with shape (attention_size, 1)
        input_feature - feature matrix with shape (size, output_shape)
    """

    dot_product = input_feature @ W1 
    dot_product = leaky_relu(dot_product) 
    importance = dot_product @ W2 

    soft_max_output = F.softmax(importance) # need to check the output 
    weighted_sum = torch.sum(input_feature * soft_max_output, dim=0)
    
    return weighted_sum


