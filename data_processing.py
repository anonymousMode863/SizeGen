import argparse 
from dataset import SpectralDesign, SimulationDataset
from torch_geometric.data import DataLoader, Data 
from torch_geometric.datasets import TUDataset
from ogb.graphproppred import PygGraphPropPredDataset
import numpy as np 
import torch 
import os 
import random 
import pickle 
import math 
torch.manual_seed(255)
np.random.seed(255)
random.seed(255)

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default="bbbp")
parser.add_argument('--train_size', default=2000, type=int, help='size of training data (synthetic)')
parser.add_argument('--val_size', default=100, type=int, help='size of validatio data (synthetic)')
parser.add_argument('--test_size', default=500, type=int, help='size of test data (synthetic)')
parser.add_argument('--train_path', default="dataset/WS_4_4/", type=str, help='path for training dataset')
parser.add_argument('--test_path', default="dataset/WS_4_4_large/", type=str, help='path for test dataset')
args = parser.parse_args() 



"""
Aim: make the class distribution similar between 
small and large loader 
"""


# def get_loader_data(loader, dataset="bbbp"):

def check_distribution(dataset, dataset_type="bbbp"):
    if dataset_type == "bbbp" or dataset_type == "bace" or dataset_type=="PROTEINS" or dataset_type=="NCI1" or dataset_type == "NCI109" or dataset_type == "MOLT-4" \
        or dataset_type == "hiv" or args.dataset == "MCF-7":
        class_distribution = torch.zeros(2)
        for sample in dataset:
            # print(sample.y.item())
            class_distribution[int(sample.y.item())] += 1  
        # exit()
    elif dataset_type == "tox21" or dataset_type == "toxcast":
        if dataset_type == "tox21":
            y_size = 12 
        elif dataset_type == "toxcast":
            y_size = 617
        class_0_dist = torch.zeros(y_size)
        class_1_dist = torch.zeros(y_size) 
        for sample in dataset:
            # print(sample.y[0][0])
            # exit()
            for i in range(y_size):
                # print(sample.y[0][i].item())
                curr_y = sample.y[0][i].item()
                if not math.isnan(curr_y):
                        # print("inside if:")
                    # print(sample.y[0][i].item())
                    if curr_y == 0:
                        # print("here")
                        class_0_dist[i] += 1 
                    elif curr_y == 1:
                        class_1_dist[i] += 1 
                    else:
                        raise NotImplementedError("Error")
        class_distribution = (class_0_dist, class_1_dist)
    else:
        raise NotImplementedError("not implemented yet")
    return class_distribution 

def get_overall_dsitribution(train_dataset, val_dataset, small_test_dataset, large_test_dataset,
                             dataset_type="bbbp"):
    # if dataset_type
    # print(dataset_type)
    train_dist = check_distribution(train_dataset, dataset_type=dataset_type)
    # print(train_dist)
    # exit()
    val_dist = check_distribution(val_dataset, dataset_type=dataset_type)
    small_test_dist = check_distribution(small_test_dataset, dataset_type=dataset_type)
    large_test_dist = check_distribution(large_test_dataset, dataset_type=dataset_type)
    return train_dist, val_dist, small_test_dist, large_test_dist


def print_out_overall_distribution(train_dist, val_dist, small_test_dist, large_test_dist, args):
    if args.dataset == "bace" or args.dataset == "bbbp" or args.dataset == "PROTEINS" or args.dataset == "NCI1" or args.dataset == "NCI109" or args.dataset == "MOLT-4" \
        or args.dataset == "hiv" or args.dataset == "MCF-7":
        print("train dataset distribution: ", train_dist)
        print("val dataset distribution: ", val_dist)
        print("small test dataset distribution: ", small_test_dist)
        print("large test dataset distribution: ", large_test_dist)
    elif args.dataset == "tox21" or args.dataset == "toxcast":
        (train_0, train_1) = train_dist
        (val_0, val_1) = val_dist 
        (small_test_0, small_test_1) = small_test_dist 
        (large_test_0, large_test_1) = large_test_dist 

        print("train dataset class 0: ", train_0)
        print("train dataset class 1: ", train_1)
        print("val dataset class 0: ", val_0)
        print("val dataset class 1: ", val_1) 
        print("small test dataset class 0: ", small_test_0)
        print("small test dataset class 1: ", small_test_1)
        print("large test dataset class 0: ", large_test_0)
        print("large test dataset class 1: ", large_test_1)

        print("####################### train_val_smalltest_class_0 ################")
        print(train_0 + val_0 + small_test_0)

        print("####################### train_val_smalltest_class_1 ################")
        print(train_1 + val_1 + small_test_1)

        print("####################### train_val_smalltest_class_ratio ################")
        train_val_smalltest_class_ratio = torch.div(train_0 + val_0 + small_test_0, train_1 + val_1 + small_test_1)
        ones_index = (train_val_smalltest_class_ratio <= 10).nonzero().squeeze()
        print("one index: ", ones_index)
        # print(torch.div(train_0 + val_0 + small_test_0, train_1 + val_1 + small_test_1))

        print("####################### largetest_class_ratio ################")
        # large_test_class_ratio = torch.div(large_test_0, large_test_1)
        # print(large_test_class_ratio)
        # print("large ratio 354: ", large_test_class_ratio[354])
        # print("large ratio 382: ", large_test_class_ratio[382])
        # print("large ratio 428: ", large_test_class_ratio[428])
        # print("large ratio 451: ", large_test_class_ratio[451])
    else:
        raise NotImplementedError("Errors")



transform = SpectralDesign(recfield=1,dv=2,nfreq=5)


if args.dataset == "PROTEINS" or args.dataset == "NCI1" or args.dataset == "NCI109" or args.dataset == "MOLT-4" \
    or args.dataset == "MCF-7":
    pre_dataset = TUDataset(root='dataset', name = args.dataset, use_edge_attr=True, use_node_attr=True, pre_transform=transform)
else:
    pre_dataset = PygGraphPropPredDataset(name ="ogbg-mol"+args.dataset, pre_transform=transform) 

y_s = pre_dataset[0].y.shape[-1]
sizes = []
for sample in pre_dataset:
    sizes.append(sample.x.shape[0])
    
sizes = np.array(sizes)
sorted_index = np.argsort(sizes) # small to large 


# BBBP and Bace
num_test = np.ceil(sizes.shape[0]*0.10).astype(int)
num_train_val_test = np.ceil(sizes.shape[0]*0.50).astype(int)
if args.dataset == "PROTEINS":
    # This is to make the case clearer, otherwise the large dataset is not well-taken 
    # as in order to balance the large test distribution it requires more searches
    num_train_val_test = np.ceil(sizes.shape[0]*0.45).astype(int)
large_test_idx = sorted_index[-num_test:]
train_val_test_idx =  sorted_index[:num_train_val_test]

train_split, val_split, test_split = 0.7, 0.15, 0.15 
# if args.dataset == "PROTEINS":
#     train_split, val_split, test_split = 0.8, 0.10, 0.10
# bbbp and bace
if args.dataset == "bbbp" or args.dataset == "bace" or args.dataset == "PROTEINS" or args.dataset == "NCI1" or args.dataset == "NCI109" or args.dataset == "MOLT-4" \
    or args.dataset == "hiv" or args.dataset == "MCF-7":
    # Small loader checking first 
    ones_idx_list, zeros_idx_list = [], []
    for idx in range(len(pre_dataset)):
        if idx in train_val_test_idx and pre_dataset[idx].x.shape[0] != 1:
            if pre_dataset[idx].y.item() == 1:
                ones_idx_list.append(idx)
            elif pre_dataset[idx].y.item() == 0.0:
                zeros_idx_list.append(idx)
            else:
                raise NotImplementedError("Error in binary check if it is bbbp or bace!")
    ones_idx = np.array(ones_idx_list)
    zeros_idx = np.array(zeros_idx_list) 

    ones_idx_perm = np.random.permutation(ones_idx)
    zeros_idx_perm = np.random.permutation(zeros_idx)
    lens_ones = len(ones_idx_list)
    lens_zeros = len(zeros_idx_list)

    num_train_ones, num_test_ones = int(lens_ones * train_split), int(lens_ones * test_split) 
    num_train_zeros, num_test_zeros = int(lens_zeros * train_split), int(lens_zeros * test_split)
    
    train_one_idx, test_one_idx, val_one_idx = ones_idx_perm[:num_train_ones], \
        ones_idx_perm[num_train_ones:num_test_ones+num_train_ones], ones_idx_perm[num_test_ones + num_train_ones:]
    train_zero_idx, test_zero_idx, val_zero_idx = zeros_idx_perm[:num_train_zeros], \
        zeros_idx_perm[num_train_zeros:num_test_zeros+num_train_zeros], zeros_idx_perm[num_test_zeros + num_train_zeros:]
    
    print(f"train zero {len(train_zero_idx)}, train one {len(train_one_idx)}")
    print(f"test zero {len(test_zero_idx)}, test one {len(test_one_idx)}")
    print(f"val zero {len(val_zero_idx)}, val one {len(val_one_idx)}")

    train_idx = np.random.permutation(np.concatenate([train_one_idx, train_zero_idx]))
    val_idx = np.random.permutation(np.concatenate([val_one_idx, val_zero_idx]))
    small_test_idx = np.random.permutation(np.concatenate([test_one_idx, test_zero_idx]))

    # proceed to large 

    large_test_idx_recorder = []
    num_large_zeros, num_large_ones = 0, 0
    # if args.dataset == "PROTEINS":
        # Not take too many searches while still not too imblanced test set
    cap_large_zero = len(test_zero_idx)
    cap_large_one = len(test_one_idx)
    for i, idx in enumerate(np.flip(sorted_index)):
        sample = pre_dataset[idx] 
        if sample.y.item() == 0 and num_large_zeros < cap_large_zero:
            large_test_idx_recorder.append(idx)
            num_large_zeros += 1 
        elif sample.y.item() == 1 and num_large_ones < cap_large_one:
            num_large_ones += 1 
            large_test_idx_recorder.append(idx)
        elif sample.y.item() != 0 and sample.y.item() != 1:
            raise NotImplementedError("Check bbbp or bace!")
        if num_large_zeros >= cap_large_zero and num_large_ones >= cap_large_one:
            print(f"Finish finding the appropriate samples, cost {i} searches for {len(large_test_idx_recorder)} samples out of total size {len(pre_dataset)} samples!")
            break
    large_test_idx = np.array(large_test_idx_recorder) 
    large_test_idx = np.random.permutation(large_test_idx)

    save_address = os.path.join("split_index", args.dataset) 
    if not os.path.exists(save_address):
        os.makedirs(save_address)
    # exit()
    np.save(f"{save_address}/train_idx.npy", train_idx)
    np.save(f"{save_address}/val_idx.npy", val_idx)
    np.save(f"{save_address}/small_test_idx.npy", small_test_idx)
    np.save(f"{save_address}/large_test_idx.npy", large_test_idx)

elif args.dataset == "toxcast" or args.dataset == "tox21":
    if args.dataset == "toxcast":
        target_task = 145#501
    elif args.dataset == "tox21":
        target_task = 7

    ones_idx_list, zeros_idx_list = [], [] 

    for idx in range(len(pre_dataset)):
        if idx in train_val_test_idx and pre_dataset[idx].x.shape[0] != 1:
            curr_y = pre_dataset[idx].y[0][target_task].item()

            if not math.isnan(curr_y):
                if curr_y == 1:
                    ones_idx_list.append(idx)
                elif curr_y == 0:
                    zeros_idx_list.append(idx)
                else:
                    raise NotImplementedError("Error!")
    
    ones_idx = np.array(ones_idx_list)
    zeros_idx = np.array(zeros_idx_list)

    ones_idx_perm = np.random.permutation(ones_idx)
    zeros_idx_perm = np.random.permutation(zeros_idx)
    lens_ones = len(ones_idx_list)
    lens_zeros = len(zeros_idx_list)

    num_train_ones, num_test_ones = int(lens_ones * train_split), int(lens_ones * test_split) 
    num_train_zeros, num_test_zeros = int(lens_zeros * train_split), int(lens_zeros * test_split)

    train_one_idx, test_one_idx, val_one_idx = ones_idx_perm[:num_train_ones], \
        ones_idx_perm[num_train_ones:num_test_ones+num_train_ones], ones_idx_perm[num_test_ones + num_train_ones:]
    train_zero_idx, test_zero_idx, val_zero_idx = zeros_idx_perm[:num_train_zeros], \
        zeros_idx_perm[num_train_zeros:num_test_zeros+num_train_zeros], zeros_idx_perm[num_test_zeros + num_train_zeros:]

    print(f"train zero {len(train_zero_idx)}, train one {len(train_one_idx)}")
    print(f"test zero {len(test_zero_idx)}, test one {len(test_one_idx)}")
    print(f"val zero {len(val_zero_idx)}, val one {len(val_one_idx)}")

    train_idx = np.random.permutation(np.concatenate([train_one_idx, train_zero_idx]))
    val_idx = np.random.permutation(np.concatenate([val_one_idx, val_zero_idx]))
    small_test_idx = np.random.permutation(np.concatenate([test_one_idx, test_zero_idx]))

    large_test_idx_recorder = []
    num_large_zeros, num_large_ones = 0, 0
    for i, idx in enumerate(np.flip(sorted_index)):
        sample = pre_dataset[idx] 
        curr_y = pre_dataset[idx].y[0][target_task].item()
        if not math.isnan(curr_y):
            if curr_y == 0 and num_large_zeros < len(test_zero_idx):
                large_test_idx_recorder.append(idx)
                num_large_zeros += 1 
            elif curr_y == 1 and num_large_ones < len(test_one_idx):
                num_large_ones += 1 
                large_test_idx_recorder.append(idx)
            elif curr_y != 0 and curr_y != 1:
                raise NotImplementedError("Check tox21!")
            if num_large_zeros >= len(test_zero_idx) and num_large_ones >= len(test_one_idx):
                print(f"Finish finding the appropriate samples, cost {i} searches for {len(large_test_idx_recorder)} samples out of total size {len(pre_dataset)} samples!")
                break
    large_test_idx = np.array(large_test_idx_recorder) 
    large_test_idx = np.random.permutation(large_test_idx)

    save_address = os.path.join("split_index", args.dataset) 
    if not os.path.exists(save_address):
        os.makedirs(save_address)
    np.save(f"{save_address}/train_idx.npy", train_idx)
    np.save(f"{save_address}/val_idx.npy", val_idx)
    np.save(f"{save_address}/small_test_idx.npy", small_test_idx)
    np.save(f"{save_address}/large_test_idx.npy", large_test_idx)

train_dataset = pre_dataset[train_idx]
val_dataset = pre_dataset[val_idx]
small_test_dataset = pre_dataset[small_test_idx] 
large_test_dataset = pre_dataset[large_test_idx] 

train_dist, val_dist, small_test_dist, large_test_dist = get_overall_dsitribution(
    train_dataset, val_dataset, small_test_dataset, large_test_dataset, dataset_type=args.dataset
)
print_out_overall_distribution(train_dist, val_dist, small_test_dist, large_test_dist, args)



