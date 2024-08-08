import argparse
from dataset import prepare_dataset
from statistics import eigen, eigen_vec, average_connectivity, distribution_connectivity, get_scalar_plot, get_cheeser_difference_plot
from statistics import get_sim_matrix, get_deg_dist, remove_egonet_based_on_measure, local_closeness
import torch 
import numpy as np 
import random 
# from model import * 
from train import * 
import pandas as pd 
import os 
# from centrality_model import *
from structural_model import * 
torch.manual_seed(255)
np.random.seed(255)
random.seed(255)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
cls_criterion = torch.nn.BCEWithLogitsLoss()


def caller(args):
    pass 



def get_statistics(args, pre_dataset, sorted_index, graph_type="connectivity", 
                   connec_measure="closeness_centrality",egonet_measure="load_centrality"):
    # print("here")
    # exit()
    if graph_type == "eigen":
        if args.approximate_graph:
            name = f'{args.dataset}_eigen_p{args.approximate_p}'
        else:
            name = f'{args.dataset}_eigen'
        # print(name)
        # exit()
        get_sim_matrix(func=eigen, name=name, 
                       pre_dataset=pre_dataset, sorted_index=sorted_index, args=args, is_distribution=True)
    elif graph_type == "eigen_vec":
        get_sim_matrix(func=eigen_vec, name=f"{args.dataset}_eigen_vec_small", 
                       pre_dataset=pre_dataset, sorted_index=sorted_index, is_distribution=True)
    elif graph_type == "deg":
        get_sim_matrix(func=get_deg_dist, name=f"{args.dataset}_deg",
                       pre_dataset=pre_dataset, sorted_index=sorted_index, is_distribution=True)
    elif graph_type == "connectivity": 
        temp_name = f"{args.dataset}_{connec_measure}_connec"
        # get_sim_matrix(connectivity, temp_name, is_distribution=False, connect_measure=measure)
        get_sim_matrix(func=distribution_connectivity, name=temp_name,
                       pre_dataset=pre_dataset, sorted_index=sorted_index, 
                       connect_measure=connec_measure, is_distribution=True)
    
    elif graph_type == "remove_egonet":
        temp_name = f"remove_egonet_{egonet_measure}_{args.dataset}_rad{args.ego_rad}_topk{args.ego_top_k}"
        if args.ego_sub_percent is not None:
            temp_name = temp_name + f"_subPercent{args.ego_sub_percent}"
        if args.ego_sub_num is not None:
            temp_name = temp_name + f"_subNum{args.ego_sub_num}"
        get_sim_matrix(func=remove_egonet_based_on_measure, name=temp_name,
                       pre_dataset=pre_dataset, sorted_index=sorted_index, 
                       is_distribution=True, egonet_measure=egonet_measure, args=args)
    elif graph_type == "edge_score":
        temp_name = f"edge_score_{args.dataset}_{args.edge_score_measure}"
        get_sim_matrix(func=remove_egonet_based_on_measure, name=temp_name,
                       pre_dataset=pre_dataset, sorted_index=sorted_index, 
                       is_distribution=True, args=args, save_name="edge_score_measures")
    elif graph_type == "local_closeness":
        temp_name = f"local_closeness_{args.dataset}_numHops_{args.numHops}"
        get_sim_matrix(func=local_closeness, name=temp_name,
                       pre_dataset=pre_dataset, sorted_index=sorted_index, 
                       is_distribution=True, args=args, save_name="local_closeness")
    elif graph_type == "local_closeness_cheeser":
        temp_name = f"local_closeness_cheeser_{args.dataset}_numHops_{args.numHops}"
        get_sim_matrix(func=local_closeness_cheeser, name=temp_name,
                       pre_dataset=pre_dataset, sorted_index=sorted_index, 
                       is_distribution=True, args=args, save_name="local_closeness_cheeser")
    else:
        raise NotImplementedError("Invalid graph_type entered!")
    

def train_function(args, train_loader, val_loader, test_loader_1, test_loader, x_s, y_s, 
                   pre_dataset=None):
    roc_1_list = []
    roc_2_list = [] 
    f1_1_list = []
    f1_2_list = []
    acc_1_list = []
    acc_2_list = []
    
    val_f1s = []
    val_rocs = []
    val_accs = []
    

    # Model Selections
    for _ in range(args.num_runs):
        if args.model == "APPNPNet":
            model = APPNPNet_Structural(args, x_s=x_s, ys=y_s).to(device)
        elif args.model == "GatNet":
            model = GatNet_Structural(args, x_s=x_s, ys=y_s).to(device)
        elif args.model == "ChebNet":            
            model = ChebNet_Structural(args, x_s=x_s, ys=y_s).to(device)
        elif args.model == "GcnNet":
            model = GcnNet_StructuralFeatures(args=args, x_s=x_s, ys=y_s).to(device)
        elif args.model == "GinNet":
            model = GinNet_Structural(args, x_s=x_s, ys=y_s).to(device)
        elif args.model == "MlpNet":
            model = MlpNet_Stuctural(args, x_s=x_s, ys=y_s).to(device)
        elif args.model == "FANet":
            model = FANet_Structural(args, x_s=x_s, ys=y_s).to(device)
        elif args.model == "GraphSAGE":
            model = GraphSAGE_Structural(args, x_s=x_s, ys=y_s).to(device)
        else:
            model = GNNML3_Structural(args, x_s=x_s, pre_dataset=pre_dataset, ys=y_s).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        print("training:")
        if args.dataset=="syn":
            pth = args.train_path+args.model+"best.pt"
        else:
            pth = "dataset/"+args.dataset+"_"+args.model+"_best.pt"
        
        b1, b2, f1_1, f1_2, acc_1, acc_2, val_f1, val_roc, val_acc= train_data(args, train_loader, val_loader, test_loader_1, test_loader, model, optimizer, pth)
        val_f1s.append(val_f1)
        val_rocs.append(val_roc)
        val_accs.append(val_acc)
        roc_1_list.append(b1)
        roc_2_list.append(b2)
        f1_1_list.append(f1_1)
        f1_2_list.append(f1_2)
        acc_1_list.append(acc_1)
        acc_2_list.append(acc_2)

        if args.save_model:
            small_loader = True if args.mode != 1 else False 
            if not os.path.exists("./model/small_loader"):
                os.makedirs("./model/small_loader") 
            if not os.path.exists("./model/big_loader"):
                os.makedirs("./model/big_loader") 
            if small_loader:
                save_file_name = f"{args.model}_run{_+1}_small{round(b1, 2)}_Big{round(b2, 2)}.pth"
                torch.save(model.state_dict(), f"./model/small_loader/{save_file_name}")
            else:
                save_file_name = f"{args.model}_run{_+1}_small{round(b2, 2)}_Big{round(b1, 2)}"
                torch.save(model.state_dict(), f"./model/big_loader/{save_file_name}")

    print("Roc results for 5 runs: t1_roc:{:.4f}, t1_std:{:.4f}, t2_roc:{:.4f}, t2_std:{:.4f}".format(np.mean(roc_1_list), np.std(roc_1_list), np.mean(roc_2_list), np.std(roc_2_list)))
    print("F1 results for 5 runs: t1_f1:{:.4f}, t1_std:{:.4f}, t2_f1:{:.4f}, t2_std:{:.4f}".format(np.mean(f1_1_list), np.std(f1_1_list), np.mean(f1_2_list), np.std(f1_2_list)))
    print("Accuracy results for 5 runs: t1_acc:{:.4f}, t1_std:{:.4f}, t2_acc:{:.4f}, t2_std:{:.4f}".format(np.mean(acc_1_list), np.std(acc_1_list), np.mean(acc_2_list), np.std(acc_2_list)))
    print("Validation results: f1 avg:{:.4f}, f1 std:{:.4f}, roc avg:{:.4f}, roc std:{:.4f}, acc avg:{:.4f}, acc std:{:.4f}".format(np.mean(val_f1s), np.std(val_f1s), np.mean(val_rocs), np.std(val_rocs), np.mean(val_accs), np.std(val_accs)))

def print_stats(statistics):
    statistics = np.array(statistics)
    q1 = np.percentile(statistics, 25)
    q2 = np.percentile(statistics, 50) 
    q3 = np.percentile(statistics, 75)
    max_val = np.max(statistics)
    min_val = np.min(statistics)

    print("Q1:", q1)
    print("Q2 (median):", q2)
    print("Q3:", q3)
    print("Max:", max_val)
    print("Min:", min_val)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default="bbbp", choices=["toxcast", "bbbp", "bace", "NCI1", "NCI109","PROTEINS"], help='real or synthetic dataset')
    parser.add_argument('--train_size', default=2000, type=int, help='size of training data (synthetic)')
    parser.add_argument('--val_size', default=100, type=int, help='size of validatio data (synthetic)')
    parser.add_argument('--test_size', default=500, type=int, help='size of test data (synthetic)')
    parser.add_argument('--train_path', default="dataset/WS_4_4/", type=str, help='path for training dataset')
    parser.add_argument('--test_path', default="dataset/WS_4_4_large/", type=str, help='path for test dataset')
    parser.add_argument('--batch_size', default=30, type=int, help='batch_size')
    parser.add_argument('--patience', default=50, type=int, help='patience for early stopping')
    parser.add_argument('--worst_case', default=200, type=int, help='worst case for early stopping')
    parser.add_argument('--model', default="MlpNet", choices=["APPNPNet", "GatNet",  "ChebNet",  "GcnNet",  "GinNet",  "MlpNet",  "FANet",  "GNNML3", "GraphSAGE"], help='select model')
    parser.add_argument('--mode', default=0, type=int, help='0: small_to_large; 1: large_to_small;')
    parser.add_argument('--aggregator', default="attention", help='average_pool; max_pool; top-k; SAG; ASAP')
    parser.add_argument('--alpha', default=1.0, type=float, help='slope coefficient')
    parser.add_argument('--gamma', default=0.7, type=float, help='exponent')
    parser.add_argument('--num_runs', default=5, type=int, help='number of runs')
    parser.add_argument("--not_remove_val_test", default=False, action="store_true")
    parser.add_argument("--use_edge_betweeness", default=False, action="store_true")
    
    parser.add_argument('--graph_type', type=str, default="deg", help="statistics")
    parser.add_argument('--connec_measure', type=str, default="closeness_centrality")

    # if doing egonet drawing 
    parser.add_argument("--egonet_measure", type=str, default="closeness_centrality")
    parser.add_argument("--ego_rad", type=int, default=1)

    # more egonet command
    parser.add_argument("--ego_sub_percent", type=float, default=None)
    parser.add_argument("--ego_sub_num", type=int, default=None)
    parser.add_argument("--ego_top_k", type=int, default=1)
    parser.add_argument("--baseline_threshold", type=float, default=0.1)
    parser.add_argument("--remove_edge", default=False, action="store_true")

    # Save mode Related to Saliency
    parser.add_argument("--save_model", default=False, action="store_true")

    parser.add_argument("--saliency_mode", default=False, action="store_true")
    parser.add_argument("--statistics_mode", default=False, action="store_true", help="not training, only look at the distribution graphs")
    parser.add_argument("--remove_edge_score_measure", type=str, default="adamic_adar_index")


    # Commands for local closeness 
    parser.add_argument("--numHops", default=3, type=int, help="Number of hops for closeness statiscs")
    # Commands for Attention on centrality Nodes
    parser.add_argument("--attention_centrality_mode", default=False, action="store_true")
    parser.add_argument("--cutLow_centrality_ratio", default=0.5, type=float, help="threshold of dividing the embeddings")
    parser.add_argument("--centrality_median_mode", default=False, action="store_true")
    parser.add_argument("--centrality_25_quantile_mode", default=False, action="store_true")
    parser.add_argument("--centrality_fixed_threshold_mode", default=False, action="store_true")
    parser.add_argument("--extreme_centrality_mode", default=False, action="store_true")
    parser.add_argument("--centrality_low_value", type=float, default=0.45) 
    parser.add_argument("--centrality_high_value", type=float, default=0.50)
    parser.add_argument("--size_power", type=float, default=1.0)

    # Closeness features:
    parser.add_argument("--closeness_engieering", default=False, action="store_true")
    parser.add_argument("--use_entire_features", default=False, action="store_true")
    parser.add_argument("--use_one_hop_features", default=False, action="store_true")
    parser.add_argument("--attention_num_feature", default=1, type=int)
    
    parser.add_argument("--gat_numhead", default=8, type=int)
    parser.add_argument("--gat_hiddensize", default=128, type=int)
    parser.add_argument("--gat_outchannel", default=16, type=int)
    
    parser.add_argument("--approximate_graph", default=False, action="store_true")
    parser.add_argument("--approximate_p", default=1, type=float)
    
    args = parser.parse_args() 
    
    if args.use_entire_features or args.use_one_hop_features:
        args.attention_num_feature = 5
    print(args)

    pre_dataset, sorted_index, train_loader, val_loader, test_loader_1, test_loader, x_s, y_s = prepare_dataset(args)
    
    if args.statistics_mode:
        get_statistics(args, pre_dataset, sorted_index, graph_type=args.graph_type, 
                   connec_measure=args.connec_measure, egonet_measure=args.egonet_measure)
    else:
        # Training 
        train_function(args, train_loader, val_loader, test_loader_1, test_loader, x_s, y_s, pre_dataset=pre_dataset)

    
