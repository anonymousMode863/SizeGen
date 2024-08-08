
This repository contains the code for KDD 2025 submission.

# Run Experiments

To install the dependencies in a new conda environment, run 

```
conda env create -f environment.yml
```

# How to run SIA 

Example command:

```
python main.py --model GcnNet --dataset bbbp --aggregator self_max --closeness_engieering --numHops 3 --use_one_hop_features
```

Example command of reproducing the plots in the paper, 

```
python main.py --statistics_mode --graph_type eigen
```

Here `graph_type` specifies the interest of plots, such as eigenvalue distribution difference.