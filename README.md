# Learning Heuristics for Minimum Latency Problem with RL and GNN

Solve the minimum latency problem (MLP) with graph neural networks + RL. Group work by 

[Mohamed Khodier](https://khodeir.github.io/)

[Salar Hosseini Khorasgani](https://salarios77.github.io/)

[Siqi Hao](https://rvl.cs.toronto.edu/team/)

[Philip Huang](https://philip-huang.github.io/)


### Training with RL + GNN 

For training MLP instances with 5 nodes and using rollout as REINFORCE baseline:
```bash
cd attention_learn_to_route
python run.py --graph_size 5 --baseline rollout --run_name 'mlp_5_rollout' --problem 'mlp'
```

To generate validation or test data for MLP (by default to 'data' directory and for graph sizes [20, 50, 100]):
```
python generate data.py --problem mlp --name test --seed 1234
```

To evaluate model on dataset and save results (by default to 'results' directory):
```
# Greedy decoding
python eval.py data/mlp/mlp20_test_seed1234.pkl --model outputs/mlp_20/mlp_20_rollout_20211112T120056/epoch-X.pt --decode_strategy greedy

#Sampling-based decoding with 1280 solutions sampled
python eval.py data/mlp/mlp20_test_seed1234.pkl --model outputs/mlp_20/mlp_20_rollout_20211112T120056/epoch-X.pt --decode_strategy sample --width 1280 --eval_batch_size 1
```

### Run GILS-RVND

To run gils-rvnd baseline, we have provided a compiled binary on Linux and a script to evaluate the results for all instances.
```
cd GILS-RVND
./run.sh
```

### Citations
This repo is based on the attention learning-to-route [repo](https://github.com/wouterkool/attention-learn-to-route).

[1]  W. Kool, H. van Hoof, and M. Welling, “Attention, learn to solve routing problems!” 2019

The GILS-RVND implementation is based on the mlp [repo](https://github.com/renatamendesc/MLP)


