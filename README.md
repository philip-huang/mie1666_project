# mie1666_project

Solve the minimum latency problem (MLP) with graph neural networks + RL

## Prepare data
Run `bash prep_data.sh` to download the tsplib data

Install the tsplib95 package via `pip install tsplib95`. The `tsplib.py` has helper function to load the data as *tsplib* problem instances and *networkx* graphs.

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

