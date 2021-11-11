# mie1666_project

Solve the minimum latency problem (MLP) with graph neural networks + RL

## Prepare data
Run `bash prep_data.sh` to download the tsplib data

Install the tsplib95 package via `pip install tsplib95`. The `tsplib.py` has helper function to load the data as *tsplib* problem instances and *networkx* graphs.

### Training with RL + GNN 

For training MLD instances with 5 nodes and using rollout as REINFORCE baseline:
```bash
cd attention_learn_to_route
python run.py --graph_size 5 --baseline rollout --run_name 'mlp_5_rollout' --problem 'mlp'
```
