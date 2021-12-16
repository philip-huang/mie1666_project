#!/bin/zsh

rm -rf results

size=( 11 21 31 51 101 )

for s in "${size[@]}"; do
    echo $s
    python eval.py data/mlp${s}_s0_test.pkl --model pretrained/mlp_${s}/best.pt --decode_strategy greedy --no_cuda
    python optimality_gap.py --rl_results_path results/mlp/mlp${s}_s0_test/mlp${s}_s0_test-mlp_${s}_best-greedy-t1-0-500.pkl --txt_path ../data/test-optimal/S0/
done

cd ../GILS_RVND
./run.sh