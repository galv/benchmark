#!/bin/bash

set -euo pipefail

workdir=trees_vs_params42_release_with_debug_info_dropout_is_all_0

export TORCH_LOGS_FORMAT="%(message)s" 
export TORCH_LOGS=graph_code,graph_breaks,aot_joint_graph,output_code

if [ -z "${HUGGING_FACE_HUB_TOKEN}" ]; then
  echo "Error: HUGGING_FACE_HUB_TOKEN is not set." >&2
  exit 1
fi

# for metric in "accuracy" "performance"; do
for metric in "accuracy"; do
# for method in "inference" "training"; do
for method in "training" "inference"; do
    # for dataset in "huggingface" "torchbench"; do
    for dataset in "huggingface" "torchbench"; do
        # for metric in "accuracy" "performance"; do
        # for metric in "accuracy"; do
            mkdir -p $workdir/${dataset}/${method}/${metric}
            echo "GALVEZ:2"
            time python run_benchmark.py dynamo --speedup-parameterized-cudagraphs-basic --output-directory $workdir/${dataset}/${method}/${metric} --${metric} --${method} --${dataset} 2>&1 | tee $workdir/${dataset}_params_${method}_${metric}.log
        # done
    done
done
done

for metric in "accuracy"; do
# for method in "inference" "training"; do
    for method in "training" "inference"; do
        # for dataset in "huggingface" "torchbench"; do
        for dataset in "huggingface" "torchbench"; do
            echo "GALVEZ:1"
            time python run_benchmark.py dynamo --inductor --output-directory $workdir/${dataset}/${method}/${metric} --${method} --${metric} --${dataset} 2>&1 | tee $workdir/${dataset}_trees_${method}_${metric}.log
        done
    done
done
