# MUSE: Multi-Grained Self-Interpretable Graph Learning

## ABSTRACT
As graph neural networks (GNNs) have shown great performance in various graph learning tasks, GNN interpretation is also developing to illustrate interpretable patterns related to specific predictions. A prevalent way is to devise post-hoc explainers for trained blackbox GNNs. But post-hoc explainers suffer the faithfulness issue for the model to be explained. A more promising approach is to endow GNNs with built-in interpretability, which is attracting increasing interest. However, current self-interpretation mechanisms for GNNs either focus merely on independent instances or raise unsatisfactory interpretations for a higher class level, which limits the ability to discover important label-specific patterns. 

In this work, we propose MUSE, a multi-grained self-interpretable graph learning framework with both instance and class-level interpretability in a unified manner. Specifically, we improve the information bottleneck principle for graph-structure data to obtain high-quality instance-level interpretable subgraphs and utilize the maximum mean discrepancy critic to generate class-level prototypes based on learned subgraphs and original graph characteristics. We conduct experiments on eight datasets to evaluate both the prediction and interpretation performance. Moreover, we visualize the class-level prototype graphs and quantitatively evaluate their quality. Results reveal that MUSE could achieve both superior prediction and interpretation performance than state-of-the-art baselines and supply reasonable typical graph patterns on a high level. Code repo for the graph pattern learning project.

## Dataset
Datasets can be downloaded manully.

## Run and evaluate the model
In run.sh:
```bash
device='cuda:0'

dataset_list=(mutag mnist spmotif_0.3 spmotif_0.5 spmotif_0.7 spmotif_0.9)

TASK=evaluation # train, evaluation
debug=false # true, false. Debug mode
model=GIN
ckpt_file='' # used during evaluation

if [[ ${TASK} == "train" ]]; then
    for dataset in ${dataset_list[*]}; do
        exp_group_name_note='default_group'
        CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 python -m ipdb -c 'continue' \
        main.py \
        dataset=${dataset} \
        model=${model} \
        exp_group_name=train_${model}_${dataset}_${exp_group_name_note} \
        train=true \
        debug=${debug} \
        device=${device} \
        training.epochs=200 \
        exp_note=${exp_group_name_note} \
    done
elif [[ ${TASK} == "evaluation" ]]; then
    for dataset in ${dataset_list[*]}; do
        echo "[Task is evaluation. Dataset is ${dataset}]"
        CUDA_LAUNCH_BLOCKING=1 HYDRA_FULL_ERROR=1 python -m ipdb -c 'continue' \
        main.py \
        evaluation.ckpt_file=${ckpt_file}
        train=false
    done
fi


```