# MUSE: Multi-Grained Self-Interpretable Graph Learning

## ABSTRACT
As graph neural networks (GNNs) have shown great performance in various graph learning tasks, GNN interpretation is also developing to illustrate interpretable patterns related to specific predictions. A prevalent way is to devise post-hoc explainers for trained blackbox GNNs. But post-hoc explainers suffer the faithfulness issue for the model to be explained. A more promising approach is to endow GNNs with built-in interpretability, which is attracting increasing interest. However, current self-interpretation mechanisms for GNNs either focus merely on independent instances or raise unsatisfactory interpretations for a higher class level, which limits the ability to discover important label-specific patterns. 

In this work, we propose MUSE, a multi-grained self-interpretable graph learning framework with both instance and class-level interpretability in a unified manner. Specifically, we improve the information bottleneck principle for graph-structure data to obtain high-quality instance-level interpretable subgraphs and utilize the maximum mean discrepancy critic to generate class-level prototypes based on learned subgraphs and original graph characteristics. We conduct experiments on eight datasets to evaluate both the prediction and interpretation performance. Moreover, we visualize the class-level prototype graphs and quantitatively evaluate their quality. Results reveal that MUSE could achieve both superior prediction and interpretation performance than state-of-the-art baselines and supply reasonable typical graph patterns on a high level. Code repo for the graph pattern learning project.

## Dataset
Datasets can be downloaded manully.

## Run
./run.sh

# Dataset info
ogbg-molhiv:
https://github.com/snap-stanford/ogb/blob/master/ogb/utils/features.py