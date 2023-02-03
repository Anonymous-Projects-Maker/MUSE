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
