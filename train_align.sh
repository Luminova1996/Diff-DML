dataset='cifar100'
dataset_part=0
model_type_A="resnet50"
model_type_B="resnet18"
baseline_acc=77.87
num_B=1

align_weight=0.9
optimizer='SGD'
lr_B=0.01

seed=4141
id=align-B${num_B}-${model_type_A}-${model_type_B}-w${align_weight}-seed-${seed}

log_path=log/log-${dataset}

if [ ! -d "$log_path/$id"  ]; then
    mkdir $log_path/$id
fi

CUDA_VISIBLE_DEVICES=3 python3 train.py --checkpoint_path=$log_path/$id --lr_A=0.1 --lr_B=${lr_B} --seed=$seed \
--dataset=${dataset} \
--dataset_part=${dataset_part} \
--optimizer=${optimizer} \
--batch_size=100 \
--align_flag=1 \
--smoothing_eps=0.0 \
--fix_A=0 --train_B=0.0 \
--align_weight=${align_weight} \
--align_weight_B=${align_weight} \
--focal=0.0 \
--model_type_A=${model_type_A} \
--model_type_B=${model_type_B} \
--num_B=${num_B} \
--lr_B_decay=0 \
--baseline_acc=${baseline_acc} \
--fudge=0 

