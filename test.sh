dataset='cifar100'
dataset_part=0
model_type_A="resnet50"
pretrained_A=0

model_A_load_from="log/log-cifar100/align-B1-SGD-part-0-resnet50-resnet18-lr0.1-w0.9-fix_A-0-train_B-0.0-decay-1-fudge0-seed-4141/model_A.overall.th"
scaling_method="None"

id=align-B1-SGD-part-0-resnet50-resnet18-lr0.1-w0.9-fix_A-0-train_B-0.0-decay-1-fudge0-seed-4141
seed=212
fudge=0

log_path=log/log-${dataset}
if [ ! -d "$log_path/$id"  ]; then
    mkdir $log_path/$id
fi

CUDA_VISIBLE_DEVICES=7 python3 test.py --checkpoint_path=$log_path/$id --model_A_load_from=${model_A_load_from} --seed=$seed \
--cutout=0 \
--batch_size=20 \
--dataset=${dataset} \
--dataset_part=${dataset_part} \
--model_type_A=${model_type_A} \
--pretrained_A=${pretrained_A} \
--scaling_method=${scaling_method} \
--fudge=${fudge} 
