dataset='cifar100'
dataset_part=0
model_type_A="resnet50"
pretrained_A=0
use_TS=0
id=test-${dataset}-${model_type_A}-part-${dataset_part}-ensemble-20250130
seed=212
#log_path=/data/liuhan/toy/modellog/log-MD-IMDB
log_path=log/log-${dataset}-ensemble
if [ ! -d "$log_path/$id"  ]; then
    mkdir $log_path/$id
fi

CUDA_VISIBLE_DEVICES=4 python3 test_ensemble.py --checkpoint_path=$log_path/$id  --seed=$seed \
--cutout=0 \
--batch_size=20 \
--dataset=${dataset} \
--dataset_part=${dataset_part} \
--model_type_A=${model_type_A} \
--pretrained_A=${pretrained_A} \
--use_TS=${use_TS}
