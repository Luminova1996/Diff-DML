dataset='cifar100'
dataset_part=0
model_type_A="resnet34"
pretrained_A=0
baseline_acc=76.93

DTDP=1

cutout=0
smoothing_eps=0.0
focal=3.0
huber_gamma=10.0
huber_alpha=0.005
optimizer='SGD'
cutout=0

seed=5555
id=DTDP-${optimizer}-part-${dataset_part}-${model_type_A}-ptrain-${pretrained_A}-smoothing-${smoothing_eps}-focal-${focal}-huber-${huber_gamma}-${huber_alpha}-seed-${seed}
log_path=log/log-${dataset}-new

if [ ! -d "$log_path/$id"  ]; then
    mkdir $log_path/$id
fi

#CUDA_VISIBLE_DEVICES=1,2 python3 -m torch.distributed.launch --nproc_per_node=2 train.py --port=$port --checkpoint_path=log/$id 
CUDA_VISIBLE_DEVICES=8 python3 train.py --checkpoint_path=$log_path/$id --lr_A=0.1 --seed=$seed \
--dataset=${dataset} \
--dataset_part=${dataset_part} \
--cutout=${cutout} \
--optimizer=${optimizer} \
--batch_size=100 \
--align_flag=0 \
--smoothing_eps=${smoothing_eps} \
--focal=${focal} \
--DTDP=${DTDP}  \
--huber_gamma=${huber_gamma} \
--huber_alpha=${huber_alpha} \
--model_type_A=${model_type_A}  \
--baseline_acc=${baseline_acc} \
--pretrained_A=${pretrained_A} 
