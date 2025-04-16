dataset='cifar100'
dataset_part=0
model_type_A="resnet50"
baseline_acc=77.87
#baseline_acc=77.54

smoothing_eps=0.0
focal=0.0
dual_focal=5.0
MDCA=0.0
FDCA=0.0
FDCA_M=10
MMCE=0.0
DCA=0.0
optimizer='SGD'
fudge=0

#model_A_load_from=/home/liuhan/data1/toy/modellog/log-cifar10/single-SGD-cutout-0-resnet34-ptrain-0-smoothing-0.0-focal-0.0-MDCA-0.0-MMCE-0.0-DCA-0.0/model1.best.th
#id=test
#seed=4141
seed=5555
id=single-${optimizer}-part-${dataset_part}-${model_type_A}-smoothing-${smoothing_eps}-focal-${focal}-dual-${dual_focal}-MDCA-${MDCA}-FDCA-${FDCA}-M-${FDCA_M}-MMCE-${MMCE}-DCA-${DCA}-fudge${fudge}-seed-${seed}

log_path=log/log-${dataset}

if [ ! -d "$log_path/$id"  ]; then
    mkdir $log_path/$id
fi

#CUDA_VISIBLE_DEVICES=1,2 python3 -m torch.distributed.launch --nproc_per_node=2 train.py --port=$port --checkpoint_path=log/$id 
CUDA_VISIBLE_DEVICES=6 python3 train.py --checkpoint_path=$log_path/$id --lr_A=0.1 --seed=$seed \
--dataset=${dataset} \
--dataset_part=${dataset_part} \
--optimizer=${optimizer} \
--batch_size=100 \
--align_flag=0 \
--smoothing_eps=${smoothing_eps} \
--dual_focal=${dual_focal} \
--focal=${focal} \
--MDCA_beta=${MDCA} \
--FDCA_beta=${FDCA} \
--FDCA_M=${FDCA_M} \
--DCA_beta=${DCA} \
--MMCE_lambda=${MMCE} \
--model_type_A=${model_type_A}  \
--baseline_acc=${baseline_acc} \
--fudge=${fudge} 

#--model_A_load_from=${model_A_load_from}
