method=deep_ensemble
data=cifar100
model=resnet18
data_path=
save_path=

seed="13 17 27"


python main.py \
    --data $data \
    --data_path ${data_path} \
    --model ${model} \
    --seed ${seed} \
    --save_path ${save_path}