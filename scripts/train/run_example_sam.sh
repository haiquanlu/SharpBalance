method=deep_ensemble_sam
data=cifar100
model=resnet18
data_path=

seed="13 17 27"
initial_epochs=100
rho=0.2
save_path=

python main.py \
    --data $data \
    --data_path ${data_path} \
    --model ${model} \
    --seed ${seed} \
    --save_path ${save_path} \
    --initial_epochs $initial_epochs \
    --sam True \
    --rho ${rho}