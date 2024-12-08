method=sharpbalance
data=cifar100
model=resnet18
data_path=

seed="13 17 27"
rho=0.2
flat_ratio=0.5
initial_epochs=100
save_path=

python main.py \
    --data $data \
    --data_path ${data_path} \
    --model ${model} \
    --seed ${seed} \
    --save_path ${save_path} \
    --sam True \
    --rho ${rho} \
    --flat_ratio ${flat_ratio} \
    --initial_epochs ${initial_epochs}
