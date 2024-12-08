dataset=
data_path=
model=
resumes=
save_path=

python measure_diversity_variance.py \
            --dataset $dataset \
            --data_path $data_path \
            --model $model \
            --resume $resumes \
            --mode variance_mse_c \
            --save $save_path