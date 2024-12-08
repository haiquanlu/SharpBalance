dataset=
data_path=
model=
resumes=
save_path=

python measure_sharpness.py \
        --dataset $dataset \
        --model $model \
        --resume $resumes \
        --data_path $data_path \
        --mode APGD_worst \
        --save $save_path