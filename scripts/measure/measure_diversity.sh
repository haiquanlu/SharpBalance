dataset=
data_path=
model=
resumes=
save_path=

python measure_diversity.py \
            --dataset $dataset \
            --data_path $data_path \
            --model $model \
            --resume $resumes \
            --mode predict,disagreement \
            --save $save_path