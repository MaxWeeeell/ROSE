is_all=1
is_linear_probe=0
model_type=mfm+register
dset=etth2


if [ ! -d "logs" ]; then
mkdir logs
fi
if [ ! -d "logs/LongForecasting" ]; then
    mkdir logs/LongForecasting
fi
if [ ! -d "logs/LongForecasting/$model_type" ]; then
    mkdir logs/LongForecasting/$model_type
fi
if [ ! -d "logs/LongForecasting/$model_type/$dset" ]; then
    mkdir logs/LongForecasting/$model_type/$dset
fi


python -u ROSE_finetune.py \
    --is_finetune 1 \
    --is_linear_probe 0 \
    --dset_finetune 'etth2'\
    --context_points 512 \
    --target_points 96 \
    --batch_size 64 \
    --num_workers 0\
    --scaler standard \
    --features M\
    --patch_len 64\
    --stride 64\
    --revin 1 \
    --n_epochs_finetune 20\
    --n_epochs_freeze 20\
    --lr 5e-4 \
    --pretrained_model full-shot\
    --finetuned_model_id 1\
    --model_type $model_type\
    --finetune_percentage 1\
    --seed 2022 >logs/LongForecasting/$model_type/$dset/$model_type'_'$dset'_tw96'.log 

python -u ROSE_finetune.py \
    --is_finetune 1 \
    --is_linear_probe 0 \
    --dset_finetune 'etth2'\
    --context_points 512 \
    --target_points 192 \
    --batch_size 64 \
    --num_workers 0\
    --scaler standard \
    --features M\
    --patch_len 64\
    --stride 64\
    --revin 1 \
    --n_epochs_finetune 20\
    --n_epochs_freeze 20\
    --lr 5e-4 \
    --pretrained_model full-shot\
    --finetuned_model_id 1\
    --model_type $model_type\
    --finetune_percentage 1\
    --seed 2024 >logs/LongForecasting/$model_type/$dset/$model_type'_'$dset'_tw192'.log 

python -u ROSE_finetune.py \
    --is_finetune 1 \
    --is_linear_probe 0 \
    --dset_finetune 'etth2'\
    --context_points 512 \
    --target_points 336 \
    --batch_size 64 \
    --num_workers 0\
    --scaler standard \
    --features M\
    --patch_len 64\
    --stride 64\
    --revin 1 \
    --n_epochs_finetune 100\
    --n_epochs_freeze 20\
    --lr 5e-4 \
    --pretrained_model full-shot\
    --finetuned_model_id 1\
    --model_type $model_type\
    --finetune_percentage 1\
    --seed 2025 >logs/LongForecasting/$model_type/$dset/$model_type'_'$dset'_tw336'.log 

python -u ROSE_finetune.py \
    --is_finetune 1 \
    --is_linear_probe 0 \
    --dset_finetune 'etth2'\
    --context_points 512 \
    --target_points 720 \
    --batch_size 64 \
    --num_workers 0\
    --scaler standard \
    --features M\
    --patch_len 64\
    --stride 64\
    --revin 1 \
    --n_epochs_finetune 30\
    --n_epochs_freeze 30\
    --lr 1e-4 \
    --pretrained_model full-shot\
    --finetuned_model_id 1\
    --model_type $model_type\
    --finetune_percentage 1\
    --seed 2020 >logs/LongForecasting/$model_type/$dset/$model_type'_'$dset'_tw720'.log 



