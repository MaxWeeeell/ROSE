finetune=all
model_type='mfm+register'
context_points=512
target_points=96
patch_len=64
stride=64
dset='weather'

# random_seed=2021

for target_points in 96 192 336 720
do
    for dset in 'etth1' 'etth2' 'ettm1' 'ettm2'
    # for dset in 'traffic'
    do
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

        python -u ROSE_zero-shot.py \
        --dset_finetune $dset \
        --context_points $context_points \
        --target_points $target_points \
        --batch_size 64 \
        --num_workers 0\
        --scaler standard \
        --features M\
        --patch_len $patch_len\
        --stride $stride\
        --revin 1 \
        --model_type $model_type >logs/LongForecasting/$model_type/$dset/$model_type'_zeroshots_'$dset'_tw'$target_points.log 
    done
done