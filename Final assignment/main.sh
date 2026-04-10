wandb login

python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 64 \
    --epochs 50 \
    --lr 0.001 \
    --precision auto \
    --backbone-train-last-n 1 \
    --backbone-lr 1e-5 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "L16_All_MLP8_res1024_unfreeze_1" \