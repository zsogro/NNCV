wandb login

python3 train.py \
    --data-dir ./data/cityscapes \
    --batch-size 64 \
    --epochs 50 \
    --lr 0.001 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "L16_mlp3_Multiplicative_09_startEpoch_10" \