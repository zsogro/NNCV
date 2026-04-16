wandb login

python3 train_ood.py \
    --data-dir ./data/cityscapes \
    --batch-size 64 \
    --epochs 50 \
    --lr 0.001 \
    --precision auto \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "CoupledQuadraticSpline_v1" \
    # --experiment-id "L16_AllMLP8_res1024_scheduler_after_15_epochs" \