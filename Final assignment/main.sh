wandb login

python3 train_ood.py \
    --data-dir ./data/cityscapes \
    --batch-size 64 \
    --epochs 100 \
    --lr 0.001 \
    --num-workers 10 \
    --seed 42 \
    --experiment-id "OOD-Detector-v1-retraining-100" \
    # --experiment-id "L16_AllMLP8_res1024_scheduler_after_15_epochs" \