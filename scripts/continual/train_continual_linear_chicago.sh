for seed in {0..4}
do
  echo "Running for seed $seed"
  python main.py --strategy "Naive" --dataset "chicago" \
        --data_dir /netscratch/mschreyer/deepNadim/100_datasets/chicago/original_data/city_payments_encoded_all_new_anomalies.csv \
        --no_epochs 100 --benchmark_configs_path "./benchmark_configs/linear.yml" --wandb_proj "deepNadim" \
        --bottleneck "tanh" --seed $seed --training_regime 'continual'
done


for seed in {0..4}
do
  echo "Running for seed $seed"
  python main.py --strategy "Replay" --dataset "chicago" \
        --data_dir /netscratch/mschreyer/deepNadim/100_datasets/chicago/original_data/city_payments_encoded_all_new_anomalies.csv \
        --no_epochs 100 --benchmark_configs_path "./benchmark_configs/linear.yml"  --wandb_proj "deepNadim" \
        --bottleneck "tanh" --replay_mem_size 500 --seed $seed --training_regime 'continual'
done


for seed in {0..4}
do
  echo "Running for seed $seed"
  python main.py --strategy "EWC" --dataset "chicago" \
        --data_dir /netscratch/mschreyer/deepNadim/100_datasets/chicago/original_data/city_payments_encoded_all_new_anomalies.csv \
        --no_epochs 100 --benchmark_configs_path "./benchmark_configs/linear.yml"  --wandb_proj "deepNadim" \
        --bottleneck "tanh" --ewc_lambda 50.0 --seed $seed --training_regime 'continual'
done
