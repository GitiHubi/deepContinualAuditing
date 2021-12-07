for seed in {0..10}
do
  echo "Running for seed $seed"
  python main.py --strategy "Naive" --dataset "chicago" \
        --data_dir /netscratch/mschreyer/deepNadim/100_datasets/chicago/original_data/city_payments_encoded_all_new_anomalies.csv \
        --no_epochs 100 --benchmark_configs_path "./benchmark_configs/linear_target-12.yml" --wandb_proj "deepNadim" --wandb_entity "aiml_cl" \
        --bottleneck "tanh" --seed $seed --training_regime 'continual'
done


for seed in {0..10}
do
  echo "Running for seed $seed"
  python main.py --strategy "Replay" --dataset "chicago" \
        --data_dir /netscratch/mschreyer/deepNadim/100_datasets/chicago/original_data/city_payments_encoded_all_new_anomalies.csv \
        --no_epochs 100 --benchmark_configs_path "./benchmark_configs/linear_target-12.yml"  --wandb_proj "deepNadim" --wandb_entity "aiml_cl" \
        --bottleneck "tanh" --replay_mem_size 500 --seed $seed --training_regime 'continual'
done


for seed in {0..10}
do
  echo "Running for seed $seed"
  python main.py --strategy "EWC" --dataset "chicago" \
        --data_dir /netscratch/mschreyer/deepNadim/100_datasets/chicago/original_data/city_payments_encoded_all_new_anomalies.csv \
        --no_epochs 100 --benchmark_configs_path "./benchmark_configs/linear_target-12.yml"  --wandb_proj "deepNadim" --wandb_entity "aiml_cl" \
        --bottleneck "tanh" --ewc_lambda 10.0 --seed $seed --training_regime 'continual'
done

