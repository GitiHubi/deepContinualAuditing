for seed in {0..10}
do
  echo "Running for seed $seed"
  python main.py --strategy "Naive" --dataset "chicago" \
        --data_dir "./Data/city_payments_encoded_all_new_anomalies.csv" \
        --no_epochs 100 --benchmark_configs_path "./BenchmarkConfigs/instant_target-12.yml" \
        --wandb_proj "DeepContinualAuditing" \
        --bottleneck "tanh" --seed $seed --training_regime 'continual'
done


for seed in {0..10}
do
  echo "Running for seed $seed"
  python main.py --strategy "Replay" --dataset "chicago" \
        --data_dir "./Data/city_payments_encoded_all_new_anomalies.csv" \
        --no_epochs 100 --benchmark_configs_path "./BenchmarkConfigs/instant_target-12.yml" \
        --wandb_proj "DeepContinualAuditing" \
        --bottleneck "tanh" --replay_mem_size 500 --seed $seed --training_regime 'continual'

done


for seed in {0..10}
do
  echo "Running for seed $seed"
  python main.py --strategy "EWC" --dataset "chicago" \
        --data_dir "./Data/city_payments_encoded_all_new_anomalies.csv" \
        --no_epochs 100 --benchmark_configs_path "./BenchmarkConfigs/instant_target-12.yml" \
        --wandb_proj "DeepContinualAuditing" \
        --bottleneck "tanh" --ewc_lambda 10.0 --seed $seed --training_regime 'continual'
done

