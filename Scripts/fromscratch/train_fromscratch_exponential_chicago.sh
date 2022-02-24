for seed in {0..10}
do
  echo "Running for seed $seed"
  python main.py --strategy "Naive" --dataset "chicago" \
        --data_dir "./Data/city_payments_encoded_all_new_anomalies.csv" \
        --no_epochs 100 --benchmark_configs_path "./BenchmarkConfigs/exponential_target-12.yml" \
        --wandb_proj "DeepContinualAuditing" \
        --bottleneck "tanh" --seed $seed --training_regime 'fromscratch'
done

