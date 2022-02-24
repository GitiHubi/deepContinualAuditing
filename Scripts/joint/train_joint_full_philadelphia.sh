for seed in {0..4}
do
  echo "Running for seed $seed"
  python main.py --strategy "Naive" --dataset "philadelphia" \
        --data_dir "./Data/city_payments_fy2017_encoded_all_new_anomalies.csv" \
        --no_epochs 100 --benchmark_configs_path "./BenchmarkConfigs/full.yml" \
        --wandb_proj "DeepContinualAuditing" \
        --bottleneck "tanh" --seed $seed --training_regime 'joint'
done

