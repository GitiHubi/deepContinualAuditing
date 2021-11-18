for seed in {0..4}
do
  echo "Running for seed $seed"
  python main_continual.py --strategy "Naive" --dataset "chicago" \
        --data_dir /netscratch/mschreyer/deepNadim/100_datasets/chicago/original_data/city_payments_encoded_all_new_anomalies.csv \
        --no_epochs 100 --benchmark_configs_path "./benchmark_configs/exponential.yml" --wandb_proj "deepNadim" \
        --bottleneck "tanh" --seed $seed --training_regime 'fromscratch'
done

