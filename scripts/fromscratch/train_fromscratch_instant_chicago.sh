for seed in {0..10}
do
  echo "Running for seed $seed"
  python main.py --strategy "Naive" --dataset "chicago" \
        --data_dir /netscratch/mschreyer/deepNadim/100_datasets/chicago/original_data/city_payments_encoded_all_new_anomalies.csv \
        --no_epochs 100 --benchmark_configs_path "./benchmark_configs/instant_target-12.yml" --wandb_proj "deepNadim" --wandb_entity "aiml_cl" \
        --bottleneck "tanh" --seed $seed --training_regime 'fromscratch'
done

