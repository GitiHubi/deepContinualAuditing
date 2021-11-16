for seed in {0..4}
do
  echo "Running for seed $seed"
  python main_continual.py --strategy "Naive" \
        --data_dir /netscratch/mschreyer/deepNadim/100_datasets/chicago/original_data/city_payments_encoded_all_new_anomalies.csv \
        --no_epochs 100 --params_path "./params/params_10exp_scenario3.yml" --wandb_proj "deepNadim" \
        --bottleneck "tanh" --seed $seed --training_regime 'joint'
done

