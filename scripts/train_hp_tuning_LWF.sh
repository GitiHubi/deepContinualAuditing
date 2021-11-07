python main_continual.py --strategy "LwF" \
      --data_dir /netscratch/mschreyer/deepNadim/100_datasets/philadelphia/original_data/city_payments_fy2017_encoded_all_new.csv \
      --no_epochs 100 --params_path "./params/params_10exp_scenario3.yml" --wandb_proj "ContinualAnomaly_LWF" \
      --bottleneck "tanh" --lwf_alpha 2.0 --lwf_temperature 1.0

python main_continual.py --strategy "LwF" \
      --data_dir /netscratch/mschreyer/deepNadim/100_datasets/philadelphia/original_data/city_payments_fy2017_encoded_all_new.csv \
      --no_epochs 100 --params_path "./params/params_10exp_scenario3.yml" --wandb_proj "ContinualAnomaly_LWF" \
      --bottleneck "tanh" --lwf_alpha 5.0 --lwf_temperature 1.0
