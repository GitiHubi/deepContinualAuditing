python main_continual.py --strategy "EWC" \
      --data_dir /netscratch/mschreyer/deepNadim/100_datasets/philadelphia/original_data/city_payments_fy2017_encoded_all_new.csv \
      --no_epochs 100 --params_path "./params/params_10exp_scenario3.yml" --wandb_proj "ContinualAnomaly_EWC" \
      --bottleneck "tanh" --ewc_lambda 50000.0

python main_continual.py --strategy "EWC" \
      --data_dir /netscratch/mschreyer/deepNadim/100_datasets/philadelphia/original_data/city_payments_fy2017_encoded_all_new.csv \
      --no_epochs 100 --params_path "./params/params_10exp_scenario3.yml" --wandb_proj "ContinualAnomaly_EWC" \
      --bottleneck "tanh" --ewc_lambda 100000.0

python main_continual.py --strategy "EWC" \
      --data_dir /netscratch/mschreyer/deepNadim/100_datasets/philadelphia/original_data/city_payments_fy2017_encoded_all_new.csv \
      --no_epochs 100 --params_path "./params/params_10exp_scenario3.yml" --wandb_proj "ContinualAnomaly_EWC" \
      --bottleneck "tanh" --ewc_lambda 20000.0
