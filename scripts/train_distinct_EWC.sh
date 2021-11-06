
python main_continual.py --strategy "EWC" \
      --data_dir /netscratch/mschreyer/deepNadim/100_datasets/philadelphia/original_data/city_payments_fy2017_encoded_all_new.csv \
      --no_epochs 50 --params_path "./params/params_10exp_distinct.yml" --n_exp 10 --wandb_proj "ContinualAnomaly_EWC" \
      --bottleneck "tanh" --ewc_lambda 5.0

python main_continual.py --strategy "EWC" \
      --data_dir /netscratch/mschreyer/deepNadim/100_datasets/philadelphia/original_data/city_payments_fy2017_encoded_all_new.csv \
      --no_epochs 50 --params_path "./params/params_10exp_distinct.yml" --n_exp 10 --wandb_proj "ContinualAnomaly_EWC" \
      --bottleneck "tanh" --ewc_lambda 20.0

python main_continual.py --strategy "EWC" \
      --data_dir /netscratch/mschreyer/deepNadim/100_datasets/philadelphia/original_data/city_payments_fy2017_encoded_all_new.csv \
      --no_epochs 50 --params_path "./params/params_10exp_distinct.yml" --n_exp 10 --wandb_proj "ContinualAnomaly_EWC" \
      --bottleneck "tanh" --ewc_lambda 100.0

python main_continual.py --strategy "EWC" \
      --data_dir /netscratch/mschreyer/deepNadim/100_datasets/philadelphia/original_data/city_payments_fy2017_encoded_all_new.csv \
      --no_epochs 50 --params_path "./params/params_10exp_distinct.yml" --n_exp 10 --wandb_proj "ContinualAnomaly_EWC" \
      --bottleneck "tanh" --ewc_lambda 500.0

