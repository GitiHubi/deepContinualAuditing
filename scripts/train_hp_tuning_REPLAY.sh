python main_continual.py --strategy "Replay" \
      --data_dir /netscratch/mschreyer/deepNadim/100_datasets/philadelphia/original_data/city_payments_fy2017_encoded_all_new_anomalies.csv \
      --no_epochs 100 --params_path "./params/params_10exp_scenario3.yml"  --wandb_proj "ContinualAnomaly_REPLAY" \
      --bottleneck "tanh" --replay_mem_size 100 --seed $seed


python main_continual.py --strategy "Replay" \
      --data_dir /netscratch/mschreyer/deepNadim/100_datasets/philadelphia/original_data/city_payments_fy2017_encoded_all_new_anomalies.csv \
      --no_epochs 100 --params_path "./params/params_10exp_scenario3.yml"  --wandb_proj "ContinualAnomaly_REPLAY" \
      --bottleneck "tanh" --replay_mem_size 50 --seed $seed


python main_continual.py --strategy "Replay" \
      --data_dir /netscratch/mschreyer/deepNadim/100_datasets/philadelphia/original_data/city_payments_fy2017_encoded_all_new_anomalies.csv \
      --no_epochs 100 --params_path "./params/params_10exp_scenario3.yml"  --wandb_proj "ContinualAnomaly_REPLAY" \
      --bottleneck "tanh" --replay_mem_size 1000 --seed $seed


python main_continual.py --strategy "Replay" \
      --data_dir /netscratch/mschreyer/deepNadim/100_datasets/philadelphia/original_data/city_payments_fy2017_encoded_all_new_anomalies.csv \
      --no_epochs 100 --params_path "./params/params_10exp_scenario3.yml"  --wandb_proj "ContinualAnomaly_REPLAY" \
      --bottleneck "tanh" --replay_mem_size 200 --seed $seed


python main_continual.py --strategy "Replay" \
      --data_dir /netscratch/mschreyer/deepNadim/100_datasets/philadelphia/original_data/city_payments_fy2017_encoded_all_new_anomalies.csv \
      --no_epochs 100 --params_path "./params/params_10exp_scenario3.yml"  --wandb_proj "ContinualAnomaly_REPLAY" \
      --bottleneck "tanh" --replay_mem_size 5000 --seed $seed
