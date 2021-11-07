python main_continual.py --strategy "Replay" \
      --data_dir /netscratch/mschreyer/deepNadim/100_datasets/philadelphia/original_data/city_payments_fy2017_encoded_all_new.csv \
      --no_epochs 100 --params_path "./params/params_10exp_scenario3.yml" --wandb_proj "ContinualAnomaly_REPLAY" \
      --bottleneck "tanh" --replay_mem_size 100


python main_continual.py --strategy "Replay" \
      --data_dir /netscratch/mschreyer/deepNadim/100_datasets/philadelphia/original_data/city_payments_fy2017_encoded_all_new.csv \
      --no_epochs 100 --params_path "./params/params_10exp_scenario3.yml" --wandb_proj "ContinualAnomaly_REPLAY" \
      --bottleneck "tanh" --replay_mem_size 50


python main_continual.py --strategy "Replay" \
      --data_dir /netscratch/mschreyer/deepNadim/100_datasets/philadelphia/original_data/city_payments_fy2017_encoded_all_new.csv \
      --no_epochs 100 --params_path "./params/params_10exp_scenario3.yml" --wandb_proj "ContinualAnomaly_REPLAY" \
      --bottleneck "tanh" --replay_mem_size 200


python main_continual.py --strategy "Replay" \
      --data_dir /netscratch/mschreyer/deepNadim/100_datasets/philadelphia/original_data/city_payments_fy2017_encoded_all_new.csv \
      --no_epochs 100 --params_path "./params/params_10exp_scenario3.yml" --wandb_proj "ContinualAnomaly_REPLAY" \
      --bottleneck "tanh" --replay_mem_size 1000
