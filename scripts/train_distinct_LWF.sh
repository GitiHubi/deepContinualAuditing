python main_continual.py --strategy "LwF" \
      --data_dir /netscratch/mschreyer/deepNadim/100_datasets/philadelphia/original_data/city_payments_fy2017_encoded_all_new.csv \
      --no_epochs 50 --params_path "./params/params_10exp_distinct.yml" --n_exp 10 --wandb_proj "ContinualAnomaly_LWF" \
      --bottleneck "tanh" --lwf_alpha 0.1 --lwf_temperature 1.0

python main_continual.py --strategy "LwF" \
      --data_dir /netscratch/mschreyer/deepNadim/100_datasets/philadelphia/original_data/city_payments_fy2017_encoded_all_new.csv \
      --no_epochs 50 --params_path "./params/params_10exp_distinct.yml" --n_exp 10 --wandb_proj "ContinualAnomaly_LWF" \
      --bottleneck "tanh" --lwf_alpha 0.01 --lwf_temperature 1.0


python main_continual.py --strategy "LwF" \
      --data_dir /netscratch/mschreyer/deepNadim/100_datasets/philadelphia/original_data/city_payments_fy2017_encoded_all_new.csv \
      --no_epochs 50 --params_path "./params/params_10exp_distinct.yml" --n_exp 10 --wandb_proj "ContinualAnomaly_LWF" \
      --bottleneck "tanh" --lwf_alpha 1.0 --lwf_temperature 2.0


python main_continual.py --strategy "LwF" \
      --data_dir /netscratch/mschreyer/deepNadim/100_datasets/philadelphia/original_data/city_payments_fy2017_encoded_all_new.csv \
      --no_epochs 50 --params_path "./params/params_10exp_distinct.yml" --n_exp 10 --wandb_proj "ContinualAnomaly_LWF" \
      --bottleneck "tanh" --lwf_alpha 0.01 --lwf_temperature 2.0


python main_continual.py --strategy "LwF" \
      --data_dir /netscratch/mschreyer/deepNadim/100_datasets/philadelphia/original_data/city_payments_fy2017_encoded_all_new.csv \
      --no_epochs 50 --params_path "./params/params_10exp_distinct.yml" --n_exp 10 --wandb_proj "ContinualAnomaly_LWF" \
      --bottleneck "tanh" --lwf_alpha 0.1 --lwf_temperature 2.0


python main_continual.py --strategy "LwF" \
      --data_dir /netscratch/mschreyer/deepNadim/100_datasets/philadelphia/original_data/city_payments_fy2017_encoded_all_new.csv \
      --no_epochs 50 --params_path "./params/params_10exp_distinct.yml" --n_exp 10 --wandb_proj "ContinualAnomaly_LWF" \
      --bottleneck "tanh" --lwf_alpha 0.1 --lwf_temperature 0.1

