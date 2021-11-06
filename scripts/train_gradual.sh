python main_continual.py --strategy "Naive" \
      --data_dir /netscratch/mschreyer/deepNadim/100_datasets/philadelphia/original_data/city_payments_fy2017_encoded_all_new.csv \
      --no_epochs 50 --params_path "./params/params_10exp_gradual.yml" --n_exp 310 --wandb_proj "ContinualAnomaly" \
      --bottleneck "tanh"

python main_continual.py --strategy "Replay" \
      --data_dir /netscratch/mschreyer/deepNadim/100_datasets/philadelphia/original_data/city_payments_fy2017_encoded_all_new.csv \
      --no_epochs 50 --params_path "./params/params_10exp_gradual.yml" --n_exp 10 --wandb_proj "ContinualAnomaly" \
      --bottleneck "tanh" --replay_mem_size 500

python main_continual.py --strategy "EWC" \
      --data_dir /netscratch/mschreyer/deepNadim/100_datasets/philadelphia/original_data/city_payments_fy2017_encoded_all_new.csv \
      --no_epochs 50 --params_path "./params/params_10exp_gradual.yml" --n_exp 10 --wandb_proj "ContinualAnomaly" \
      --bottleneck "tanh" --ewc_lambda 500.0

python main_continual.py --strategy "LwF" \
      --data_dir /netscratch/mschreyer/deepNadim/100_datasets/philadelphia/original_data/city_payments_fy2017_encoded_all_new.csv \
      --no_epochs 50 --params_path "./params/params_10exp_gradual.yml" --n_exp 10 --wandb_proj "ContinualAnomaly" \
      --bottleneck "tanh" --lwf_alpha 1.0 --lwf_temperature 1.0

python main_continual.py --strategy "SynapticIntelligence" \
      --data_dir /netscratch/mschreyer/deepNadim/100_datasets/philadelphia/original_data/city_payments_fy2017_encoded_all_new.csv \
      --no_epochs 50 --params_path "./params/params_10exp_gradual.yml" --n_exp 10 --wandb_proj "ContinualAnomaly" \
      --bottleneck "tanh" --si_lambda 1.0 --si_eps 0.001

python main_continual.py --strategy "JointTraining" \
      --data_dir /netscratch/mschreyer/deepNadim/100_datasets/philadelphia/original_data/city_payments_fy2017_encoded_all_new.csv \
      --no_epochs 50 --params_path "./params/params_10exp_gradual.yml" --n_exp 10 --wandb_proj "ContinualAnomaly" \
      --bottleneck "tanh" --si_lambda 1.0 --si_eps 0.001

