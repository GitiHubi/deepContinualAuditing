python main_continual.py --strategy "Replay" \
      --data_dir /netscratch/mschreyer/deepNadim/100_datasets/philadelphia/original_data/city_payments_fy2017_encoded_all.csv \
      --no_epochs 20 --params_path "./params/params_3exp_distinct.yml" --n_exp 3 --wandb_proj "ContinualAnomaly"


python main_continual.py --strategy "Replay" \
      --data_dir /netscratch/mschreyer/deepNadim/100_datasets/philadelphia/original_data/city_payments_fy2017_encoded_all.csv \
      --no_epochs 20 --params_path "./params/params_5exp_gradual.yml" --n_exp 5 --wandb_proj "ContinualAnomaly"

