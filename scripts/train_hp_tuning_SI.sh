python main_continual.py --strategy "SynapticIntelligence" \
      --data_dir /netscratch/mschreyer/deepNadim/100_datasets/philadelphia/original_data/city_payments_fy2017_encoded_all_new.csv \
      --no_epochs 50 --params_path "./params/params_10exp_scenario3.yml" --wandb_proj "ContinualAnomaly_SI" \
      --bottleneck "tanh" --si_lambda 10.0 --si_eps 0.001

python main_continual.py --strategy "SynapticIntelligence" \
      --data_dir /netscratch/mschreyer/deepNadim/100_datasets/philadelphia/original_data/city_payments_fy2017_encoded_all_new.csv \
      --no_epochs 50 --params_path "./params/params_10exp_scenario3.yml" --wandb_proj "ContinualAnomaly_SI" \
      --bottleneck "tanh" --si_lambda 10.0 --si_eps 0.01
