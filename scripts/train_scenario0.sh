for seed in {0..4}
do
  echo "Running for seed $seed"
  python main_continual.py --strategy "Naive" \
        --data_dir /netscratch/mschreyer/deepNadim/100_datasets/philadelphia/original_data/city_payments_fy2017_encoded_all_new_anomalies.csv \
        --no_epochs 100 --params_path "./params/params_10exp_scenario0.yml" --wandb_proj "ContinualAnomaly_Scenario0" \
        --bottleneck "tanh" --seed $seed
done


for seed in {0..4}
do
  echo "Running for seed $seed"
  python main_continual.py --strategy "Replay" \
        --data_dir /netscratch/mschreyer/deepNadim/100_datasets/philadelphia/original_data/city_payments_fy2017_encoded_all_new_anomalies.csv \
        --no_epochs 100 --params_path "./params/params_10exp_scenario0.yml" --wandb_proj "ContinualAnomaly_Scenario0" \
        --bottleneck "tanh" --replay_mem_size 500 --seed $seed
done


#python main_continual.py --strategy "EWC" \
#      --data_dir /netscratch/mschreyer/deepNadim/100_datasets/philadelphia/original_data/city_payments_fy2017_encoded_all_new_anomalies.csv \
#      --no_epochs 100 --params_path "./params/params_10exp_scenario0.yml" --wandb_proj "ContinualAnomaly_Scenario0" \
#      --bottleneck "tanh" --ewc_lambda 50000.0
#
#python main_continual.py --strategy "LwF" \
#      --data_dir /netscratch/mschreyer/deepNadim/100_datasets/philadelphia/original_data/city_payments_fy2017_encoded_all_new_anomalies.csv \
#      --no_epochs 100 --params_path "./params/params_10exp_scenario0.yml" --wandb_proj "ContinualAnomaly_Scenario0" \
#      --bottleneck "tanh" --lwf_alpha 0.1 --lwf_temperature 2.0
#
#python main_continual.py --strategy "SynapticIntelligence" \
#      --data_dir /netscratch/mschreyer/deepNadim/100_datasets/philadelphia/original_data/city_payments_fy2017_encoded_all_new_anomalies.csv \
#      --no_epochs 100 --params_path "./params/params_10exp_scenario0.yml" --wandb_proj "ContinualAnomaly_Scenario0" \
#      --bottleneck "tanh" --si_lambda 5.0 --si_eps 0.001
