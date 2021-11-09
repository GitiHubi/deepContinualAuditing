
for seed in {0..4}
do
  echo "Running for seed $seed"
  python main_continual.py --strategy "Naive" \
        --data_dir /netscratch/mschreyer/deepNadim/100_datasets/philadelphia/original_data/city_payments_fy2017_encoded_all_new_anomalies.csv \
        --no_epochs 100 --params_path "./params/params_10exp_scenarioB0.yml" --wandb_proj "ContinualAnomaly_ScenarioB" \
        --bottleneck "tanh" --seed $seed
done


for seed in {0..4}
do
  echo "Running for seed $seed"
  python main_continual.py --strategy "Naive" \
        --data_dir /netscratch/mschreyer/deepNadim/100_datasets/philadelphia/original_data/city_payments_fy2017_encoded_all_new_anomalies.csv \
        --no_epochs 100 --params_path "./params/params_10exp_scenarioB1.yml" --wandb_proj "ContinualAnomaly_ScenarioB" \
        --bottleneck "tanh" --seed $seed
done



for seed in {0..4}
do
  echo "Running for seed $seed"
  python main_continual.py --strategy "Naive" \
        --data_dir /netscratch/mschreyer/deepNadim/100_datasets/philadelphia/original_data/city_payments_fy2017_encoded_all_new_anomalies.csv \
        --no_epochs 100 --params_path "./params/params_10exp_scenarioB2.yml" --wandb_proj "ContinualAnomaly_ScenarioB" \
        --bottleneck "tanh" --seed $seed
done
