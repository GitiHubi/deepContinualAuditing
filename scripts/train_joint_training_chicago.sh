for dept_seed in {0..4}
do
  echo "Running for seed $dept_seed"
  python main_joint.py --strategy "JointTraining" \
        --data_dir /netscratch/mschreyer/deepNadim/100_datasets/chicago/original_data/city_payments_encoded_all_new_anomalies.csv\
        --no_epochs 200 --params_path "./params/params_10exp_scenario3.yml" --wandb_proj "ContinualAnomaly_JointTraining_Chicago" \
        --bottleneck "tanh" --dept_seed $dept_seed
done


for dept_seed in {0..4}
do
  echo "Running for seed $dept_seed"
  python main_joint.py --strategy "JointTraining" \
        --data_dir /netscratch/mschreyer/deepNadim/100_datasets/chicago/original_data/city_payments_encoded_all_new_anomalies.csv \
        --no_epochs 200 --params_path "./params/params_10exp_scenario4.yml" --wandb_proj "ContinualAnomaly_JointTraining_Chicago" \
        --bottleneck "tanh" --dept_seed $dept_seed
done

