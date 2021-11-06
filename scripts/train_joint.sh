python main_joint.py --strategy "JointTraining" \
      --data_dir /netscratch/mschreyer/deepNadim/100_datasets/philadelphia/original_data/city_payments_fy2017_encoded_all_new.csv \
      --no_epochs 200 --params_path "./params/params_10exp_distinct.yml" --n_exp 10 --wandb_proj "ContinualAnomaly" \
      --bottleneck "tanh"
