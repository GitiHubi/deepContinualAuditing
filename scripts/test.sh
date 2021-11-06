python test_continual.py --data_dir /netscratch/mschreyer/deepNadim/100_datasets/philadelphia/original_data/city_payments_fy2017_encoded_all_new.csv \
        --params_path ./params/params_5exp_gradual.yml --outputs_path ./outputs/ --run_name Naive_nexp10_20211105165253 \
        --bottleneck "tanh"

python test_continual.py --data_dir /netscratch/mschreyer/deepNadim/100_datasets/philadelphia/original_data/city_payments_fy2017_encoded_all_new.csv \
        --params_path ./params/params_5exp_gradual.yml --outputs_path ./outputs/ --run_name Replay_nexp10_20211105172216 \
        --bottleneck "tanh"

python test_continual.py --data_dir /netscratch/mschreyer/deepNadim/100_datasets/philadelphia/original_data/city_payments_fy2017_encoded_all_new.csv \
        --params_path ./params/params_3exp_distinct.yml --outputs_path ./outputs/ --run_name EWC_nexp10_20211105180533 \
        --bottleneck "tanh"

python test_continual.py --data_dir /netscratch/mschreyer/deepNadim/100_datasets/philadelphia/original_data/city_payments_fy2017_encoded_all_new.csv \
        --params_path ./params/params_3exp_distinct.yml --outputs_path ./outputs/ --run_name LwF_nexp10_20211105184610 \
        --bottleneck "tanh"

python test_continual.py --data_dir /netscratch/mschreyer/deepNadim/100_datasets/philadelphia/original_data/city_payments_fy2017_encoded_all_new.csv \
        --params_path ./params/params_3exp_distinct.yml --outputs_path ./outputs/ --run_name SynapticIntelligence_nexp10_20211105191548 \
        --bottleneck "tanh"
