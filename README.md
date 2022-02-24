# Deep Continual Auditing

PyTorch implementation of **Continual Learning for Unsupervised Anomaly 
Detection in Continuous Auditing of Financial Accounting Data** 
<a href="https://arxiv.org/abs/2112.13215"> [link to the paper] </a>


## Code Structure

    
    ├── DeepContinualAuditing                    
        ├── BenchmarkConfigs
            ├── ...                   # benchmark config files as YAML files
        ├── Data
            ├── ...                   # Datasets as CSV files (should be copied here)
        ├── ExperimentHandler
            ├── ...                   # Implementation of different strategies (CL, Scratch, Joint)
        ├── NetworkHandler
            ├── ...                   # Implementation of the autoencoder model used in the experiments
        ├── Scripts
            ├── ...                   # Scripts for reproducibility
        ├── UtilsHandler
            ├── ...                   # Different util files for strategy and benchmark
        ├── main.py                   # Main function is implemented here.


## Datasets
Datasets used in the paper can be downloaded from here:<br>
LINK-TO-BE-ADDED <br>
After downloading the CSV files, copy them to `./Data/` in the main
directory of the repository.

## Running an experiment
All scripts to reproduce results are saved under `./Scripts/`, therefore 
to run an experiment you can simply execute the following command:

```shell
bash Scripts/FOLDER_NAME/BASH_SCRIPT_FILENAME.sh 
```

If datasets are stored in a different folder than `./Data`, you need to change
`--data_dir` in the script you aim to run correspondingly.
