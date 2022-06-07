# ml-repository-azureml-cli-v2

## dependencies

- Python >= 3.8
- conda

## preparation

```bash
python prepare.py
```

prepare.py script create a compute cluster, a dataset for regression and an environment optionally.

## job execution

```bash
az ml job create -f ./job/search_hyperparameter.yml -g <resource_group> -w <ml_workspace>
```

# ref

https://docs.microsoft.com/en-us/azure/developer/github/connect-from-azure?tabs=azure-portal%2Clinux
https://docs.microsoft.com/en-us/azure/machine-learning/how-to-train-cli#sweep-hyperparameters
https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-register-data-assets?tabs=Python-SDK#uris
