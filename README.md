# ml-repository-azureml-cli-v2

Sample repository for MLOps (especially Continuous Deployment or Training).

## Dependencies

- Python >= 3.8
- conda

## Preparation

### Create conda environment

```bash
conda env create -f environment.yml
conda activate env-ml-reposiotry-azureml-cli-v2
```

### pre-commit

```bash
pre-commit install
```

### Create resources to execute job

Set below variables for your environment.

- subscription_id = "subscription_id"
- resource_group = "resource_group_name"
- workspace_name = "ml_workspace_name"

```bash
python prepare.py
```

prepare.py script create a compute cluster, a dataset for regression and an environment optionally.

## Job execution

```bash
az ml job create -f ./job/search_hyperparameter.yml -g <resource_group> -w <ml_workspace>
```

## Continuous Training

### Required

1. Register application and grant Azure ML access.
1. Set below variables from registered application and Azure ML workspace as Github secrets.
   1. AZURE_CLIENT_ID
   1. AZURE_TENANT_ID
   1. AZURE_SUBSCRIPTION_ID
   1. AZURE_RESOURCE_GROUP_NAME
   1. AZURE_ML_WORKSPACE_NAME

https://docs.microsoft.com/en-us/azure/developer/github/connect-from-azure?tabs=azure-portal%2Clinux

## Code quality

To maintain code quality, below libraries are used in this repository.

- flake8 : lint based on PEP8
- black : auto-format based on PEP8 (flake8)
- isort : auto-sort import
- mypy : check type
- pre-commit : check code before commit

Code quality is important for team development.

## Reference

https://docs.microsoft.com/en-us/azure/machine-learning/how-to-train-cli#sweep-hyperparameters
https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-register-data-assets?tabs=Python-SDK#uris
