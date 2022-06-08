# ml-repository-azureml-cli-v2

## Dependencies

- Python >= 3.8
- conda

## Preparation

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

# Reference

https://docs.microsoft.com/en-us/azure/machine-learning/how-to-train-cli#sweep-hyperparameters
https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-register-data-assets?tabs=Python-SDK#uris
