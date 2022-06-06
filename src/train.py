import argparse
from typing import Any, Dict, TypedDict

import lightgbm as lgb
import mlflow
import pandas as pd
from azureml.core import Run
from mlflow.models.signature import infer_signature
from sklearn.model_selection import train_test_split


class LoadDatasetOutput(TypedDict):
    x_train: pd.DataFrame
    y_train: pd.DataFrame
    x_test: pd.DataFrame
    y_test: pd.DataFrame


def get_parameters() -> Dict[str, Any]:
    # 引数取得
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data", type=str)
    parser.add_argument("--boosting_type", type=str, default="gbdt")
    parser.add_argument("--metric", type=str, default="rmse")
    parser.add_argument("--learning_rate", type=float, default=0.1)
    parser.add_argument("--num_leaves", type=int, default=10)
    parser.add_argument("--min_data_in_leaf", type=int, default=1)
    parser.add_argument("--num_iteration", type=int, default=100)

    args = parser.parse_args()

    params = {
        "task": "train",
        "boosting_type": args.boosting_type,
        "objective": "regression",
        "metric": args.metric,
        "learning_rate": args.learning_rate,
        "num_leaves": args.num_leaves,
        "min_data_in_leaf": args.min_data_in_leaf,
        "num_iteration": args.num_iteration,
    }

    return params


def load_dataset() -> LoadDatasetOutput:
    run = Run.get_context()

    # 入力した Dataset インスタンス取得

    dataset = run.input_datasets["nyc_taxi_dataset"]

    # データ取得＆加工

    df = dataset.to_pandas_dataframe()

    train, test = train_test_split(df, test_size=0.2, random_state=1234)

    x_train = train[train.columns[train.columns != "totalAmount"]]
    y_train = train["totalAmount"]

    x_test = test[test.columns[test.columns != "totalAmount"]]
    y_test = test["totalAmount"]

    return {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test}


def train_lgb_model(params: Dict[str, Any], datasets: LoadDatasetOutput) -> lgb.Booster:
    # mlflow autolog 開始
    # ジョブ実行の場合 Azure ML が初期設定する環境変数をもとに mlflow が自動でセッティングされる
    # 対話的な実験で実行したような URI の取得とセットは不要
    # mlflow_uri = ws.get_mlflow_tracking_uri()
    # mlflow.set_tracking_uri(mlflow_uri)

    train_dataset = lgb.Dataset(datasets["x_train"], datasets["y_train"])
    eval_dataset = lgb.Dataset(datasets["x_test"], datasets["y_test"], reference=train_dataset)

    mlflow.lightgbm.autolog()

    # パラメーター記録

    mlflow.log_params(params)

    # 学習

    gbm = lgb.train(params, train_dataset, num_boost_round=50, valid_sets=eval_dataset, early_stopping_rounds=10)

    return gbm


def save_model(model: lgb.Booster, x_test: pd.DataFrame) -> None:
    y_pred = model.predict(x_test, num_iteration=model.best_iteration)
    signature = infer_signature(x_test, y_pred)

    # model保存
    # run の output に model が紐づく
    # GUI から model の登録が可能に

    mlflow.lightgbm.save_model(model, "MLflow")
    mlflow.lightgbm.log_model(model, artifact_path="MLflow", signature=signature)


def main() -> None:
    params = get_parameters()
    datasets = load_dataset()
    model = train_lgb_model(params=params, datasets=datasets)
    save_model(model, datasets["x_test"])


if __name__ == "__main__":
    main()
