import argparse
from typing import TypedDict

import lightgbm as lgb
import mlflow
import pandas as pd
from azureml.core import Dataset, Run
from sklearn.model_selection import train_test_split


class GetArgsOutput(TypedDict):
    input_dataset_name: str
    boosting_type: str
    metric: str
    learning_rate: float
    num_leaves: int
    min_data_in_leaf: int
    num_iteration: int


class LoadDatasetOutput(TypedDict):
    x_train: pd.DataFrame
    y_train: pd.DataFrame
    x_test: pd.DataFrame
    y_test: pd.DataFrame


def get_args() -> GetArgsOutput:
    # 引数取得
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dataset_name", type=str)
    parser.add_argument("--boosting_type", type=str, default="gbdt")
    parser.add_argument("--metric", type=str, default="rmse")
    parser.add_argument("--learning_rate", type=float, default=0.1)
    # FIXME: cli v2 の都合で int の探索空間を設定できないため float を渡してキャスト
    parser.add_argument("--num_leaves", type=float, default=10)
    # FIXME: cli v2 の都合で int の探索空間を設定できないため float を渡してキャスト
    parser.add_argument("--min_data_in_leaf", type=float, default=1)
    parser.add_argument("--num_iteration", type=int, default=100)

    args = parser.parse_args()

    params: GetArgsOutput = {
        "input_dataset_name": args.input_dataset_name,
        "boosting_type": args.boosting_type,
        "metric": args.metric,
        "learning_rate": args.learning_rate,
        "num_leaves": int(args.num_leaves),
        "min_data_in_leaf": int(args.min_data_in_leaf),
        "num_iteration": args.num_iteration,
    }

    return params


def load_dataset(input_dataset_name: str) -> LoadDatasetOutput:
    run = Run.get_context()
    ws = run.experiment.workspace

    # Dataset インスタンス取得
    dataset = Dataset.get_by_name(ws, input_dataset_name)

    # データ取得＆加工

    df = dataset.to_pandas_dataframe()

    train, test = train_test_split(df, test_size=0.2, random_state=1234)

    x_train = train[train.columns[train.columns != "totalAmount"]]
    y_train = train["totalAmount"]

    x_test = test[test.columns[test.columns != "totalAmount"]]
    y_test = test["totalAmount"]

    output: LoadDatasetOutput = {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test}

    return output


def train_lgb_model(args: GetArgsOutput, datasets: LoadDatasetOutput) -> lgb.Booster:
    # mlflow autolog 開始
    # ジョブ実行の場合 Azure ML が初期設定する環境変数をもとに mlflow が自動でセッティングされる

    train_dataset = lgb.Dataset(datasets["x_train"], datasets["y_train"])
    eval_dataset = lgb.Dataset(datasets["x_test"], datasets["y_test"], reference=train_dataset)

    mlflow.lightgbm.autolog(registered_model_name="nyc_taxi_regressor_lightgbm")

    # パラメーター記録

    params = {
        "boosting_type": args["boosting_type"],
        "metric": args["metric"],
        "learning_rate": args["learning_rate"],
        "num_leaves": args["num_leaves"],
        "min_data_in_leaf": args["min_data_in_leaf"],
        "num_iteration": args["num_iteration"],
        "task": "train",
        "objective": "regression",
    }
    mlflow.log_params(params)

    # 学習
    gbm = lgb.train(params, train_dataset, num_boost_round=50, valid_sets=eval_dataset, early_stopping_rounds=10)
    mlflow.log_metric("best" + params["metric"], gbm.best_score["valid_0"]["rmse"])

    return gbm


def main() -> None:
    args = get_args()
    datasets = load_dataset(args["input_dataset_name"])
    train_lgb_model(args=args, datasets=datasets)


if __name__ == "__main__":
    main()
