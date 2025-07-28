import sys
sys.path.insert(0, r"D:\gitdesktop\Qtrade\qlib")
import qlib
from qlib.constant import REG_CN
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord
from qlib.contrib.model.pytorch_alstm import ALSTM
from qlib.contrib.data.handler import Alpha158
from qlib.data.dataset import DatasetH
from qlib.tests.data import GetData
from qlib.tests.config import CSI300_BENCH

if __name__ == "__main__":
    provider_uri = r"D:\gitdesktop\Qtrade\cn_alldata"
    GetData().qlib_data(target_dir=provider_uri, region=REG_CN, exists_skip=True)
    qlib.init(provider_uri=provider_uri, region=REG_CN)

    ### 自定义数据集设置 ###
    handler = Alpha158(
        instruments="csi300",
        start_time="2018-01-01",
        end_time="2025-05-01",
    )

    dataset = DatasetH(
        handler=handler,
        segments={
            "train": ("2018-01-01", "2020-12-31"),
            "valid": ("2021-01-01", "2023-12-31"),
            "test": ("2024-01-01", "2025-05-01"),
        }
    )



    # ======== 特征数据处理 ========
    df_feature_raw = dataset.handler.fetch(col_set="feature")

    # 打印特征个数
    print("特征个数：", df_feature_raw.shape[1])
    print("特征样本数：", df_feature_raw.shape[0])

    # 1. 打印 NaN 统计信息（特征）
    print("【特征 - NaN 分析】")
    print("总 NaN 数量：", df_feature_raw.isna().sum().sum())
    print("每列 NaN 数量：")
    print(df_feature_raw.isna().sum().sort_values(ascending=False))

    # 2. 清理特征数据（此处用 0 填充，可替换为其他策略）
    df_feature_clean = df_feature_raw.fillna(0)

    # 3. 更新 handler 中的 feature 数据
    dataset.handler._data["feature"] = df_feature_clean

    # ======== 标签数据处理 ========
    df_label_raw = dataset.handler.fetch(col_set="label")

    # 1. 打印 NaN 统计信息（标签）
    print("\n【标签 - NaN 分析】")
    print("总 NaN 数量：", df_label_raw.isna().sum().sum())
    print("每列 NaN 数量：")
    print(df_label_raw.isna().sum().sort_values(ascending=False))

    # 2. 清理标签数据（此处用 0 填充，也可以限制区间）
    df_label_clean = df_label_raw.fillna(-0.0000001)    # 使用 -0.0000001 填充 NaN，避免与实际数据冲突

    # 3. 更新 handler 中的 label 数据
    dataset.handler._data["label"] = df_label_clean

    # ======== 可选：打印清洗后的示例行 ========
    print("\n【特征样本行】：")
    print(df_feature_clean.iloc[0])

    print("\n【标签样本行】：")
    print(df_label_clean.iloc[0])




    # ### 使用 Qlib 自带 GRU 模型 ###
    # model = ALSTM(
    #     d_feat=158,           # 使用 Alpha158，特征维度为 158
    #     hidden_size=64,
    #     num_layers=2,
    #     dropout=0.0,
    #     n_epochs=20,
    #     lr=0.001,
    #     early_stop=5,
    #     batch_size=800,
    #     metric="loss",
    #     loss="mse",
    #     GPU=0,                # 改为 None 使用 CPU
    # )

    # ### 回测设置 ###
    # port_analysis_config = {
    #     "executor": {
    #         "class": "SimulatorExecutor",
    #         "module_path": "qlib.backtest.executor",
    #         "kwargs": {
    #             "time_per_step": "day",
    #             "generate_portfolio_metrics": True,
    #         },
    #     },
    #     "strategy": {
    #         "class": "TopkDropoutStrategy",
    #         "module_path": "qlib.contrib.strategy.signal_strategy",
    #         "kwargs": {
    #             "signal": (model, dataset),
    #             "topk": 10,
    #             "n_drop": 1,
    #         },
    #     },
    #     "backtest": {
    #         "start_time": "2023-01-01",
    #         "end_time": "2025-05-01",
    #         "account": 50000,
    #         "benchmark": CSI300_BENCH,
    #         "exchange_kwargs": {
    #             "freq": "day",
    #             "limit_threshold": 0.095,
    #             "deal_price": "close",
    #             "open_cost": 0.0005,
    #             "close_cost": 0.0015,
    #             "min_cost": 5,
    #         },
    #     },
    # }

    # # 检查训练数据
    # print(dataset.prepare("train").head())

    # ### Qlib 记录流程 ###
    # with R.start(experiment_name="workflow_tft_custom"):
    #     model.fit(dataset)
    #     R.save_objects(**{"params.pkl": model})

    #     recorder = R.get_recorder()
    #     sr = SignalRecord(model, dataset, recorder)
    #     sr.generate()

    #     sar = SigAnaRecord(recorder)
    #     sar.generate()

    #     par = PortAnaRecord(recorder, port_analysis_config, "day")
    #     par.generate()
