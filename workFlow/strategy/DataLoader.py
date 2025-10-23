import numpy as np
import pandas as pd
import TradingDate as TD

class DataLoader:
    def __init__(self, stock15featsPath, dataPath,
                 train_start_date, train_end_date,
                 valid_start_date, valid_end_date,
                 test_start_date, test_end_date,):
        
        self.data = pd.read_pickle(stock15featsPath)
        print(self.data.index.names)
        self.tradingDate = TD.TradingDate(dataPath)
        self.train_dates = self.tradingDate.getTradingDays(train_start_date, train_end_date)
        self.valid_dates = self.tradingDate.getTradingDays(valid_start_date, valid_end_date)
        self.test_dates = self.tradingDate.getTradingDays(test_start_date, test_end_date)

    def get_daily_data(self, date):
        """获取某一天的所有股票数据"""
        if not self.tradingDate.inTradingDays(date):
            print(f"Warning: {date} is not in trading days.")
            return None
        daily_data = self.data.xs(date, level='date', drop_level=False)
        return daily_data




# unit test
if __name__ == "__main__":
    data_loader = DataLoader(
        stock15featsPath="/Users/walterswang/Documents/GitHub/Qtrade/workFlow/strategy/stock15feats.bin",
        dataPath="/Users/walterswang/Documents/GitHub/qlib_bin/qlib_bin_norm",
        train_start_date="2021-09-01",
        train_end_date="2022-08-31",
        valid_start_date="2022-09-01",
        valid_end_date="2023-08-31",
        test_start_date="2023-09-01",
        test_end_date="2024-08-31"
    )

    # 读取数据
    print("Train Dates:", data_loader.train_dates)
    print("Valid Dates:", data_loader.valid_dates)
    print("Test Dates:", data_loader.test_dates)

    # 获取某一天的数据
    sample_date = data_loader.train_dates[0]
    daily_data = data_loader.get_daily_data(sample_date)
    print(f"Data for {sample_date}:\n", daily_data)



