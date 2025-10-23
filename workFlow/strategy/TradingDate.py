import os
import numpy as np
import pandas as pd

# dayfuture_set 是包含未来的交易日
# day_set 是已经经过的交易日
class TradingDate:
    def __init__(self, dataPath):
        self.dataPath = dataPath

        # 去查看是否有calendars文件夹
        calendars_path = os.path.join(dataPath, 'calendars')
        if not os.path.exists(calendars_path):
            raise FileNotFoundError(f"Calendars directory not found in {dataPath}")
        else:
            # 查看calendars目录下是否有day.txt 和 day_future.txt
            day_file = os.path.join(calendars_path, 'day.txt')
            dayfuture_set = os.path.join(calendars_path, 'day_future.txt')
            if not os.path.exists(day_file):
                raise FileNotFoundError(f"day.txt not found in {calendars_path}")
            if not os.path.exists(dayfuture_set):
                raise FileNotFoundError(f"dayfeature.txt not found in {calendars_path}")
            # 如果都存在，则读取day.txt和day_future.txt
            with open(day_file, 'r') as f:
                self.day_set = pd.to_datetime([line.strip() for line in f if line.strip()])
            with open(dayfuture_set, 'r') as f:
                self.dayfuture_set = pd.to_datetime([line.strip() for line in f if line.strip()])

    def getTradingDays(self, start_date, end_date):
        """获取已经经过的交易日列表"""
        mask = (self.day_set >= start_date) & (self.day_set <= end_date)
        trading_days = self.day_set[mask].sort_values()
        return pd.DatetimeIndex(trading_days)

    def getTradingDayFuture(self, start_date, end_date):
        """获取所有交易日特征列表【包括未来的交易日】"""
        mask = (self.dayfuture_set >= start_date) & (self.dayfuture_set <= end_date)
        trading_days = self.dayfuture_set[mask].sort_values()
        return pd.DatetimeIndex(trading_days)
    
    def inTradingDays(self, date):
        """判断某个日期是否为已经经过的交易日"""
        return date in self.day_set
    
    def inTradingDayFuture(self, date):
        """判断某个日期是否在所有交易日特征中【包括未来的交易日】"""
        return date in self.dayfuture_set
    





# unit test
if __name__ == "__main__":
    dataPath = "/Users/walterswang/Documents/GitHub/qlib_bin/qlib_bin_norm"
    trading_date = TradingDate(dataPath)
    print("Day Set:", trading_date.day_set)
    print("Day Future Set:", trading_date.dayfuture_set)

    print("Trading Days from 2025-01-01 to 2025-12-31:", 
          trading_date.getTradingDays("2025-01-01", "2025-12-31"))
    
    print("Trading Day Future from 2023-01-01 to 2023-12-31:", 
          trading_date.getTradingDayFuture("2023-01-01", "2023-12-31"))
    
    print("Is 2023-06-15 a trading day?", trading_date.inTradingDays("2023-06-15"))
    print("Is 2025-12-25 a trading day?", trading_date.inTradingDays("2025-12-25"))
    print("Is 2023-06-15 in trading day Future?", trading_date.inTradingDayFuture("2023-06-15"))
