import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import DataLoader


class BackTest:
    def __init__(self, model: nn.Module, data_loader: DataLoader, total_cash=10000, open_cost=0.001, close_cost=0.001, min_cost=5):
        self.model = model
        self.data_loader = data_loader
        self.total_cash = total_cash
        self.open_cost = open_cost
        self.close_cost = close_cost
        self.min_cost = min_cost

        # 初始化记录变量
        self.daily_positions = []  # 每日持仓mask
        self.daily_values = []     # 每日总价值
        self.daily_cash = []       # 每日现金
        self.daily_stock_values = [] # 每日股票价值

    @staticmethod
    def stockMask_topk(scores, x):
        _, idx = torch.topk(scores, x)
        mask = torch.zeros_like(scores)
        mask[idx] = 1
        return mask  # [n]，只有x个1，其余为0
    
    @staticmethod
    def stockMask_overx(scores, x):
        mask = (scores > x).int()
        return mask  # [n]，大于x的为1，其余为0

    # type=1表示训练集，2表示验证集，3表示测试集
    def run(self, type=1, selection_method='threshold', selection_param=0.5):
        if type == 1:
            days = self.data_loader.train_dates
        elif type == 2:
            days = self.data_loader.valid_dates
        elif type == 3:
            days = self.data_loader.test_dates
        else:
            raise ValueError("Invalid type. Use 1 for train, 2 for valid, 3 for test.")

        if len(days) == 0:
            raise ValueError("No trading days in the specified range.")
        
        current_cash = self.total_cash
        current_positions = {}
        self.model.eval()

        with torch.no_grad():
            for i, date in enumerate(days):
                print(f"Processing date: {date}")

                # 获取当天数据
                daily_data = self.data_loader.get_daily_data(date)
                if daily_data is None:
                    continue
                
                # 获取特征
                stock_codes = daily_data.index.tolist()
                features = torch.tensor(daily_data.values, dtype=torch.float32)

                # 创建当前持仓mask。 TODO 真的用mask吗？
                current_position_mask = torch.zeros(len(stock_codes))
                for j, stock in enumerate(stock_codes):
                    if stock in current_positions and current_positions[stock] > 0:
                        current_position_mask[j] = 1.0
                
                # 将特征和持仓mask拼接
                model_input = torch.cat([features, current_position_mask.unsqueeze(1)], dim=1)

                # 模型预测
                scores = self.model(model_input.unsqueeze(0)).squeeze(0)

                # 根据选择方法生成新的持仓mask
                if selection_method == 'topk':
                    new_position_mask = self.stockMask_topk(scores, selection_param)
                else:
                    new_position_mask = self.stockMask_overx(scores, selection_param)

                # 保存持仓mask
                self.daily_positions.append({
                    'date': date,
                    'mask': new_position_mask.numpy(),
                    'stocks': stock_codes
                })

                # 第一天不交易，只记录预测
                if i == 0:
                    self.daily_cash.append(current_cash)
                    self.daily_stock_values.append(0)
                    self.daily_values.append(current_cash)
                    continue

                # 执行交易  TODO
                # 1. 先卖出不在新持仓中的股票
                stocks_to_sell = []
                for stock in current_positions:
                    if stock in stock_codes:
                        stock_idx = stock_codes.index(stock)
                        if new_position_mask[stock_idx] == 0 and current_positions[stock] > 0:
                            stocks_to_sell.append(stock)
                    else:
                        # 股票不在今天的列表中，全部卖出
                        stocks_to_sell.append(stock)
                
                for stock in stocks_to_sell:
                    if stock in stock_codes:
                        stock_idx = stock_codes.index(stock)
                        # 假设使用收盘价交易
                        price = daily_data.iloc[stock_idx]['close'] if 'close' in daily_data.columns else daily_data.iloc[stock_idx, -1]
                        sell_value = current_positions[stock] * price
                        sell_cost = self.calculate_cost(sell_value, is_buy=False)
                        current_cash += sell_value - sell_cost
                    del current_positions[stock]

                # 2. 买入新持仓中的股票
                stocks_to_buy = []
                for j, stock in enumerate(stock_codes):
                    if new_position_mask[j] == 1 and (stock not in current_positions or current_positions[stock] == 0):
                        stocks_to_buy.append((j, stock))

                # 计算每只股票分配的资金
                if len(stocks_to_buy) > 0:
                    # 预留一部分资金用于手续费
                    available_cash = current_cash * 0.98
                    cash_per_stock = available_cash / len(stocks_to_buy)
                    
                    for stock_idx, stock in stocks_to_buy:
                        price = daily_data.iloc[stock_idx]['close'] if 'close' in daily_data.columns else daily_data.iloc[stock_idx, -1]
                        if price > 0:
                            shares = int(cash_per_stock / price)
                            if shares > 0:
                                buy_value = shares * price
                                buy_cost = self.calculate_cost(buy_value, is_buy=True)
                                if current_cash >= buy_value + buy_cost:
                                    current_positions[stock] = shares
                                    current_cash -= buy_value + buy_cost
                # 计算当日总价值
                stock_value = 0
                for stock, shares in current_positions.items():
                    if stock in stock_codes:
                        stock_idx = stock_codes.index(stock)
                        price = daily_data.iloc[stock_idx]['close'] if 'close' in daily_data.columns else daily_data.iloc[stock_idx, -1]
                        stock_value += shares * price
                
                total_value = current_cash + stock_value
                
                # 记录每日数据
                self.daily_cash.append(current_cash)
                self.daily_stock_values.append(stock_value)
                self.daily_values.append(total_value)
        
        # 生成回测报告  TODO
        self.generate_report()
        
        return self.daily_values         # TODO

