"""
本程序实现了一个类：RiskAppetite
根据每周的行业指数、市场指数（如HS300）、行业交易量
计算出每周对应的收益率、beta系数，使用spearman相关系数计算出三个情绪指数，并可以做假设检验给出看多看空信号。
通过模拟交易可以评估该模型的好坏。
"""
import numpy as np
from datetime import datetime, timedelta
import csv
from enum import Enum
import scipy
from scipy import stats
import matplotlib.pyplot as plt


class RiskAppetiteType(Enum):
    BETA_YIELD = 1   # beta系数-收益率
    VOLUME_YIELD = 2 # 交易量-收益率
    VOLUME_BETA = 3  # 交易量-beta系数

class TrendSignalType(Enum):
    BULL = 1    # 牛市信号（上涨）
    SHOKING = 2 # 震荡市信号（持平）
    BEAR = 3    # 熊市信号（下跌）

class RiskAppetite:


    def __init__(self) -> None:
        # 一级行业个数，申万行业为31
        self.industry_count = None
        # 一级行业指数数据，二维数组，[i][j][k]表示第i个行业第j天的
        # 指数的第k个数据
        # 数据：日期、收盘指数、成交量
        self.industry_index = []
        # 有一些行业是近几年新增的，数据不足，不参与计算
        self.industry_available = None
        # 市场指数，如HS300指数，[j][k]表示第j天的第k个数据
        self.market_index = []
        # 银行利率，[j]表示第j天的基准银行利率
        self.bank_rate = []
        # 周收益率每一次计算的日期
        self.yield_date = []
        # 开始日期
        self.begin_date = None
        # 结束日期
        self.end_date = None
        # beta计算时间窗口（单位：周）
        self.beta_time_window = None
        # 显著性水平
        self.significance_level = 0.05
    
    
    # 输入各个指数数据文件
    def set_index(self,
                  industry_filename : tuple, # 各行业指数文件路径
                  market_filename : str,     # 市场指数文件路径
                  bank_filename : str        # 银行利率文件路径
                  ) -> None:
        # 读取各行业数据
        self.industry_count = len(industry_filename)
        for filename in industry_filename : 
            with open(filename, 'r') as infile:
                reader = csv.reader(infile)
                next(reader) # 跳过第一行表头
                single_industry_index = [] # 本行业的指数数据
                for row in reader:
                    date_str = row[2]
                    closing_index = float(row[6])
                    volume = float(row[7])
                    date = datetime.strptime(date_str[0:10], "%Y-%m-%d")
                    single_industry_index.append((date, closing_index, volume))
            single_industry_index.reverse()
            self.industry_index.append(single_industry_index)

        # 读取市场数据
        with open(market_filename, 'r') as infile:
            reader = csv.reader(infile)
            next(reader) # 跳过第一行表头
            for row in reader:
                date_str = row[0]
                closing_index = float(row[3])
                volume = float(row[10])
                date_str_list = date_str.split('/')
                date = datetime(int(date_str_list[0]), int(date_str_list[1]), int(date_str_list[2]))
                self.market_index.append((date, closing_index, volume))
        self.market_index.reverse()

        # 读取银行数据
        with open(bank_filename, 'r') as infile:
            reader = csv.reader(infile)
            next(reader) # 跳过第一行表头
            for row in reader:
                date_str = row[0]
                deposit_interest_rate = float(row[1])
                date_str_list = date_str.split('/')
                date = datetime(int(date_str_list[0]), int(date_str_list[1]), int(date_str_list[2]))
                self.bank_rate.append((date, deposit_interest_rate))
        self.bank_rate.reverse()


    # 设置起始和终止的日期
    def set_date(self, start_date : str, end_date : str) -> None:
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.end_date =   datetime.strptime(end_date, "%Y-%m-%d")
    

    # 设置beta系数计算的时间窗口长度（单位：周）
    def set_beta_time_window(self, beta_time_window : int) -> None:
        self.beta_time_window = beta_time_window
    

    # 设置显著性水平
    def set_significance_level(self, siginificance_level : float) -> None:
        self.significance_level = siginificance_level

    # 计算收益率、周交易量和beta系数
    def calc(self) -> None:
        self.industry_available = [True for i in range(self.industry_count)]
        self.yield_date = set() # 登记所有的每周最后一天的日期（也就是周收益率的对应日期）
        # 计算各行业周收益率
        self.industry_yield = []
        self.industry_week_volume = []
        for pos, index in enumerate(self.industry_index):
            # 查看是否时间过短的行业类型
            if index[0][0] > self.start_date:
                self.industry_available[pos] = False
                continue
            # 计算该行业周收益率和周成交量
            single_industry_yield = []
            single_industry_volume = []
            for i, day in enumerate(index):
                try:
                    if day[0] < self.start_date \
                    or day[0] > self.end_date \
                    or i == len(index) - 1 \
                    or day[0] + timedelta(days=1) == index[i + 1][0]:
                        continue
                    # 对于每周末的数据进行处理
                    single_industry_yield.append(
                        (day[1] - index[i - 5][1]) / index[i - 5][1])

                    week_vol = 0
                    for j in range(i - 4, i + 1):
                        week_vol += index[j][2]
                    single_industry_volume.append(week_vol)

                    if pos == 0:
                        self.yield_date.add(day[0])
                except IndexError:
                    pass
                except Exception as e:
                    raise e

            # print("length = %d" % len(single_industry_yield))
            self.industry_yield.append(np.array(single_industry_yield))
            self.industry_week_volume.append(np.array(single_industry_volume))

        # 计算市场周收益率
        market_yield = []
        for i, day in enumerate(self.market_index):
            if day[0] not in self.yield_date:
                continue
            # 对于每周五的收盘数据进行处理
            market_yield.append(
                (day[1] - self.market_index[i - 5][1]) / self.market_index[i - 5][1])
        self.market_yield = np.array(market_yield)
        self.yield_date = list(self.yield_date)
        self.yield_date.sort()

        # 计算行业beta系数
        self.beta_coefficient = []
        for single_industry_yield in self.industry_yield:
            single_beta_coefficient = []
            for i, day in enumerate(single_industry_yield):
                if i < self.beta_time_window - 1:
                    continue
                industry_yield_window = single_industry_yield[i - self.beta_time_window + 1 : i+1]
                market_yield_window = self.market_yield[i - self.beta_time_window + 1 : i+1]
                cov = np.cov(np.vstack((industry_yield_window, market_yield_window)))
                single_beta_coefficient.append(cov[0, 1] / cov[1, 1])
            self.beta_coefficient.append(np.array(single_beta_coefficient))
        # print("%d, %d, %d" % (\
        #     self.market_yield.size,\
        #     len(self.beta_coefficient[0]),\
        #     len(self.market_yield)))
        print("> Involved industries: %d" % len(self.industry_yield))
        print("> Involved time scale: %d weeks" % len(self.yield_date))
        self.market_week_index = np.array([day[1] for day in self.market_index if day[0] in self.yield_date[self.beta_time_window - 1:]]) # 每周周末市场收盘指数，用于绘图
    

    # 找到列表中小于等于x的最大的元素。要求L升序排列。
    def _LowerBound(L : list, x) -> int:
        l = 0
        r = len(L) - 1
        ans = None
        while l <= r:
            mid = (l + r) // 2
            if L[mid] <= x:
                ans = mid
                l = mid + 1
            else:
                r = mid - 1
        return ans


    # 输入一个时间点，和想要的情绪指数类型，获取当天的情绪指数
    def RiskAppetiteIndex(self, date : datetime, type : RiskAppetiteType) -> float:
        id = RiskAppetite._LowerBound(self.yield_date, date)
        
        if id == None:
            raise TypeError("Time too early")
        # 情绪指数就是spearman系数
        if type == RiskAppetiteType.BETA_YIELD:
            try:
                return stats.spearmanr(
                    [single[id  - self.beta_time_window + 1] for single in self.beta_coefficient],
                    [single[id] for single in self.industry_yield])
            except IndexError:
                return None
            except Exception as ex:
                raise ex
        elif type == RiskAppetiteType.VOLUME_BETA:
            try:
                return stats.spearmanr(
                    [single[id - self.beta_time_window + 1] for single in self.beta_coefficient],
                    [single[id] for single in self.industry_week_volume])
            except IndexError:
                return None
            except Exception as ex:
                raise ex
        else:
            try:
                res = stats.spearmanr(
                    [single[id] for single in self.industry_yield],
                    [single[id] for single in self.industry_week_volume])
                return (res[0], res[1])
            except IndexError:
                return None
            except Exception as ex:
                raise ex

            


    # 输入一个时间点，和想要的情绪指数类型，获取当天的多空信号
    # 使用假设检验
    def TrendSignal(self, date : datetime, type : RiskAppetiteType) -> TrendSignalType:
        try:
            res = self.RiskAppetiteIndex(date, type)
            # print("[%s] res[0] = %f %s" % (str(date), res[0], str(res[1] < self.significance_level)))
            
            if res[1] < self.significance_level:
                if res[0] < 0:
                    return TrendSignalType.BEAR
                else: 
                    return TrendSignalType.BULL
            else:
                return TrendSignalType.SHOKING
        except TypeError:
            return TrendSignalType.SHOKING
        except Exception as ex:
            raise ex

    
    # 输出一个包含市场指数，Beta，RA指数的图
    def PrintBasicGraph(self, type : RiskAppetiteType) -> None:
        cutted_yield_date = self.yield_date[self.beta_time_window-1:]
        week_risk_appetite_index = np.array([self.RiskAppetiteIndex(date, type)[0] for date in cutted_yield_date])
        cutted_yield_date_str = [datetime.strftime(date, "%Y-%m-%d") for date in cutted_yield_date]

        fig, axs = plt.subplots(2, 1, figsize=(6, 8))
        ind = np.arange(len(cutted_yield_date))
        xtick_interval = 90

        axs[0].set_xticks([])
        axs[0].plot(ind, self.market_week_index, color = 'g', linestyle = '-', label = 'Market Index')
        axs[0].set_ylabel("Index")
        axs[0].set_title("%s Model Basic Graph" % type.name)
        axs[0].legend()

        axs[1].set_xticks(ind[::xtick_interval], cutted_yield_date_str[::xtick_interval], rotation=40)
        axs[1].plot(ind, self.beta_coefficient[0], color = 'k', linestyle = '--', label = 'Beta Coefficient')
        axs[1].bar(ind, week_risk_appetite_index, color = 'b', label = "Risk Appetite Index")
        axs[1].set_xlabel("Date")
        axs[1].set_ylabel("Index")
        axs[1].legend()
        fig.tight_layout()
        plt.savefig("./output_figure/%s_Model_Basic_Graph" % type.name, dpi = 100, facecolor = "#f1f1f1")

    
    # 根据多空信号进行模拟交易，验证收益率，结果直接输出
    def SimulateTrading(self, type : RiskAppetiteType) -> None:
        cutted_yield_date = self.yield_date[self.beta_time_window-1:]

        pre_signal = None
        capital = 1000.0 # 初始资金
        holding = 0.0     # 持仓
        max_history_capital = capital # 历史最大资金，用于计算最大回撤
        max_rollback = 0.0  # 最大回撤
        self.assets_history = [] # 总资产历史，用于生成图像
        self.signals = [] # 信号历史，用于生成图像
        for ind, date in enumerate(cutted_yield_date):
            self.assets_history.append(capital + holding * self.market_week_index[ind])
            signal = self.TrendSignal(date, type)
            if signal == TrendSignalType.SHOKING:
                continue
                
            if pre_signal == None:
                pre_signal = signal
                continue
            if pre_signal == signal and holding == 0:
                self.signals.append((ind, signal))
                if signal == TrendSignalType.BULL:
                    # 看多
                    holding += capital / self.market_week_index[ind]
                    capital = 0.0
                else:
                    # 看空
                    holding -= capital / self.market_week_index[ind]
                    capital += capital
            elif pre_signal != signal:
                # 恢复
                capital += holding * self.market_week_index[ind]
                holding = 0
            pre_signal = signal

            # if signal == TrendSignalType.BULL:
            #     capital += holding * market_week_index[ind]
            #     holding = 0
            #     # 看多
            #     holding += capital / market_week_index[ind]
            #     capital = 0.0
            # else:
            #     capital += holding * market_week_index[ind]
            #     holding = 0
            #     # 看空
            #     holding -= capital / market_week_index[ind]
            #     capital += capital
            # print("[%s] price = %f, capital = %f, holding = %f" % (str(date), self.market_week_index[ind], capital, holding))
            max_history_capital = max(max_history_capital, capital + holding * self.market_week_index[ind])
            max_rollback = max(max_rollback, 1 - (capital + holding * self.market_week_index[ind]) / max_history_capital)
        
        # 将结束时的股票换算成资金
        capital += holding * self.market_week_index[ind]
        holding = 0
        rate = capital / 1000.0
        print("> Yield: %f%%" % ((rate - 1) * 100))
        year_count = (self.end_date - self.start_date).days / 365
        print("> Annualized rate of return: %f%%" % (((rate ** (1 / year_count)) - 1) * 100))
        print("> Maximum rollback: %f%%" % (max_rollback * 100.0))



    # 将模拟交易的结果输出为图表
    def PrintSimulateTradingGraph(self, type : RiskAppetiteType) -> None:
        cutted_yield_date = self.yield_date[self.beta_time_window-1 :]
        self.SimulateTrading(type)
        cutted_yield_date_str = [datetime.strftime(date, "%Y-%m-%d") for date in cutted_yield_date]
        fig, ax = plt.subplots(figsize=(6, 4), dpi = 300)
        ind = np.arange(len(cutted_yield_date))
        xtick_interval = 90

        ax.set_xticks(ind[::xtick_interval], cutted_yield_date_str[::xtick_interval], rotation=40)
        ax.plot(ind, self.market_week_index, color = 'k', linestyle = '-', label = 'Market Index')
        ax.plot(ind, self.assets_history, color = 'b', linestyle = '--', label = "Assets History")
        bull_id = []
        bull_index = []
        bear_id = []
        bear_index = []
        for signal in self.signals:
            if signal[1] == TrendSignalType.BULL:
                bull_id.append(signal[0])
                bull_index.append(self.market_week_index[signal[0]])
            else:
                bear_id.append(signal[0])
                bear_index.append(self.market_week_index[signal[0]])
        print("BULL signal count: %d" % len(bull_id))
        print("BEAR signal count: %d" % len(bear_id))
        bull_id = np.array(bull_id)
        bull_index = np.array(bull_index)
        bear_id = np.array(bear_id)
        bear_index = np.array(bear_index)
        ax.scatter(bull_id, bull_index, color = 'red')
        ax.scatter(bear_id, bear_index, color = 'green')
        ax.set_ylabel("Index")
        ax.set_title("%s Model Basic Graph" % type.name)
        ax.legend()
        fig.tight_layout()
        plt.savefig("./output_figure/%s_Simulation_Graph" % type.name, dpi = 100, facecolor = "#f1f1f1")






                
                
            

