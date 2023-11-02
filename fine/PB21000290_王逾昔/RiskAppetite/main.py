"""
投资者情绪指数择时模型
"""
import risk_appetite

"""
参数设定
"""
significance_level = 0.05             # 显著性水平
start_date_str = "2005-01-04"         # 开始时间
# start_date_str = "2014-02-21"       # 另一个开始时间
end_date_str = "2023-01-06"           # 结束时间
beta_time_window = 100                # beta系数计算时间窗口（单位：周）

"""
数据文件
"""
industry_filename = (
    "./data/801010.csv",
    "./data/801030.csv",
    "./data/801040.csv",
    "./data/801050.csv",
    "./data/801080.csv",
    "./data/801110.csv",
    "./data/801120.csv",
    "./data/801130.csv",
    "./data/801140.csv",
    "./data/801150.csv",
    "./data/801160.csv",
    "./data/801170.csv",
    "./data/801180.csv",
    "./data/801200.csv",
    "./data/801210.csv",
    "./data/801230.csv",
    "./data/801710.csv",
    "./data/801720.csv",
    "./data/801730.csv",
    "./data/801740.csv",
    "./data/801750.csv",
    "./data/801760.csv",
    "./data/801770.csv",
    "./data/801780.csv",
    "./data/801790.csv",
    "./data/801880.csv",
    "./data/801890.csv",
    "./data/801950.csv",
    "./data/801960.csv",
    "./data/801970.csv",
    "./data/801980.csv") # 各行业指数

bank_filename = "./data/bank.csv"     # 银行利率文件
market_filename = "./data/399300.csv" # 沪深300指数文件


"""
计算程序
"""
# 初始化和计算
print("Begin initiation...")
appe = risk_appetite.RiskAppetite()
appe.set_index(industry_filename, market_filename, bank_filename)
appe.set_date(start_date_str, end_date_str)
appe.set_beta_time_window(beta_time_window)
print("Initiation complete.\nBegin calculation...")
appe.calc()
print("Calculation complete.")

# # 风险偏好指数
# for type in risk_appetite.RiskAppetiteType:
#     print("Testing type %s" % type.name)
#     for date in appe.yield_date:
#         try:
#             res = appe.RiskAppetiteIndex(date, risk_appetite.RiskAppetiteType.BETA_YIELD)
#             if(abs(res[1]) < significance_level):
#                 print("[%s] %f" % (str(date), res[0]))
#         except TypeError:
#             pass
#         except Exception as ex:
#             raise ex

# # 多空信号
# appe.set_significance_level(significance_level)
# for type in risk_appetite.RiskAppetiteType:
#     signals = []
#     print("Testing type %s" % type.name)
#     for date in appe.yield_date:
#         res = appe.TrendSignal(date, type)
#         if res == risk_appetite.TrendSignalType.BULL:
#             signals.append("%s LONG" % str(date))
#         elif res == risk_appetite.TrendSignalType.BEAR:
#             signals.append("%s SHORT" % str(date))
#         res = appe.RiskAppetiteIndex(date, type)
#     print("Signal count: %d" % len(signals))
#     print("Average interval: %s" % str((appe.end_date - appe.start_date) / len(signals)))
#     print("Signals:")
#     for signal in signals:
#         print(signal)

# 基本绘图
for type in risk_appetite.RiskAppetiteType:
    print("Begin %s Model Basic Graph generation..." % type.name)
    appe.PrintBasicGraph(type)
    print("Generation completed.")

# 模拟交易
# for type in risk_appetite.RiskAppetiteType:
#     print("Begin %s Simulate Trading..." % type.name)
#     appe.SimulateTrading(type)
    
#     print("Simulate Trading completed.")

# 模拟交易绘图
for type in risk_appetite.RiskAppetiteType:
    print("Begin %s Simulate Trading..." % type.name)
    appe.PrintSimulateTradingGraph(type)
    print("Simulate Trading completed.")
    
