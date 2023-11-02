yield_date:
#     try:
#         res = appe.RiskAppetite(date, risk_appetite.RiskAppetiteType.BETA_YIELD)
#         if(abs(res[1]) < significance_level):
#             print("[%s] %f" % (str(date), res[0]))
#     except TypeError:
#         pass
#     except Exception as ex:
#         raise ex