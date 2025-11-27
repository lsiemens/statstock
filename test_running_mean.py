from matplotlib import pyplot as plt
import numpy as np
from scipy import signal

import portfolio


p_day = portfolio.Portfolio("all_holdings.csv", 2000, "OneDay")
p_week = portfolio.Portfolio("all_holdings.csv", 2000, "OneWeek")

auto_width = 20


def autocorr(x):
    x = x - np.mean(x)
    result = signal.correlate(x, x, mode="same")
    result = result[:len(result)//2 + 1]
    result = result/result[-1]
    result = result[-auto_width:]
    return result


def cumulative_mean(x):
    return np.cumsum(x)/(np.arange(len(x)) + 1)


for ticker in ["SPY", "GLW"]:
    index_day = p_day.tickers.index(ticker)
    index_week = p_week.tickers.index(ticker)

    mask_day = np.isfinite(p_day.logprice[index_day])
    mask_week = np.isfinite(p_week.logprice[index_week])
    logprice_day = p_day.logprice[index_day, mask_day]
    logprice_week = p_week.logprice[index_week, mask_week]
    lnRet_day = np.diff(logprice_day)
    lnRet_week = np.diff(logprice_week)

    plt.hist(lnRet_day, bins=int(np.sqrt(len(lnRet_day))), alpha=0.5, density=True, label="day")
    plt.hist(lnRet_week, bins=int(np.sqrt(len(lnRet_week))), alpha=0.5, density=True, label="week")
    plt.title(f"Distribution: {ticker}")
    plt.legend()
    plt.show()

    auto_day = autocorr(lnRet_day)
    auto_week = autocorr(lnRet_week)
    x_day = np.arange(len(auto_day))
    x_day -= x_day[-1]
    x_week = 5*np.arange(len(auto_week))
    x_week -= x_week[-1]
    plt.plot(x_day, auto_day)
    plt.plot(x_week, auto_week)
    plt.title(f"Auto correlation log returns: {ticker}")
    plt.show()

    Rmean_lnRet_day = cumulative_mean(lnRet_day)
    Rmean_lnRet_day -= Rmean_lnRet_day[-1]
    Rmean_lnRet_week = cumulative_mean(lnRet_week)
    Rmean_lnRet_week -= Rmean_lnRet_week[-1]
    x_day = np.arange(len(lnRet_day))
    x_day -= x_day[-1]
    x_week = 5*np.arange(len(lnRet_week))
    x_week -= x_week[-1]
    plt.plot(x_day, Rmean_lnRet_day, label="day")
    plt.plot(x_week, Rmean_lnRet_week, label="week")
    plt.title(f"cumulative mean log returns: {ticker}")
    plt.legend()
    plt.show()

    auto2_day = autocorr(lnRet_day**2)
    auto2_week = autocorr(lnRet_week**2)
    x_day = np.arange(len(auto_day))
    x_day -= x_day[-1]
    x_week = 5*np.arange(len(auto_week))
    x_week -= x_week[-1]
    plt.plot(x_day, auto2_day)
    plt.plot(x_week, auto2_week)
    plt.title(f"Auto correlation (log returns)^2: {ticker}")
    plt.show()

    Rmean_lnRet2_day = cumulative_mean(lnRet_day**2)
    Rmean_lnRet2_day -= Rmean_lnRet2_day[-1]
    Rmean_lnRet2_week = cumulative_mean(lnRet_week**2)
    Rmean_lnRet2_week -= Rmean_lnRet2_week[-1]
    x_day = np.arange(len(lnRet_day))
    x_day -= x_day[-1]
    x_week = 5*np.arange(len(lnRet_week))
    x_week -= x_week[-1]
    plt.plot(x_day, Rmean_lnRet2_day, label="day")
    plt.plot(x_week, Rmean_lnRet2_week, label="week")
    plt.title(f"cumulative mean (log returns)^2: {ticker}")
    plt.legend()
    plt.show()

    auto2_day = autocorr(np.abs(lnRet_day))
    auto2_week = autocorr(np.abs(lnRet_week))
    x_day = np.arange(len(auto_day))
    x_day -= x_day[-1]
    x_week = 5*np.arange(len(auto_week))
    x_week -= x_week[-1]
    plt.plot(x_day, auto2_day)
    plt.plot(x_week, auto2_week)
    plt.title(f"Auto correlation abs(log returns): {ticker}")
    plt.show()

    Rmean_lnRet_abs_day = cumulative_mean(np.abs(lnRet_day))
    Rmean_lnRet_abs_day -= Rmean_lnRet_abs_day[-1]
    Rmean_lnRet_abs_week = cumulative_mean(np.abs(lnRet_week))
    Rmean_lnRet_abs_week -= Rmean_lnRet_abs_week[-1]
    x_day = np.arange(len(lnRet_day))
    x_day -= x_day[-1]
    x_week = 5*np.arange(len(lnRet_week))
    x_week -= x_week[-1]
    plt.plot(x_day, Rmean_lnRet_abs_day, label="day")
    plt.plot(x_week, Rmean_lnRet_abs_week, label="week")
    plt.title(f"cumulative mean abs(log returns): {ticker}")
    plt.legend()
    plt.show()
