import stockdata
import virtualmarket

dir = "/DATA/lsiemens/Data/stocks/"
extension = ".us.txt"

etfs = []
with open("etf_market_tickers.txt", "r") as fin:
    etfs = fin.read().split("\n")

tickers = etfs
#tickers = [ticker for path, ticker in stockdata.list_stock_files(dir, extension)]
tickers = ["googl", "spy", "glw", "dia"]

cli1 = virtualmarket.Trivial_BuyHold(100000, ["spy", "glw"], [100, 10])
cli2 = virtualmarket.Trivial_BuyHold(10000, ["spy", "dia", "googl"], [10, 30, 40])

market = virtualmarket.Market(tickers, [cli1, cli2])
market.start()
