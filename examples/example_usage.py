from elliott_miner_engine.data_sources import YahooMarketData
from elliott_miner_engine.engine import ElliottWaveEngine
from elliott_miner_engine.plotting import plot_scan_result


def main():
    data = YahooMarketData()
    engine = ElliottWaveEngine(min_reversal_pct=0.025, atr_mult=1.4)

    symbol = 'BTC-USD'
    df = data.fetch(symbol, interval='1d', period='5y')
    result = engine.analyze(df, symbol=symbol, market='crypto', interval='1d')

    print(result)
    plot_scan_result(df, result, output_path='example_btc.png')


if __name__ == '__main__':
    main()
