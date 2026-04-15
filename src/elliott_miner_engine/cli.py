from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from .data_sources import CoinGeckoUniverseLoader, ExchangeUniverseLoader, YahooMarketData
from .engine import ElliottWaveEngine
from .plotting import plot_scan_result
from .scanner import MarketScanner


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description='Elliott Wave Miner-style scanner')
    sub = p.add_subparsers(dest='cmd', required=True)

    one = sub.add_parser('analyze', help='Analyze one symbol')
    one.add_argument('--symbol', required=True)
    one.add_argument('--market', default='custom')
    one.add_argument('--interval', default='1d')
    one.add_argument('--period', default='max')
    one.add_argument('--plot', default='')

    scan = sub.add_parser('scan', help='Scan a market universe')
    scan.add_argument('--market', required=True, choices=['ihsg', 'us_stocks', 'forex', 'commodities', 'crypto'])
    scan.add_argument('--interval', default='1d')
    scan.add_argument('--period', default='max')
    scan.add_argument('--limit', type=int, default=50)
    scan.add_argument('--out', default='scan_results.csv')

    return p


def load_universe(market: str):
    if market == 'ihsg':
        return ExchangeUniverseLoader.load_idx()['symbol'].tolist()
    if market == 'us_stocks':
        return ExchangeUniverseLoader.load_us_equities(common_only=True)['symbol'].tolist()
    if market == 'forex':
        return ExchangeUniverseLoader.load_forex_default()['symbol'].tolist()
    if market == 'commodities':
        return ExchangeUniverseLoader.load_commodities_default()['symbol'].tolist()
    if market == 'crypto':
        cg = CoinGeckoUniverseLoader.load()
        # Yahoo covers major coins more consistently than the full CoinGecko universe.
        # For fully exhaustive crypto scanning, replace with an exchange or CoinGecko OHLC adapter.
        majors = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD', 'BNB-USD', 'ADA-USD', 'DOGE-USD', 'TRX-USD', 'AVAX-USD', 'LINK-USD']
        return majors
    raise ValueError(f'Unsupported market: {market}')


def main() -> None:
    args = build_parser().parse_args()
    engine = ElliottWaveEngine()
    data = YahooMarketData()
    scanner = MarketScanner(engine=engine, data=data)

    if args.cmd == 'analyze':
        df = data.fetch(args.symbol, interval=args.interval, period=args.period)
        result = engine.analyze(df, symbol=args.symbol, market=args.market, interval=args.interval)
        print(json.dumps({
            'symbol': result.symbol,
            'pattern': None if result.best_candidate is None else result.best_candidate.pattern_type,
            'direction': None if result.best_candidate is None else result.best_candidate.direction,
            'score': None if result.best_candidate is None else result.best_candidate.score,
            'confidence': None if result.best_candidate is None else result.best_candidate.confidence,
            'invalidation': None if result.best_candidate is None else result.best_candidate.invalidation,
            'price_targets': [] if result.best_candidate is None else [asdict(t) for t in result.best_candidate.fib_price_targets],
            'time_targets': [] if result.best_candidate is None else [asdict(t) for t in result.best_candidate.fib_time_targets],
            'wave_duration_projections': [] if result.best_candidate is None else [asdict(t) for t in result.best_candidate.wave_duration_projections],
            'notes': [] if result.best_candidate is None else result.best_candidate.momentum_notes,
        }, default=str, indent=2))
        if args.plot:
            plot_scan_result(df, result, args.plot)

    elif args.cmd == 'scan':
        symbols = load_universe(args.market)
        results = scanner.scan_symbols(symbols, market=args.market, interval=args.interval, period=args.period, limit=args.limit)
        frame = scanner.to_frame(results)
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(args.out, index=False)
        print(frame.head(20).to_string(index=False))


if __name__ == '__main__':
    main()
