from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional
from urllib.request import Request, urlopen

import pandas as pd


@dataclass(slots=True)
class YahooMarketData:
    auto_adjust: bool = False
    timeout: int = 30

    def fetch(self, symbol: str, interval: str = '1d', period: str = 'max') -> pd.DataFrame:
        import yfinance as yf

        df = yf.download(
            tickers=symbol,
            interval=interval,
            period=period,
            auto_adjust=self.auto_adjust,
            progress=False,
            threads=False,
        )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if df.empty:
            raise ValueError(f'No data returned for {symbol}')
        return df[['Open', 'High', 'Low', 'Close', 'Volume']].dropna(subset=['Open', 'High', 'Low', 'Close'])


class ExchangeUniverseLoader:
    IDX_STOCK_LIST_URL = 'https://www.idx.co.id/en/market-data/stocks-data/stock-list'
    NASDAQ_LISTED_URL = 'https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt'
    OTHER_LISTED_URL = 'https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt'

    @staticmethod
    def _read_url_text(url: str, timeout: int = 30) -> str:
        req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urlopen(req, timeout=timeout) as resp:
            return resp.read().decode('utf-8', errors='ignore')

    @classmethod
    def load_idx(cls) -> pd.DataFrame:
        tables = pd.read_html(cls.IDX_STOCK_LIST_URL)
        if not tables:
            raise ValueError('IDX stock list not found')
        df = tables[0].copy()
        code_col = next((c for c in df.columns if 'Code' in str(c) or 'Kode' in str(c)), df.columns[0])
        name_col = next((c for c in df.columns if 'Company' in str(c) or 'Perusahaan' in str(c) or 'Name' in str(c)), df.columns[1])
        out = pd.DataFrame({'symbol': df[code_col].astype(str).str.strip() + '.JK', 'raw_symbol': df[code_col].astype(str).str.strip(), 'name': df[name_col].astype(str).str.strip()})
        return out.drop_duplicates('symbol').reset_index(drop=True)

    @classmethod
    def _parse_nasdaq_symdir(cls, txt: str, symbol_col: str, name_col: str) -> pd.DataFrame:
        lines = [x for x in txt.splitlines() if x and not x.startswith('File Creation Time') and '|' in x]
        header = lines[0].split('|')
        rows = [line.split('|') for line in lines[1:] if not line.startswith('File Creation Time')]
        df = pd.DataFrame(rows, columns=header)
        df = df[df[symbol_col].str.upper() != 'SYMBOL']
        df = df[~df[symbol_col].str.contains('File Creation Time', case=False, na=False)]
        df = df[df[symbol_col].str.fullmatch(r'[A-Z0-9\.\-\$]+', na=False)]
        out = pd.DataFrame({'symbol': df[symbol_col].astype(str).str.strip(), 'name': df[name_col].astype(str).str.strip()})
        return out.drop_duplicates('symbol').reset_index(drop=True)

    @classmethod
    def load_us_equities(cls, common_only: bool = False) -> pd.DataFrame:
        nasdaq_txt = cls._read_url_text(cls.NASDAQ_LISTED_URL)
        other_txt = cls._read_url_text(cls.OTHER_LISTED_URL)
        nasdaq = cls._parse_nasdaq_symdir(nasdaq_txt, 'Symbol', 'Security Name')
        other = cls._parse_nasdaq_symdir(other_txt, 'ACT Symbol', 'Security Name')
        out = pd.concat([nasdaq, other], ignore_index=True).drop_duplicates('symbol').reset_index(drop=True)
        if common_only:
            mask = ~out['symbol'].str.contains(r'\$|\^|/|\\')
            mask &= ~out['symbol'].str.endswith(('W', 'R', 'U'))
            out = out[mask].reset_index(drop=True)
        return out

    @staticmethod
    def load_forex_default() -> pd.DataFrame:
        pairs = [
            'EURUSD=X', 'GBPUSD=X', 'USDJPY=X', 'USDCHF=X', 'AUDUSD=X', 'NZDUSD=X',
            'USDCAD=X', 'EURJPY=X', 'EURGBP=X', 'EURCHF=X', 'AUDJPY=X', 'GBPJPY=X',
            'CHFJPY=X', 'NZDJPY=X', 'AUDNZD=X', 'EURAUD=X', 'GBPAUD=X', 'GBPCAD=X',
            'EURCAD=X', 'CADJPY=X', 'USDIDR=X',
        ]
        return pd.DataFrame({'symbol': pairs, 'name': pairs})

    @staticmethod
    def load_commodities_default() -> pd.DataFrame:
        symbols = {
            'GC=F': 'Gold Futures',
            'SI=F': 'Silver Futures',
            'CL=F': 'Crude Oil WTI Futures',
            'BZ=F': 'Brent Crude Futures',
            'NG=F': 'Natural Gas Futures',
            'HG=F': 'Copper Futures',
            'ZC=F': 'Corn Futures',
            'ZW=F': 'Wheat Futures',
            'ZS=F': 'Soybean Futures',
            'KC=F': 'Coffee Futures',
            'CT=F': 'Cotton Futures',
        }
        return pd.DataFrame({'symbol': list(symbols.keys()), 'name': list(symbols.values())})


class CoinGeckoUniverseLoader:
    COINS_LIST_URL = 'https://api.coingecko.com/api/v3/coins/list'

    @classmethod
    def load(cls) -> pd.DataFrame:
        req = Request(cls.COINS_LIST_URL, headers={'User-Agent': 'Mozilla/5.0'})
        with urlopen(req, timeout=30) as resp:
            raw = resp.read().decode('utf-8', errors='ignore')
        df = pd.read_json(io.StringIO(raw))
        out = df.rename(columns={'id': 'cg_id', 'symbol': 'symbol', 'name': 'name'})
        out['symbol'] = out['symbol'].str.upper()
        return out[['cg_id', 'symbol', 'name']]


def format_symbol_for_market(symbol: str, market: str) -> str:
    symbol = symbol.strip().upper()
    if market == 'ihsg' and not symbol.endswith('.JK'):
        return f'{symbol}.JK'
    return symbol
