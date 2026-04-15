from __future__ import annotations

import io
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

import pandas as pd


PACKAGE_ROOT = Path(__file__).resolve().parent
RESOURCE_DIR = PACKAGE_ROOT / 'resources'


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


@dataclass(slots=True)
class UniverseLoadResult:
    market: str
    df: pd.DataFrame
    source: str
    is_live: bool
    is_complete: Optional[bool]
    notes: List[str]


class ExchangeUniverseLoader:
    IDX_STOCK_LIST_URL = 'https://www.idx.co.id/en/market-data/stocks-data/stock-list'
    NASDAQ_LISTED_URL = 'https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt'
    OTHER_LISTED_URL = 'https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt'

    # Fallback snapshot repository. Not authoritative, but useful when IDX blocks bots.
    IDX_GITHUB_SECTOR_CSVS = [
        'Basic Materials.csv',
        'Consumer Cyclicals.csv',
        'Consumer Non-Cyclicals.csv',
        'Energy.csv',
        'Financials.csv',
        'Healthcare.csv',
        'Industrials.csv',
        'Infrastructures.csv',
        'Properties & Real Estate.csv',
        'Technology.csv',
        'Transportation & Logistic.csv',
    ]
    IDX_GITHUB_RAW_TEMPLATE = 'https://raw.githubusercontent.com/wildangunawan/Dataset-Saham-IDX/master/List%20Emiten/Sectors/{filename}'

    @staticmethod
    def _read_url_text(url: str, timeout: int = 30) -> str:
        req = Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urlopen(req, timeout=timeout) as resp:
            return resp.read().decode('utf-8', errors='ignore')

    @staticmethod
    def _read_csv_url(url: str, timeout: int = 30) -> pd.DataFrame:
        text = ExchangeUniverseLoader._read_url_text(url, timeout=timeout)
        return pd.read_csv(io.StringIO(text))

    @staticmethod
    def _read_local_csv(name: str) -> pd.DataFrame:
        path = RESOURCE_DIR / name
        if not path.exists():
            raise FileNotFoundError(f'Local resource not found: {path}')
        return pd.read_csv(path)

    @classmethod
    def load_idx(cls) -> pd.DataFrame:
        tables = pd.read_html(cls.IDX_STOCK_LIST_URL)
        if not tables:
            raise ValueError('IDX stock list not found')
        df = tables[0].copy()
        code_col = next((c for c in df.columns if 'Code' in str(c) or 'Kode' in str(c)), df.columns[0])
        name_col = next((c for c in df.columns if 'Company' in str(c) or 'Perusahaan' in str(c) or 'Name' in str(c)), df.columns[1])
        out = pd.DataFrame(
            {
                'symbol': df[code_col].astype(str).str.strip().str.upper() + '.JK',
                'raw_symbol': df[code_col].astype(str).str.strip().str.upper(),
                'name': df[name_col].astype(str).str.strip(),
            }
        )
        return out.drop_duplicates('symbol').reset_index(drop=True)

    @classmethod
    def load_idx_github_snapshot(cls) -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        for filename in cls.IDX_GITHUB_SECTOR_CSVS:
            url = cls.IDX_GITHUB_RAW_TEMPLATE.format(filename=quote(filename))
            try:
                df = cls._read_csv_url(url)
            except Exception:
                continue
            cols = {c.lower(): c for c in df.columns}
            code_col = cols.get('code') or cols.get('ticker') or cols.get('symbol')
            name_col = cols.get('nama') or cols.get('name') or cols.get('company') or cols.get('emiten')
            if code_col is None:
                continue
            if name_col is None:
                name_series = df[code_col].astype(str)
            else:
                name_series = df[name_col].astype(str)
            tmp = pd.DataFrame(
                {
                    'symbol': df[code_col].astype(str).str.strip().str.upper().str.replace('.JK', '', regex=False) + '.JK',
                    'raw_symbol': df[code_col].astype(str).str.strip().str.upper().str.replace('.JK', '', regex=False),
                    'name': name_series.str.strip(),
                }
            )
            frames.append(tmp)
        if not frames:
            raise ValueError('IDX GitHub fallback snapshot could not be loaded')
        return pd.concat(frames, ignore_index=True).drop_duplicates('symbol').reset_index(drop=True)

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
            'EURCAD=X', 'CADJPY=X', 'USDNOK=X', 'USDSEK=X', 'USDSGD=X', 'USDHKD=X',
            'USDZAR=X', 'USDMXN=X', 'USDTRY=X', 'USDIDR=X', 'USDCNH=X',
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
            'PL=F': 'Platinum Futures',
            'PA=F': 'Palladium Futures',
            'ZC=F': 'Corn Futures',
            'ZW=F': 'Wheat Futures',
            'ZS=F': 'Soybean Futures',
            'KC=F': 'Coffee Futures',
            'CT=F': 'Cotton Futures',
            'SB=F': 'Sugar #11 Futures',
            'CC=F': 'Cocoa Futures',
            'OJ=F': 'Orange Juice Futures',
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

    @staticmethod
    def load_yahoo_stable_fallback() -> pd.DataFrame:
        rows = [
            ('BTC-USD', 'Bitcoin'), ('ETH-USD', 'Ethereum'), ('SOL-USD', 'Solana'), ('XRP-USD', 'XRP'),
            ('BNB-USD', 'BNB'), ('ADA-USD', 'Cardano'), ('DOGE-USD', 'Dogecoin'), ('TRX-USD', 'Tron'),
            ('AVAX-USD', 'Avalanche'), ('LINK-USD', 'Chainlink'), ('HBAR-USD', 'Hedera'), ('DOT-USD', 'Polkadot'),
            ('TON-USD', 'Toncoin'), ('SUI-USD', 'Sui'), ('APT-USD', 'Aptos'), ('NEAR-USD', 'NEAR Protocol'),
            ('TAO-USD', 'Bittensor'), ('RENDER-USD', 'Render'), ('FET-USD', 'Artificial Superintelligence Alliance'),
            ('ONDO-USD', 'Ondo'),
        ]
        return pd.DataFrame(rows, columns=['symbol', 'name'])


def _safe_universe(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = None
    out = out[cols]
    if 'symbol' in out.columns:
        out['symbol'] = out['symbol'].astype(str).str.strip()
        out = out[out['symbol'].ne('')]
        out = out.drop_duplicates('symbol')
    return out.reset_index(drop=True)


def load_market_universe_safe(market: str, *, us_common_only: bool = False, allow_unverified_third_party: bool = False) -> UniverseLoadResult:
    notes: List[str] = []
    market = market.strip().lower()

    if market == 'ihsg':
        try:
            df = _safe_universe(ExchangeUniverseLoader.load_idx(), ['symbol', 'name'])
            notes.append('Loaded live official IDX stock list.')
            return UniverseLoadResult(market, df, 'IDX official stock list', True, True, notes)
        except Exception as e:
            notes.append(f'Official IDX live load failed: {e}')
        if allow_unverified_third_party:
            try:
                df = _safe_universe(ExchangeUniverseLoader.load_idx_github_snapshot(), ['symbol', 'name'])
                notes.append('Loaded third-party GitHub mirror snapshot because official IDX blocked the request.')
                notes.append('Treat this as convenience only, not authoritative. It may lag current listings or names.')
                return UniverseLoadResult(market, df, 'third-party GitHub mirror snapshot', True, False, notes)
            except Exception as e:
                notes.append(f'Third-party GitHub fallback failed: {e}')
        else:
            notes.append('Third-party IDX fallback is disabled by default to avoid silently showing stale listings.')
        df = _safe_universe(ExchangeUniverseLoader._read_local_csv('ihsg_fallback.csv'), ['symbol', 'name'])
        notes.append('Using bundled local fallback watchlist only. This is not the full IDX universe. Manual symbol entry is still supported.')
        return UniverseLoadResult(market, df, 'bundled local fallback watchlist', False, False, notes)

    if market == 'us_stocks':
        try:
            df = _safe_universe(ExchangeUniverseLoader.load_us_equities(common_only=us_common_only), ['symbol', 'name'])
            notes.append('Loaded live official Nasdaq Trader symbol directory.')
            if us_common_only:
                notes.append('Applied common-only filter to remove many suffix-heavy or special-share lines.')
                complete = False
            else:
                complete = True
            return UniverseLoadResult(market, df, 'Nasdaq Trader symbol directory', True, complete, notes)
        except Exception as e:
            notes.append(f'Official US symbol directory load failed: {e}')
        df = _safe_universe(ExchangeUniverseLoader._read_local_csv('us_fallback.csv'), ['symbol', 'name'])
        notes.append('Using bundled local fallback sample only.')
        return UniverseLoadResult(market, df, 'bundled local fallback sample', False, False, notes)

    if market == 'forex':
        df = _safe_universe(ExchangeUniverseLoader.load_forex_default(), ['symbol', 'name'])
        notes.append('Forex list is a curated major/minor/emerging set, not a complete global provider master list.')
        return UniverseLoadResult(market, df, 'curated Yahoo-compatible forex list', False, False, notes)

    if market == 'commodities':
        df = _safe_universe(ExchangeUniverseLoader.load_commodities_default(), ['symbol', 'name'])
        notes.append('Commodities list is a curated futures core set, not a complete global commodity universe.')
        return UniverseLoadResult(market, df, 'curated Yahoo-compatible commodity futures list', False, False, notes)

    if market == 'crypto':
        try:
            df = _safe_universe(CoinGeckoUniverseLoader.load(), ['symbol', 'name'])
            notes.append('Loaded live CoinGecko coin list for discovery/universe browsing.')
            notes.append('Analysis still uses the symbol you type. For best stability on Yahoo, use symbols like BTC-USD or ETH-USD.')
            return UniverseLoadResult(market, df, 'CoinGecko coins list', True, True, notes)
        except Exception as e:
            notes.append(f'Live CoinGecko coin list failed: {e}')
        df = _safe_universe(CoinGeckoUniverseLoader.load_yahoo_stable_fallback(), ['symbol', 'name'])
        notes.append('Using bundled Yahoo-stable crypto sample list only.')
        return UniverseLoadResult(market, df, 'bundled Yahoo-stable crypto sample', False, False, notes)

    return UniverseLoadResult(market, pd.DataFrame(columns=['symbol', 'name']), 'unknown', False, False, ['Unknown market'])


def format_symbol_for_market(symbol: str, market: str) -> str:
    symbol = (symbol or '').strip().upper()
    if market == 'ihsg' and symbol and not symbol.endswith('.JK'):
        return f'{symbol}.JK'
    return symbol
