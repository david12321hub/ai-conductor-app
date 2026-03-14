# Question
I would like the ai models to search the internet for all available financial data and correlate news events to price changes in assets and commodities and then make a plan for ai agents to make trades to take advantage of the price movements

**Final Result**

**Plan Summary:**
# Unified AI-Powered News-to-Trade System: Complete Implementation Plan

## ⚠️ Critical Disclaimers

```
┌─────────────────────────────────────────────────────────────┐
│                    MANDATORY WARNINGS                        │
├─────────────────────────────────────────────────────────────┤
│ • Trading involves SUBSTANTIAL risk of capital loss          │
│ • Requires regulatory compliance (SEC, FINRA, MiFID II)     │
│ • Algorithmic trading may require broker/exchange approval   │
│ • Past correlations DO NOT guarantee future performance      │
│ • Consult licensed financial and legal advisors FIRST        │
│ • This is an architectural/educational framework             │
└─────────────────────────────────────────────────────────────┘
```

---

## System Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     COMPLETE TRADING SYSTEM                          │
├──────────────┬───────────────┬──────────────┬───────────────────────┤
│  LAYER 1     │  LAYER 2      │  LAYER 3     │  LAYER 4              │
│  DATA        │  ANALYSIS     │  DECISION    │  EXECUTION            │
│              │               │              │                        │
│ • News APIs  │ • NLP/LLM     │ • Strategy   │ • Broker APIs         │
│ • Price Feeds│ • Sentiment   │ • Risk Mgmt  │ • Order Management    │
│ • Social     │ • Correlation │ • Backtesting│ • Position Monitoring │
│ • Economic   │ • ML Models   │ • Agent      │ • P&L Tracking        │
│   Calendar   │ • Causality   │   Planning   │ • Auto Risk Stops     │
├──────────────┴───────────────┴──────────────┴───────────────────────┤
│                    OVERSIGHT & COMPLIANCE LAYER                      │
│         Human Review • Audit Logs • Regulatory Reporting            │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Data Acquisition Layer

### 1A. News & Event Aggregation

```python
# data/news_aggregator.py

import asyncio
import aiohttp
import feedparser
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional
import json

@dataclass
class NewsEvent:
    """Standardized news event across all sources"""
    source: str
    headline: str
    content: str
    timestamp: datetime
    url: str
    assets_mentioned: List[str] = field(default_factory=list)
    event_type: str = "general"        # earnings|macro|geopolitical|regulatory
    sentiment_score: float = 0.0       # -1.0 to 1.0
    impact_score: float = 0.0          # 0.0 to 1.0
    urgency: str = "normal"            # immediate|hours|days|weeks
    raw_data: Dict = field(default_factory=dict)

class NewsAggregator:
    """
    Aggregates financial news from multiple tiers of sources.
    
    Tier 1 (Premium/Fast): Reuters, Bloomberg, Dow Jones
    Tier 2 (Mid):          NewsAPI, Finnhub, Benzinga
    Tier 3 (Free/RSS):     Reuters RSS, WSJ RSS, CNBC RSS
    Tier 4 (Alternative):  Reddit, StockTwits, Twitter/X
    Tier 5 (Official):     SEC EDGAR, Fed Releases, Gov Data
    """

    # ── Source Registry ──────────────────────────────────────────────
    SOURCES = {
        # Tier 2 – API-based (affordable)
        "newsapi":        "https://newsapi.org/v2/everything",
        "finnhub_news":   "https://finnhub.io/api/v1/news",
        "alpha_vantage":  "https://www.alphavantage.co/query",
        "benzinga":       "https://api.benzinga.com/api/v2/news",

        # Tier 3 – Free RSS feeds
        "reuters_rss":    "https://feeds.reuters.com/reuters/businessNews",
        "wsj_rss":        "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
        "cnbc_rss":       "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        "ft_rss":         "https://www.ft.com/rss/home/uk",

        # Tier 5 – Official government/regulatory
        "sec_edgar":      "https://efts.sec.gov/LATEST/search-index",
        "fed_releases":   "https://www.federalreserve.gov/feeds/press_all.xml",
        "fred_api":       "https://api.stlouisfed.org/fred/releases",
        "bls_api":        "https://api.bls.gov/publicAPI/v2/timeseries/data/",
    }

    # Asset keyword mapping for entity extraction
    ASSET_KEYWORDS = {
        "equities":    ["stock", "shares", "equity", "earnings", "EPS", "revenue",
                        "guidance", "IPO", "buyback", "dividend"],
        "commodities": ["oil", "crude", "WTI", "Brent", "gold", "silver", "copper",
                        "wheat", "corn", "soybeans", "natural gas", "LNG"],
        "crypto":      ["bitcoin", "BTC", "ethereum", "ETH", "crypto", "blockchain",
                        "DeFi", "stablecoin", "altcoin"],
        "forex":       ["dollar", "euro", "yen", "pound", "yuan", "currency",
                        "exchange rate", "interest rate", "central bank"],
        "bonds":       ["treasury", "yield", "bond", "debt", "credit", "spread",
                        "Federal Reserve", "ECB", "rate hike", "rate cut"],
        "macro":       ["GDP", "inflation", "CPI", "PPI", "unemployment", "NFP",
                        "PMI", "retail sales", "trade deficit", "recession"],
    }

    # High-impact scheduled events (move markets significantly)
    HIGH_IMPACT_EVENTS = [
        "Federal Reserve rate decision",
        "Non-Farm Payrolls",
        "Consumer Price Index",
        "Producer Price Index",
        "GDP release",
        "OPEC meeting",
        "Earnings release",
        "FDA approval/rejection",
        "Merger announcement",
        "Bankruptcy filing",
    ]

    def __init__(self, api_keys: Dict[str, str]):
        self.api_keys = api_keys
        self.seen_urls = set()          # Deduplication cache
        self.session: Optional[aiohttp.ClientSession] = None

    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *args):
        await self.session.close()

    # ── Main Entry Point ─────────────────────────────────────────────
    async def fetch_all(self) -> List[NewsEvent]:
        """Fetch from all sources concurrently, deduplicate, and return."""
        tasks = [
            self._fetch_newsapi(),
            self._fetch_finnhub(),
            self._fetch_rss_feeds(),
            self._fetch_sec_filings(),
            self._fetch_economic_calendar(),
            self._fetch_fed_releases(),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_events = []
        for result in results:
            if isinstance(result, Exception):
                print(f"Source error: {result}")
                continue
            all_events.extend(result)

        return self._deduplicate(all_events)

    # ── Individual Source Fetchers ────────────────────────────────────
    async def _fetch_newsapi(self) -> List[NewsEvent]:
        """NewsAPI.org – broad financial news coverage"""
        params = {
            "q": "finance OR stocks OR commodities OR economy",
            "language": "en",
            "sortBy": "publishedAt",
            "apiKey": self.api_keys.get("newsapi", ""),
            "pageSize": 100,
        }
        async with self.session.get(self.SOURCES["newsapi"], params=params) as r:
            data = await r.json()

        events = []
        for article in data.get("articles", []):
            if article["url"] in self.seen_urls:
                continue
            self.seen_urls.add(article["url"])
            events.append(NewsEvent(
                source="newsapi",
                headline=article.get("title", ""),
                content=article.get("description", "") + " " +
                        article.get("content", ""),
                timestamp=datetime.fromisoformat(
                    article["publishedAt"].replace("Z", "+00:00")),
                url=article["url"],
                assets_mentioned=self._extract_assets(
                    article.get("title", "") + " " +
                    article.get("description", "")),
            ))
        return events

    async def _fetch_sec_filings(self) -> List[NewsEvent]:
        """
        SEC EDGAR – 8-K filings signal material corporate events.
        8-K types that move prices:
          1.01 – Material agreements
          1.03 – Bankruptcy
          2.02 – Earnings results
          5.02 – Executive changes
          8.01 – Other events
        """
        url = "https://efts.sec.gov/LATEST/search-index?q=%228-K%22&dateRange=custom&startdt=TODAY&forms=8-K"
        async with self.session.get(url) as r:
            data = await r.json()

        events = []
        for filing in data.get("hits", {}).get("hits", []):
            src = filing.get("_source", {})
            events.append(NewsEvent(
                source="sec_edgar",
                headline=f"SEC 8-K: {src.get('entity_name', 'Unknown')} – "
                         f"{src.get('file_date', '')}",
                content=src.get("period_of_report", ""),
                timestamp=datetime.now(),
                url=f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany"
                    f"&CIK={src.get('entity_id', '')}&type=8-K",
                event_type="regulatory",
                urgency="immediate",
                assets_mentioned=[src.get("ticker", "")],
            ))
        return events

    async def _fetch_economic_calendar(self) -> List[NewsEvent]:
        """
        Scheduled macro events – highest market impact.
        Sources: Finnhub economic calendar, Trading Economics API
        """
        url = f"https://finnhub.io/api/v1/calendar/economic"
        params = {"token": self.api_keys.get("finnhub", "")}
        async with self.session.get(url, params=params) as r:
            data = await r.json()

        events = []
        for item in data.get("economicCalendar", []):
            impact = item.get("impact", "low")
            if impact not in ("high", "medium"):
                continue                         # Skip low-impact events

            events.append(NewsEvent(
                source="economic_calendar",
                headline=f"Economic Release: {item.get('event', '')} "
                         f"[{item.get('country', '')}]",
                content=(f"Actual: {item.get('actual', 'TBD')} | "
                         f"Forecast: {item.get('estimate', 'N/A')} | "
                         f"Previous: {item.get('prev', 'N/A')}"),
                timestamp=datetime.fromisoformat(item.get("time", datetime.now().isoformat())),
                url="",
                event_type="macro",
                urgency="immediate" if impact == "high" else "hours",
                impact_score=1.0 if impact == "high" else 0.6,
            ))
        return events

    async def _fetch_rss_feeds(self) -> List[NewsEvent]:
        """Parse free RSS feeds from major financial outlets"""
        rss_sources = {
            "reuters":  self.SOURCES["reuters_rss"],
            "wsj":      self.SOURCES["wsj_rss"],
            "cnbc":     self.SOURCES["cnbc_rss"],
        }
        events = []
        for name, url in rss_sources.items():
            feed = feedparser.parse(url)
            for entry in feed.entries[:20]:
                if entry.link in self.seen_urls:
                    continue
                self.seen_urls.add(entry.link)
                events.append(NewsEvent(
                    source=name,
                    headline=entry.get("title", ""),
                    content=entry.get("summary", ""),
                    timestamp=datetime(*entry.published_parsed[:6])
                              if hasattr(entry, "published_parsed") else datetime.now(),
                    url=entry.link,
                    assets_mentioned=self._extract_assets(
                        entry.get("title", "") + " " + entry.get("summary", "")),
                ))
        return events

    async def _fetch_fed_releases(self) -> List[NewsEvent]:
        """Federal Reserve press releases – critical for rate/policy signals"""
        feed = feedparser.parse(self.SOURCES["fed_releases"])
        events = []
        for entry in feed.entries[:10]:
            events.append(NewsEvent(
                source="federal_reserve",
                headline=entry.get("title", ""),
                content=entry.get("summary", ""),
                timestamp=datetime(*entry.published_parsed[:6])
                          if hasattr(entry, "published_parsed") else datetime.now(),
                url=entry.get("link", ""),
                event_type="macro",
                urgency="immediate",
                impact_score=0.95,
            ))
        return events

    # ── Utilities ─────────────────────────────────────────────────────
    def _extract_assets(self, text: str) -> List[str]:
        """Simple keyword-based asset extraction (NER model preferred in prod)"""
        text_lower = text.lower()
        found = []
        for category, keywords in self.ASSET_KEYWORDS.items():
            if any(kw.lower() in text_lower for kw in keywords):
                found.append(category)
        return found

    def _deduplicate(self, events: List[NewsEvent]) -> List[NewsEvent]:
        """Remove duplicate events by headline similarity"""
        seen_headlines = set()
        unique = []
        for event in events:
            key = event.headline[:80].lower().strip()
            if key not in seen_headlines:
                seen_headlines.add(key)
                unique.append(event)
        return sorted(unique, key=lambda e: e.timestamp, reverse=True)
```

### 1B. Multi-Asset Price Feed

```python
# data/price_feed.py

import pandas as pd
import numpy as np
import yfinance as yf
import websocket
import json
from typing import Dict, List, Callable, Optional
from datetime import datetime, timedelta
import threading

class MultiAssetPriceFeed:
    """
    Unified price feed across all asset classes.
    
    Free/Low-Cost Sources:
    ├── Equities:    Yahoo Finance (yfinance), Alpaca free tier
    ├── Crypto:      Binance WebSocket, Coinbase WebSocket
    ├── Forex:       OANDA free demo, ExchangeRate-API
    ├── Commodities: Yahoo Finance futures, Quandl free tier
    └── Bonds:       FRED API (free), Yahoo Finance
    """

    # Asset universe organized by class
    ASSET_UNIVERSE = {
        "equities": {
            "indices":    ["^GSPC", "^DJI", "^IXIC", "^RUT", "^VIX"],
            "sector_etfs":["XLF", "XLE", "XLK", "XLV", "XLI", "XLB",
                           "XLU", "XLP", "XLY", "XLRE"],
            "mega_caps":  ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
                           "META", "TSLA", "BRK-B", "JPM", "JNJ"],
        },
        "commodities": {
            "energy":     ["CL=F", "BZ=F", "NG=F", "RB=F", "HO=F"],
            "metals":     ["GC=F", "SI=F", "HG=F", "PL=F", "PA=F"],
            "agriculture":["ZW=F", "ZC=F", "ZS=F", "KC=F", "CT=F"],
        },
        "crypto": {
            "major":      ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD"],
            "defi":       ["UNI-USD", "AAVE-USD", "LINK-USD"],
        },
        "forex": {
            "majors":     ["EURUSD=X", "GBPUSD=X", "USDJPY=X",
                           "USDCHF=X", "AUDUSD=X", "USDCAD=X"],
            "em":         ["USDBRL=X", "USDMXN=X", "USDCNY=X"],
        },
        "bonds": {
            "treasuries": ["^TNX", "^TYX", "^FVX", "^IRX"],
            "etfs":       ["TLT", "IEF", "SHY", "HYG", "LQD"],
        },
    }

    def __init__(self):
        self.price_cache: Dict[str, pd.DataFrame] = {}
        self.live_prices: Dict[str, float] = {}
        self.callbacks: List[Callable] = []
        self._ws_connections: List[websocket.WebSocketApp] = []

    # ── Historical Data ───────────────────────────────────────────────
    def get_historical(
        self,
        symbols: List[str],
        period: str = "2y",
        interval: str = "1d",
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV history via yfinance (free, no API key needed).
        
        period options:  1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
        interval options: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo
        """
        result = {}
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                df = ticker.history(period=period, interval=interval)
                df.index = pd.to_datetime(df.index, utc=True)
                df["returns"] = df["Close"].pct_change()
                df["log_returns"] = np.log(df["Close"] / df["Close"].shift(1))
                df["volatility_20d"] = df["returns"].rolling(20).std() * np.sqrt(252)
                result[symbol] = df
                self.price_cache[symbol] = df
            except Exception as e:
                print(f"Failed to fetch {symbol}: {e}")
        return result

    def get_all_assets_history(self) -> Dict[str, pd.DataFrame]:
        """Fetch history for entire asset universe"""
        all_symbols = []
        for asset_class in self.ASSET_UNIVERSE.values():
            for symbols in asset_class.values():
                all_symbols.extend(symbols)
        return self.get_historical(all_symbols)

    # ── Real-Time Crypto WebSocket ────────────────────────────────────
    def start_crypto_stream(self, symbols: List[str] = None):
        """
        Binance WebSocket for real-time crypto prices (free, no auth).
        symbols: ["btcusdt", "ethusdt", ...]
        """
        if symbols is None:
            symbols = ["btcusdt", "ethusdt", "bnbusdt", "solusdt"]

        streams = "/".join([f"{s}@ticker" for s in symbols])
        url = f"wss://stream.binance.com:9443/stream?streams={streams}"

        def on_message(ws, message):
            data = json.loads(message)
            ticker = data.get("data", {})
            symbol = ticker.get("s", "")
            price = float(ticker.get("c", 0))
            self.live_prices[symbol] = price
            for cb in self.callbacks:
                cb(symbol, price, ticker)

        def on_error(ws, error):
            print(f"WebSocket error: {error}")

        ws = websocket.WebSocketApp(url, on_message=on_message, on_error=on_error)
        thread = threading.Thread(target=ws.run_forever, daemon=True)
        thread.start()
        self._ws_connections.append(ws)

    def register_price_callback(self, callback: Callable):
        """Register function to call on every price update"""
        self.callbacks.append(callback)

    # ── Derived Metrics ───────────────────────────────────────────────
    def calculate_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicators used as model features"""
        close = df["Close"]

        # Trend
        df["sma_20"]  = close.rolling(20).mean()
        df["sma_50"]  = close.rolling(50).mean()
        df["sma_200"] = close.rolling(200).mean()
        df["ema_12"]  = close.ewm(span=12).mean()
        df["ema_26"]  = close.ewm(span=26).mean()
        df["macd"]    = df["ema_12"] - df["ema_26"]

        # Momentum
        df["rsi_14"]  = self._rsi(close, 14)
        df["roc_10"]  = close.pct_change(10)

        # Volatility
        df["bb_upper"] = df["sma_20"] + 2 * close.rolling(20).std()
        df["bb_lower"] = df["sma_20"] - 2 * close.rolling(20).std()
        df["atr_14"]   = self._atr(df, 14)

        # Volume
        df["volume_sma_20"] = df["Volume"].rolling(20).mean()
        df["volume_ratio"]  = df["Volume"] / df["volume_sma_20"]

        return df

    @staticmethod
    def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period).mean()
        rs    = gain / loss
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high, low, close = df["High"], df["Low"], df["Close"]
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low  - close.shift()).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean()
```

---

## Phase 2: Analysis Layer

### 2A. LLM News Analysis Engine

```python
# analysis/llm_analyzer.py

import json
import asyncio
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from typing import List, Dict, Optional
from data.news_aggregator import NewsEvent

class LLMNewsAnalyzer:
    """
    Multi-model LLM analysis pipeline for financial news.
    
    Model Selection Strategy:
    ├── GPT-4o:        Complex multi-asset analysis, earnings calls
    ├── GPT-4o-mini:   High-volume routine news (cost efficient)
    ├── Claude 3.5:    Fed statements, regulatory analysis (nuanced)
    └── FinBERT:       Bulk sentiment scoring (local, fast, free)
    """

    SYSTEM_PROMPT = """You are a senior quantitative analyst at a top-tier hedge fund 
    with 20 years of experience trading equities, commodities, forex, and derivatives.
    
    Your role is to analyze financial news and extract precise, actionable trading signals.
    
    Key principles:
    - Focus on SURPRISE vs EXPECTATION (markets price in expectations)
    - Consider second and third-order effects across asset classes
    - Distinguish between immediate (minutes), short-term (hours/days), 
      and medium-term (weeks) impacts
    - Be conservative: only flag HIGH confidence signals
    - Always identify what could make your thesis WRONG
    - Consider market microstructure (liquidity, time of day, options expiry)"""

    def __init__(self, openai_key: str, anthropic_key: str = None):
        self.openai  = AsyncOpenAI(api_key=openai_key)
        self.claude  = AsyncAnthropic(api_key=anthropic_key) if anthropic_key else None

    # ── Core Analysis ─────────────────────────────────────────────────
    async def analyze_event(self, event: NewsEvent) -> Dict:
        """
        Full LLM analysis of a news event.
        Returns structured trading intelligence.
        """
        prompt = self._build_analysis_prompt(event)

        response = await self.openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,        # Low temp = consistent, analytical output
            max_tokens=2000,
        )

        analysis = json.loads(response.choices[0].message.content)
        analysis["event_id"]  = id(event)
        analysis["timestamp"] = event.timestamp.isoformat()
        analysis["source"]    = event.source
        return analysis

    async def analyze_batch(
        self,
        events: List[NewsEvent],
        max_concurrent: int = 5,
    ) -> List[Dict]:
        """Process multiple events concurrently with rate limiting"""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def analyze_with_limit(event):
            async with semaphore:
                try:
                    return await self.analyze_event(event)
                except Exception as e:
                    print(f"Analysis failed for '{event.headline[:50]}': {e}")
                    return None

        results = await asyncio.gather(
            *[analyze_with_limit(e) for e in events]
        )
        return [r for r in results if r is not None]

    async def analyze_fed_statement(self, statement_text: str) -> Dict:
        """
        Specialized Fed analysis – highest market impact event.
        Detects: hawkish/dovish shift, forward guidance changes,
                 balance sheet signals, dot plot interpretation.
        """
        prompt = f"""
        Analyze this Federal Reserve statement for monetary policy signals:
        
        STATEMENT:
        {statement_text}
        
        Provide analysis in JSON:
        {{
            "policy_stance": "hawkish|dovish|neutral",
            "stance_change": "more_hawkish|more_dovish|unchanged",
            "rate_path_signal": "hikes|cuts|hold|uncertain",
            "key_phrases": ["list of market-moving phrases"],
            "balance_sheet_signal": "tightening|easing|unchanged",
            "inflation_assessment": "concerned|less_concerned|neutral",
            "growth_assessment": "optimistic|pessimistic|neutral",
            "asset_impacts": {{
                "equities":    {{"direction": "up|down", "magnitude": "large|medium|small", "reasoning": ""}},
                "bonds":       {{"direction": "up|down", "magnitude": "large|medium|small", "reasoning": ""}},
                "gold":        {{"direction": "up|down", "magnitude": "large|medium|small", "reasoning": ""}},
                "usd":         {{"direction": "up|down", "magnitude": "large|medium|small", "reasoning": ""}},
                "commodities": {{"direction": "up|down", "magnitude": "large|medium|small", "reasoning": ""}}
            }},
            "surprise_factor": 0.0,
            "confidence": 0.0,
            "key_risks": ["list of risks to this thesis"]
        }}
        """
        response = await self.openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.05,
        )
        return json.loads(response.choices[0].message.content)

    async def analyze_earnings(self, earnings_data: Dict) -> Dict:
        """
        Earnings analysis – compare actual vs estimates.
        Key metrics: EPS beat/miss, revenue beat/miss, guidance change.
        """
        prompt = f"""
        Analyze this earnings report vs Wall Street expectations:
        
        COMPANY: {earnings_data.get('company')} ({earnings_data.get('ticker')})
        
        RESULTS:
        - EPS Actual:      {earnings_data.get('eps_actual')}
        - EPS Estimate:    {earnings_data.get('eps_estimate')}
        - Revenue Actual:  {earnings_data.get('revenue_actual')}
        - Revenue Est:     {earnings_data.get('revenue_estimate')}
        - Guidance:        {earnings_data.get('guidance', 'Not provided')}
        - Key Comments:    {earnings_data.get('management_comments', '')}
        
        Provide JSON analysis:
        {{
            "eps_surprise_pct": 0.0,
            "revenue_surprise_pct": 0.0,
            "guidance_change": "raised|lowered|maintained|withdrawn|none",
            "overall_quality": "beat|miss|inline",
            "expected_price_reaction": {{
                "direction": "up|down",
                "magnitude_pct": 0.0,
                "confidence": 0.0
            }},
            "sector_read_through": [
                {{"company": "", "impact": "positive|negative|neutral", "reasoning": ""}}
            ],
            "key_risks": [],
            "time_horizon": "immediate|short_term"
        }}
        """
        response = await self.openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        return json.loads(response.choices[0].message.content)

    async def find_cross_asset_effects(
        self,
        primary_analysis: Dict,
        asset_universe: List[str],
    ) -> Dict:
        """
        Identify second-order effects across asset classes.
        Example: Oil spike → airlines down, defense up, USD up, gold up
        """
        prompt = f"""
        Given this primary market event analysis:
        {json.dumps(primary_analysis, indent=2

**Code Agent Output:**
```python
# Unified AI-Powered News-to-Trade System

```python
"""
================================================================================
UNIFIED AI-POWERED NEWS-TO-TRADE SYSTEM
================================================================================

⚠️  MANDATORY DISCLAIMERS:
    • Trading involves SUBSTANTIAL risk of capital loss
    • Requires regulatory compliance (SEC, FINRA, MiFID II)
    • Algorithmic trading may require broker/exchange approval
    • Past correlations DO NOT guarantee future performance
    • Consult licensed financial and legal advisors FIRST
    • This is an architectural/educational framework ONLY

================================================================================
"""

# ── Standard Library ──────────────────────────────────────────────────────────
import asyncio
import json
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

# ── Third-Party (install via requirements.txt) ────────────────────────────────
import aiohttp
import feedparser
import numpy as np
import pandas as pd
import websocket
import yfinance as yf
from anthropic import AsyncAnthropic
from openai import AsyncOpenAI

# ── Logging Configuration ─────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("news_to_trade")


# ==============================================================================
# SHARED ENUMERATIONS & CONSTANTS
# ==============================================================================

class OrderSide(str, Enum):
    BUY  = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET      = "market"
    LIMIT       = "limit"
    STOP        = "stop"
    STOP_LIMIT  = "stop_limit"


class OrderStatus(str, Enum):
    PENDING   = "pending"
    SUBMITTED = "submitted"
    FILLED    = "filled"
    CANCELLED = "cancelled"
    REJECTED  = "rejected"


class Urgency(str, Enum):
    IMMEDIATE = "immediate"
    HOURS     = "hours"
    DAYS      = "days"
    WEEKS     = "weeks"


class PolicyStance(str, Enum):
    HAWKISH = "hawkish"
    DOVISH  = "dovish"
    NEUTRAL = "neutral"


class Direction(str, Enum):
    UP   = "up"
    DOWN = "down"
    FLAT = "flat"


# ==============================================================================
# LAYER 1 — DATA ACQUISITION
# ==============================================================================

# ── 1A. News & Event Aggregation ──────────────────────────────────────────────

@dataclass
class NewsEvent:
    """
    Standardised news event produced by every source adapter.
    All downstream components consume this single schema.
    """
    source:           str
    headline:         str
    content:          str
    timestamp:        datetime
    url:              str
    assets_mentioned: List[str]        = field(default_factory=list)
    event_type:       str              = "general"   # earnings|macro|geopolitical|regulatory
    sentiment_score:  float            = 0.0         # -1.0 → 1.0
    impact_score:     float            = 0.0         # 0.0 → 1.0
    urgency:          str              = Urgency.HOURS
    raw_data:         Dict[str, Any]   = field(default_factory=dict)
    event_id:         str              = field(default_factory=lambda: str(uuid.uuid4()))


class NewsAggregator:
    """
    Aggregates financial news from multiple source tiers concurrently.

    Tier 1 (Premium/Fast) : Reuters, Bloomberg, Dow Jones
    Tier 2 (Mid)          : NewsAPI, Finnhub, Benzinga
    Tier 3 (Free/RSS)     : Reuters RSS, WSJ RSS, CNBC RSS
    Tier 4 (Alternative)  : Reddit, StockTwits, Twitter/X
    Tier 5 (Official)     : SEC EDGAR, Fed Releases, Gov Data
    """

    # Source registry
    SOURCES: Dict[str, str] = {
        "newsapi":       "https://newsapi.org/v2/everything",
        "finnhub_news":  "https://finnhub.io/api/v1/news",
        "alpha_vantage": "https://www.alphavantage.co/query",
        "reuters_rss":   "https://feeds.reuters.com/reuters/businessNews",
        "wsj_rss":       "https://feeds.a.dj.com/rss/RSSMarketsMain.xml",
        "cnbc_rss":      "https://www.cnbc.com/id/100003114/device/rss/rss.html",
        "sec_edgar":     "https://efts.sec.gov/LATEST/search-index",
        "fed_releases":  "https://www.federalreserve.gov/feeds/press_all.xml",
        "finnhub_econ":  "https://finnhub.io/api/v1/calendar/economic",
    }

    # Keyword → asset-class mapping (NER model preferred in production)
    ASSET_KEYWORDS: Dict[str, List[str]] = {
        "equities":    ["stock", "shares", "equity", "earnings", "EPS",
                        "revenue", "guidance", "IPO", "buyback", "dividend"],
        "commodities": ["oil", "crude", "WTI", "Brent", "gold", "silver",
                        "copper", "wheat", "corn", "natural gas"],
        "crypto":      ["bitcoin", "BTC", "ethereum", "ETH", "crypto",
                        "blockchain", "DeFi", "stablecoin"],
        "forex":       ["dollar", "euro", "yen", "pound", "yuan",
                        "exchange rate", "interest rate", "central bank"],
        "bonds":       ["treasury", "yield", "bond", "debt", "credit",
                        "Federal Reserve", "ECB", "rate hike", "rate cut"],
        "macro":       ["GDP", "inflation", "CPI", "PPI", "unemployment",
                        "NFP", "PMI", "retail sales", "recession"],
    }

    def __init__(self, api_keys: Dict[str, str]) -> None:
        self.api_keys      = api_keys
        self._seen_urls:   set = set()
        self._session:     Optional[aiohttp.ClientSession] = None
        self._log          = logging.getLogger(self.__class__.__name__)

    # ── Context manager ───────────────────────────────────────────────────────

    async def __aenter__(self) -> "NewsAggregator":
        self._session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._session:
            await self._session.close()

    # ── Public API ────────────────────────────────────────────────────────────

    async def fetch_all(self) -> List[NewsEvent]:
        """
        Fetch from every source concurrently, deduplicate, and return
        events sorted newest-first.
        """
        tasks = [
            self._fetch_newsapi(),
            self._fetch_finnhub_news(),
            self._fetch_rss_feeds(),
            self._fetch_sec_filings(),
            self._fetch_economic_calendar(),
            self._fetch_fed_releases(),
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        all_events: List[NewsEvent] = []
        for result in results:
            if isinstance(result, Exception):
                self._log.warning("Source error: %s", result)
                continue
            all_events.extend(result)

        return self._deduplicate(all_events)

    # ── Source adapters ───────────────────────────────────────────────────────

    async def _fetch_newsapi(self) -> List[NewsEvent]:
        """NewsAPI.org — broad financial news coverage."""
        params = {
            "q":        "finance OR stocks OR commodities OR economy",
            "language": "en",
            "sortBy":   "publishedAt",
            "apiKey":   self.api_keys.get("newsapi", ""),
            "pageSize": 100,
        }
        async with self._session.get(self.SOURCES["newsapi"], params=params) as resp:
            data = await resp.json()

        events: List[NewsEvent] = []
        for article in data.get("articles", []):
            url = article.get("url", "")
            if url in self._seen_urls:
                continue
            self._seen_urls.add(url)

            text = f"{article.get('title', '')} {article.get('description', '')}"
            events.append(NewsEvent(
                source    = "newsapi",
                headline  = article.get("title", ""),
                content   = (article.get("description", "") + " " +
                             article.get("content", "")),
                timestamp = datetime.fromisoformat(
                    article["publishedAt"].replace("Z", "+00:00")),
                url       = url,
                assets_mentioned = self._extract_assets(text),
            ))
        return events

    async def _fetch_finnhub_news(self) -> List[NewsEvent]:
        """Finnhub general market news."""
        params = {
            "category": "general",
            "token":    self.api_keys.get("finnhub", ""),
        }
        async with self._session.get(
            self.SOURCES["finnhub_news"], params=params
        ) as resp:
            data = await resp.json()

        events: List[NewsEvent] = []
        for item in data if isinstance(data, list) else []:
            url = item.get("url", "")
            if url in self._seen_urls:
                continue
            self._seen_urls.add(url)

            text = f"{item.get('headline', '')} {item.get('summary', '')}"
            events.append(NewsEvent(
                source    = "finnhub",
                headline  = item.get("headline", ""),
                content   = item.get("summary", ""),
                timestamp = datetime.fromtimestamp(item.get("datetime", 0)),
                url       = url,
                assets_mentioned = self._extract_assets(text),
            ))
        return events

    async def _fetch_rss_feeds(self) -> List[NewsEvent]:
        """Parse free RSS feeds from major financial outlets."""
        rss_sources = {
            "reuters": self.SOURCES["reuters_rss"],
            "wsj":     self.SOURCES["wsj_rss"],
            "cnbc":    self.SOURCES["cnbc_rss"],
        }
        events: List[NewsEvent] = []
        for name, url in rss_sources.items():
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:20]:
                    link = getattr(entry, "link", "")
                    if link in self._seen_urls:
                        continue
                    self._seen_urls.add(link)

                    published = (
                        datetime(*entry.published_parsed[:6])
                        if hasattr(entry, "published_parsed")
                        else datetime.utcnow()
                    )
                    text = f"{entry.get('title', '')} {entry.get('summary', '')}"
                    events.append(NewsEvent(
                        source    = name,
                        headline  = entry.get("title", ""),
                        content   = entry.get("summary", ""),
                        timestamp = published,
                        url       = link,
                        assets_mentioned = self._extract_assets(text),
                    ))
            except Exception as exc:
                self._log.warning("RSS feed %s failed: %s", name, exc)
        return events

    async def _fetch_sec_filings(self) -> List[NewsEvent]:
        """
        SEC EDGAR 8-K filings — signal material corporate events.

        High-impact 8-K items:
            1.01 Material agreements | 1.03 Bankruptcy
            2.02 Earnings results    | 5.02 Executive changes
        """
        url = (
            "https://efts.sec.gov/LATEST/search-index"
            "?q=%228-K%22&dateRange=custom&startdt=TODAY&forms=8-K"
        )
        async with self._session.get(url) as resp:
            data = await resp.json()

        events: List[NewsEvent] = []
        for filing in data.get("hits", {}).get("hits", []):
            src = filing.get("_source", {})
            ticker = src.get("ticker", "")
            events.append(NewsEvent(
                source    = "sec_edgar",
                headline  = (f"SEC 8-K: {src.get('entity_name', 'Unknown')} — "
                             f"{src.get('file_date', '')}"),
                content   = src.get("period_of_report", ""),
                timestamp = datetime.utcnow(),
                url       = (f"https://www.sec.gov/cgi-bin/browse-edgar"
                             f"?action=getcompany&CIK={src.get('entity_id', '')}"
                             f"&type=8-K"),
                event_type       = "regulatory",
                urgency          = Urgency.IMMEDIATE,
                assets_mentioned = [ticker] if ticker else [],
            ))
        return events

    async def _fetch_economic_calendar(self) -> List[NewsEvent]:
        """
        Scheduled macro releases — highest market-moving potential.
        Only medium/high-impact events are retained.
        """
        params = {"token": self.api_keys.get("finnhub", "")}
        async with self._session.get(
            self.SOURCES["finnhub_econ"], params=params
        ) as resp:
            data = await resp.json()

        events: List[NewsEvent] = []
        for item in data.get("economicCalendar", []):
            impact = item.get("impact", "low")
            if impact not in ("high", "medium"):
                continue

            events.append(NewsEvent(
                source    = "economic_calendar",
                headline  = (f"Economic Release: {item.get('event', '')} "
                             f"[{item.get('country', '')}]"),
                content   = (f"Actual: {item.get('actual', 'TBD')} | "
                             f"Forecast: {item.get('estimate', 'N/A')} | "
                             f"Previous: {item.get('prev', 'N/A')}"),
                timestamp = datetime.fromisoformat(
                    item.get("time", datetime.utcnow().isoformat())),
                url          = "",
                event_type   = "macro",
                urgency      = Urgency.IMMEDIATE if impact == "high" else Urgency.HOURS,
                impact_score = 1.0 if impact == "high" else 0.6,
            ))
        return events

    async def _fetch_fed_releases(self) -> List[NewsEvent]:
        """Federal Reserve press releases — critical for rate/policy signals."""
        feed   = feedparser.parse(self.SOURCES["fed_releases"])
        events: List[NewsEvent] = []
        for entry in feed.entries[:10]:
            published = (
                datetime(*entry.published_parsed[:6])
                if hasattr(entry, "published_parsed")
                else datetime.utcnow()
            )
            events.append(NewsEvent(
                source       = "federal_reserve",
                headline     = entry.get("title", ""),
                content      = entry.get("summary", ""),
                timestamp    = published,
                url          = entry.get("link", ""),
                event_type   = "macro",
                urgency      = Urgency.IMMEDIATE,
                impact_score = 0.95,
            ))
        return events

    # ── Utilities ─────────────────────────────────────────────────────────────

    def _extract_assets(self, text: str) -> List[str]:
        """Keyword-based asset extraction (replace with NER model in production)."""
        text_lower = text.lower()
        return [
            category
            for category, keywords in self.ASSET_KEYWORDS.items()
            if any(kw.lower() in text_lower for kw in keywords)
        ]

    def _deduplicate(self, events: List[NewsEvent]) -> List[NewsEvent]:
        """Remove near-duplicate events by headline prefix, sort newest-first."""
        seen:   set             = set()
        unique: List[NewsEvent] = []
        for event in events:
            key = event.headline[:80].lower().strip()
            if key not in seen:
                seen.add(key)
                unique.append(event)
        return sorted(unique, key=lambda e: e.timestamp, reverse=True)


# ── 1B. Multi-Asset Price Feed ────────────────────────────────────────────────

class MultiAssetPriceFeed:
    """
    Unified OHLCV + real-time price feed across all asset classes.

    Free/low-cost sources:
        Equities    — Yahoo Finance (yfinance), Alpaca free tier
        Crypto      — Binance WebSocket (no auth required)
        Forex       — Yahoo Finance
        Commodities — Yahoo Finance futures
        Bonds       — FRED API, Yahoo Finance
    """

    ASSET_UNIVERSE: Dict[str, Dict[str, List[str]]] = {
        "equities": {
            "indices":     ["^GSPC", "^DJI", "^IXIC", "^RUT", "^VIX"],
            "sector_etfs": ["XLF", "XLE", "XLK", "XLV", "XLI",
                            "XLB", "XLU", "XLP", "XLY", "XLRE"],
            "mega_caps":   ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
                            "META", "TSLA", "BRK-B", "JPM", "JNJ"],
        },
        "commodities": {
            "energy":      ["CL=F", "BZ=F", "NG=F"],
            "metals":      ["GC=F", "SI=F", "HG=F"],
            "agriculture": ["ZW=F", "ZC=F", "ZS=F"],
        },
        "crypto": {
            "major": ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD"],
        },
        "forex": {
            "majors": ["EURUSD=X", "GBPUSD=X", "USDJPY=X",
                       "USDCHF=X", "AUDUSD=X", "USDCAD=X"],
        },
        "bonds": {
            "treasuries": ["^TNX", "^TYX", "^FVX", "^IRX"],
            "etfs":       ["TLT", "IEF", "SHY", "HYG", "LQD"],
        },
    }

    def __init__(self) -> None:
        self.price_cache:  Dict[str, pd.DataFrame] = {}
        self.live_prices:  Dict[str, float]         = {}
        self._callbacks:   List[Callable]            = []
        self._ws_threads:  List[threading.Thread]    = []
        self._log          = logging.getLogger(self.__class__.__name__)

    # ── Historical data ───────────────────────────────────────────────────────

    def get_historical(
        self,
        symbols: List[str],
        period:   str = "2y",
        interval: str = "1d",
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch OHLCV history via yfinance (free, no API key required).

        period   : 1d 5d 1mo 3mo 6mo 1y 2y 5y 10y ytd max
        interval : 1m 2m 5m 15m 30m 60m 90m 1h 1d 5d 1wk 1mo
        """
        result: Dict[str, pd.DataFrame] = {}
        for symbol in symbols:
            try:
                df = yf.Ticker(symbol).history(period=period, interval=interval)
                df.index          = pd.to_datetime(df.index, utc=True)
                df["returns"]     = df["Close"].pct_change()
                df["log_returns"] = np.log(df["Close"] / df["Close"].shift(1))
                df["vol_20d"]     = df["returns"].rolling(20).std() * np.sqrt(252)
                df                = self.add_technical_indicators(df)
                result[symbol]    = df
                self.price_cache[symbol] = df
                self._log.debug("Fetched %s (%d rows)", symbol, len(df))
            except Exception as exc:
                self._log.warning("Failed to fetch %s: %s", symbol, exc)
        return result

    def get_full_universe(self) -> Dict[str, pd.DataFrame]:
        """Fetch history for the entire asset universe."""
        symbols = [
            sym
            for asset_class in self.ASSET_UNIVERSE.values()
            for group in asset_class.values()
            for sym in group
        ]
        return self.get_historical(symbols)

    # ── Real-time crypto stream ───────────────────────────────────────────────

    def start_crypto_stream(self, symbols: Optional[List[str]] = None) -> None:
        """
        Binance WebSocket for real-time crypto prices (free, no auth).
        symbols: lowercase Binance pairs, e.g. ["btcusdt", "ethusdt"]
        """
        symbols = symbols or ["btcusdt", "ethusdt", "bnbusdt", "solusdt"]
        streams = "/".join(f"{s}@ticker" for s in symbols)
        url     = f"wss://stream.binance.com:9443/stream?streams={streams}"

        def on_message(_ws: Any, message: str) -> None:
            payload = json.loads(message).get("data", {})
            symbol  = payload.get("s", "")
            price   = float(payload.get("c", 0))
            self.live_prices[symbol] = price
            for cb in self._callbacks:
                try:
                    cb(symbol, price, payload)
                except Exception as exc:
                    self._log.warning("Price callback error: %s", exc)

        def on_error(_ws: Any, error: Any) -> None:
            self._log.error("Crypto WebSocket error: %s", error)

        ws     = websocket.WebSocketApp(url, on_message=on_message, on_error=on_error)
        thread = threading.Thread(target=ws.run_forever, daemon=True)
        thread.start()
        self._ws_threads.append(thread)
        self._log.info("Crypto stream started for: %s", symbols)

    def register_price_callback(self, callback: Callable) -> None:
        """Register a function called on every live price tick."""
        self._callbacks.append(callback)

    # ── Technical indicators ──────────────────────────────────────────────────

    @staticmethod
    def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute a standard set of technical indicators used as model features.
        Operates on a DataFrame with columns: Open, High, Low, Close, Volume.
        """
        close = df["Close"]

        # Trend
        df["sma_20"]  = close.rolling(20).mean()
        df["sma_50"]  = close.rolling(50).mean()
        df["sma_200"] = close.rolling(200).mean()
        df["ema_12"]  = close.ewm(span=12, adjust=False).mean()
        df["ema_26"]  = close.ewm(span=26, adjust=False).mean()
        df["macd"]    = df["ema_12"] - df["ema_26"]

        # Momentum
        df["rsi_14"] = MultiAssetPriceFeed._rsi(close, 14)
        df["roc_10"] = close.pct_change(10)

        # Volatility / Bollinger Bands
        rolling_std    = close.rolling(20).std()
        df["bb_upper"] = df["sma_20"] + 2 * rolling_std
        df["bb_lower"] = df["sma_20"] - 2 * rolling_std
        df["atr_14"]   = MultiAssetPriceFeed._atr(df, 14)

        # Volume
        df["vol_sma_20"]  = df["Volume"].rolling(20).mean()
        df["volume_ratio"] = df["Volume"] / df["vol_sma_20"].replace(0, np.nan)

        return df

    @staticmethod
    def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain  = delta.clip(lower=0).rolling(period).mean()
        loss  = (-delta.clip(upper=0)).rolling(period).mean()
        rs    = gain / loss.replace(0, np.nan)
        return 100 - (100 / (1 + rs))

    @staticmethod
    def _atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high, low, prev_close = df["High"], df["Low"], df["Close"].shift(1)
        true_range = pd.concat([
            high - low,
            (high - prev_close).abs(),
            (low  - prev_close).abs(),
        ], axis=1).max(axis=1)
        return true_range.rolling(period).mean()


# ==============================================================================
# LAYER 2 — ANALYSIS
# ==============================================================================

# ── 2A. LLM News Analysis Engine ──────────────────────────────────────────────

@dataclass
class TradeSignal:
    """
    Structured trading signal produced by the LLM analysis pipeline.
    Consumed by the Strategy and Risk layers.
    """
    event_id:         str
    timestamp:        datetime
    source_headline:  str
    asset_class:      str
    specific_assets:  List[str]
    direction:        str                    # up | down | flat
    confidence:       float                  # 0.0 → 1.0
    time_horizon:     str                    # immediate | short_term | medium_term
    expected_move_pct: float
    reasoning:        str
    risks:            List[str]
    raw_analysis:     Dict[str, Any]         = field(default_factory=dict)
    signal_id:        str                    = field(
        default_factory=lambda: str(uuid.uuid4()))


class LLMNewsAnalyzer:
    """
    Multi-model LLM analysis pipeline for financial news.

    Model routing:
        GPT-4o       — complex multi-asset analysis, earnings calls
        GPT-4o-mini  — high-volume routine news (cost-efficient)
        Claude 3.5   — Fed statements, regulatory nuance
    """

    _SYSTEM_PROMPT = (
        "You are a senior quantitative analyst at a top-tier hedge fund with "
        "20 years of experience trading equities, commodities, forex, and "
        "derivatives.\n\n"
        "Key principles:\n"
        "- Focus on SURPRISE vs EXPECTATION (markets price in expectations)\n"
        "- Consider second and third-order effects across asset classes\n"
        "- Distinguish immediate (minutes), short-term (hours/days), and "
        "medium-term (weeks) impacts\n"
        "- Only flag HIGH-confidence signals\n"
        "- Always identify what could make your thesis WRONG\n"
        "- Consider market microstructure (liquidity, time of day, options expiry)"
    )

    def __init__(
        self,
        openai_key:    str,
        anthropic_key: Optional[str] = None,
    ) -> None:
        self._openai  = AsyncOpenAI(api_key=openai_key)
        self._claude  = AsyncAnthropic(api_key=anthropic_key) if anthropic_key else None
        self._log     = logging.getLogger(self.__class__.__name__)

    # ── Core analysis ─────────────────────────────────────────────────────────

    async def analyze_event(self, event: NewsEvent) -> Optional[Dict[str, Any]]:
        """
        Full LLM analysis of a single news event.
        Returns a structured dict of trading intelligence or None on failure.
        """
        prompt = self._build_event_prompt(event)
        try:
            response = await self._openai.chat.completions.create(
                model           = "gpt-4o",
                messages        = [
                    {"role": "system", "content": self._SYSTEM_PROMPT},
                    {"role": "user",   "content": prompt},
                ],
                response_format = {"type": "json_object"},
                temperature     = 0.1,
                max_tokens      = 2000,
            )
            analysis                = json.loads(response.choices[0].message.content)
            analysis["event_id"]    = event.event_id
            analysis["timestamp"]   = event.timestamp.isoformat()
            analysis["source"]      = event.source
            return analysis
        except Exception as exc:
            self._log.error("analyze_event failed: %s", exc)
            return None

    async def analyze_batch(
        self,
        events:          List[NewsEvent],
        max_concurrent:  int = 5,
    ) -> List[Dict[str, Any]]:
        """Process multiple events concurrently with a concurrency cap."""
        semaphore = asyncio.Semaphore(max_concurrent)

        async def _bounded(event: NewsEvent) -> Optional[Dict[str, Any]]:
            async with semaphore:
                return await self.analyze_event(event)

        results = await asyncio.gather(*[_bounded(e) for e in events])
        return [r for r in results if r is not None]

    async def analyze_fed_statement(self, statement_text: str) -> Dict[str, Any]:
        """
        Specialised Federal Reserve statement analysis.
        Detects hawkish/dovish shifts, forward guidance, balance-sheet signals.
        """
        prompt = f"""
Analyse this Federal Reserve statement for monetary policy signals.

STATEMENT:
{statement_text}

Return JSON with this exact schema:
{{
    "policy_stance":        "hawkish|dovish|neutral",
    "stance_change":        "more_hawkish|more_dovish|unchanged",
    "rate_path_signal":     "hikes|cuts|hold|uncertain",
    "key_phrases":          ["..."],
    "balance_sheet_signal": "tightening|easing|unchanged",
    "inflation_assessment": "concerned|less_concerned|neutral",
    "growth_assessment":    "optimistic|pessimistic|neutral",
    "asset_impacts": {{
        "equities":    {{"direction": "up|down", "magnitude": "large|medium|small", "reasoning": ""}},
        "bonds":       {{"direction": "up|down", "magnitude": "large|medium|small", "reasoning": ""}},
        "gold":        {{"direction": "up|down", "magnitude": "large|medium|small", "reasoning": ""}},
        "usd":         {{"direction": "up|down", "magnitude": "large|medium|small", "reasoning": ""}},
        "commodities": {{"direction": "up|down", "magnitude": "large|medium|small", "reasoning": ""}}
    }},
    "surprise_factor": 0.0
```

**Planning Agent Output:**
# Unified AI-Powered News-to-Trade System: Detailed Implementation Steps

---

## ⚠️ MANDATORY WARNINGS BEFORE PROCEEDING

1. **Acknowledge financial risk**: Trading involves substantial risk of capital loss. Never trade with money you cannot afford to lose entirely.
2. **Seek professional advice**: Consult a licensed financial advisor and legal counsel before building or deploying any trading system.
3. **Understand regulatory requirements**: Research SEC, FINRA, MiFID II, and any other applicable regulations in your jurisdiction before proceeding.
4. **Obtain necessary approvals**: Contact your broker and any relevant exchanges to confirm algorithmic trading is permitted under your account agreement.
5. **Treat this as educational**: This framework is architectural and educational in nature. Past correlations do not guarantee future performance.

---

## SECTION 1: SYSTEM ARCHITECTURE SETUP

### Layer Planning

6. **Define your four system layers** before writing any code:
   - Layer 1 (Data): News APIs, price feeds, social data, economic calendars
   - Layer 2 (Analysis): NLP/LLM processing, sentiment scoring, ML models, correlation analysis
   - Layer 3 (Decision): Strategy logic, risk management, backtesting, agent planning
   - Layer 4 (Execution): Broker API connections, order management, position monitoring, P&L tracking

7. **Plan your oversight layer**: Design human review checkpoints, audit logging, and regulatory reporting mechanisms before building any automated execution.

8. **Create your project directory structure**:
   ```
   trading_system/
   ├── data/
   │   ├── news_aggregator.py
   │   └── price_feed.py
   ├── analysis/
   │   └── llm_analyzer.py
   ├── decision/
   ├── execution/
   ├── oversight/
   ├── tests/
   └── config/
   ```

9. **Set up a Python virtual environment**:
   ```bash
   python -m venv trading_env
   source trading_env/bin/activate  # Linux/Mac
   trading_env\Scripts\activate     # Windows
   ```

10. **Install core dependencies**:
    ```bash
    pip install aiohttp feedparser pandas numpy yfinance websocket-client openai anthropic asyncio
    ```

11. **Create a secure configuration file** (`config/keys.yaml`) to store all API keys, and add it immediately to `.gitignore` to prevent accidental exposure.

12. **Set up version control**:
    ```bash
    git init
    echo "config/keys.yaml" >> .gitignore
    echo "*.env" >> .gitignore
    git add .
    git commit -m "Initial project structure"
    ```

---

## SECTION 2: PHASE 1A — NEWS AGGREGATION LAYER

### Step 2A-1: Define the NewsEvent Data Structure

13. **Open** `data/news_aggregator.py` and import all required libraries:
    ```python
    import asyncio
    import aiohttp
    import feedparser
    from dataclasses import dataclass, field
    from datetime import datetime
    from typing import List, Dict, Optional
    import json
    ```

14. **Define the `NewsEvent` dataclass** to standardize all incoming news regardless of source:
    - `source`: Which outlet or API provided the event
    - `headline`: The article or filing title
    - `content`: Full or summarized body text
    - `timestamp`: When the event occurred (always use UTC)
    - `url`: Original source link for audit trail
    - `assets_mentioned`: List of asset categories detected in the text
    - `event_type`: Classify as `general`, `earnings`, `macro`, `geopolitical`, or `regulatory`
    - `sentiment_score`: Float from -1.0 (very negative) to +1.0 (very positive)
    - `impact_score`: Float from 0.0 (negligible) to 1.0 (extreme market impact)
    - `urgency`: One of `immediate`, `hours`, `days`, or `weeks`
    - `raw_data`: Dictionary to store the original API response for debugging

15. **Apply `field(default_factory=list)`** to all list and dict fields to avoid Python's mutable default argument pitfall.

### Step 2A-2: Define the NewsAggregator Class Constants

16. **Define the `SOURCES` dictionary** mapping source names to their base URLs, organized into five tiers:
    - Tier 1 (Premium/Fast): Reuters, Bloomberg, Dow Jones — note these require expensive subscriptions
    - Tier 2 (Mid-cost API): NewsAPI, Finnhub, Alpha Vantage, Benzinga
    - Tier 3 (Free RSS): Reuters RSS, WSJ RSS, CNBC RSS, Financial Times RSS
    - Tier 4 (Alternative/Social): Reddit, StockTwits, Twitter/X — plan for rate limits and noise
    - Tier 5 (Official Government): SEC EDGAR, Federal Reserve releases, FRED API, BLS API

17. **Define the `ASSET_KEYWORDS` dictionary** mapping asset categories to trigger keywords:
    - `equities`: stock, shares, equity, earnings, EPS, revenue, guidance, IPO, buyback, dividend
    - `commodities`: oil, crude, WTI, Brent, gold, silver, copper, wheat, corn, soybeans, natural gas, LNG
    - `crypto`: bitcoin, BTC, ethereum, ETH, crypto, blockchain, DeFi, stablecoin, altcoin
    - `forex`: dollar, euro, yen, pound, yuan, currency, exchange rate, interest rate, central bank
    - `bonds`: treasury, yield, bond, debt, credit, spread, Federal Reserve, ECB, rate hike, rate cut
    - `macro`: GDP, inflation, CPI, PPI, unemployment, NFP, PMI, retail sales, trade deficit, recession

18. **Define the `HIGH_IMPACT_EVENTS` list** of event types that historically cause the largest market moves:
    - Federal Reserve rate decisions
    - Non-Farm Payrolls (NFP)
    - Consumer Price Index (CPI)
    - Producer Price Index (PPI)
    - GDP releases
    - OPEC meetings
    - Earnings releases
    - FDA approval or rejection announcements
    - Merger and acquisition announcements
    - Bankruptcy filings

### Step 2A-3: Initialize the NewsAggregator

19. **Write the `__init__` method** accepting an `api_keys` dictionary:
    - Store `api_keys` as an instance variable
    - Initialize `seen_urls` as an empty set for deduplication
    - Initialize `session` as `None` (will be created in async context)

20. **Implement async context manager methods** (`__aenter__` and `__aexit__`):
    - `__aenter__`: Create an `aiohttp.ClientSession()` and return `self`
    - `__aexit__`: Call `await self.session.close()` to properly release connections

### Step 2A-4: Implement the Main Fetch Orchestrator

21. **Write the `fetch_all` async method**:
    - Create a list of tasks calling each individual source fetcher
    - Use `asyncio.gather(*tasks, return_exceptions=True)` to run all fetchers concurrently
    - Iterate through results, logging any exceptions without crashing the entire pipeline
    - Collect all valid `NewsEvent` lists into a single flat list
    - Return the result of `_deduplicate(all_events)`

22. **Understand why `return_exceptions=True` is critical**: Without it, a single failing API source would crash the entire data collection pipeline. With it, failures are captured as exception objects that you can log and skip.

### Step 2A-5: Implement Individual Source Fetchers

23. **Write `_fetch_newsapi`**:
    - Build query parameters: search query covering finance/stocks/commodities/economy, English language, sorted by publication date, page size of 100
    - Make async GET request to the NewsAPI endpoint
    - Parse the JSON response and iterate through `articles`
    - Skip any article whose URL is already in `seen_urls`
    - Add new URLs to `seen_urls` immediately after checking
    - Create a `NewsEvent` for each new article, combining title and description for asset extraction
    - Parse the ISO 8601 timestamp, replacing `Z` with `+00:00` for Python compatibility

24. **Write `_fetch_sec_filings`**:
    - Target SEC EDGAR's full-text search for 8-K filings from today
    - Understand the key 8-K item types that move stock prices:
      - Item 1.01: Entry into material agreements
      - Item 1.03: Bankruptcy or receivership
      - Item 2.02: Results of operations (earnings)
      - Item 5.02: Departure or appointment of executives
      - Item 8.01: Other material events
    - Set `event_type` to `"regulatory"` and `urgency` to `"immediate"` for all 8-K filings
    - Extract the ticker symbol from the filing metadata for `assets_mentioned`

25. **Write `_fetch_economic_calendar`**:
    - Call Finnhub's economic calendar endpoint with your API token
    - Filter to only `"high"` and `"medium"` impact events — skip low-impact events to reduce noise
    - Format the content string to show: Actual value | Forecast | Previous value
    - Set `impact_score` to `1.0` for high-impact and `0.6` for medium-impact events
    - Set `urgency` to `"immediate"` for high-impact and `"hours"` for medium-impact events
    - Handle the case where `"actual"` is not yet available (event hasn't occurred) by showing `"TBD"`

26. **Write `_fetch_rss_feeds`**:
    - Define a dictionary of RSS source names to URLs (Reuters, WSJ, CNBC)
    - Use `feedparser.parse(url)` for each — this is synchronous, so consider running in an executor for production
    - Limit to the 20 most recent entries per feed to avoid processing stale news
    - Check `seen_urls` before processing each entry
    - Handle the `published_parsed` attribute carefully — use `datetime.now()` as fallback if missing

27. **Write `_fetch_fed_releases`**:
    - Parse the Federal Reserve's official RSS feed using feedparser
    - Limit to the 10 most recent releases
    - Hard-code `impact_score` to `0.95` — Fed communications are among the highest-impact events
    - Set `urgency` to `"immediate"` and `event_type` to `"macro"` for all Fed releases

### Step 2A-6: Implement Utility Methods

28. **Write `_extract_assets`**:
    - Convert the input text to lowercase for case-insensitive matching
    - Iterate through each category in `ASSET_KEYWORDS`
    - Check if any keyword from that category appears in the text
    - Return a list of matching category names
    - Note: This is a simple keyword approach. For production, use a Named Entity Recognition (NER) model like spaCy with a financial entity model for better accuracy

29. **Write `_deduplicate`**:
    - Create an empty set `seen_headlines` and empty list `unique`
    - For each event, create a key from the first 80 characters of the headline, lowercased and stripped
    - Only add events whose key hasn't been seen before
    - Sort the final list by `timestamp` in descending order (newest first)
    - Return the sorted unique list

---

## SECTION 3: PHASE 1B — MULTI-ASSET PRICE FEED

### Step 3A: Define the Asset Universe

30. **Open** `data/price_feed.py` and import required libraries:
    ```python
    import pandas as pd
    import numpy as np
    import yfinance as yf
    import websocket
    import json
    from typing import Dict, List, Callable, Optional
    from datetime import datetime, timedelta
    import threading
    ```

31. **Define the `ASSET_UNIVERSE` dictionary** organized by asset class and sub-category:
    - **Equities**:
      - Indices: S&P 500 (`^GSPC`), Dow Jones (`^DJI`), NASDAQ (`^IXIC`), Russell 2000 (`^RUT`), VIX (`^VIX`)
      - Sector ETFs: XLF (financials), XLE (energy), XLK (technology), XLV (healthcare), XLI (industrials), XLB (materials), XLU (utilities), XLP (consumer staples), XLY (consumer discretionary), XLRE (real estate)
      - Mega-cap stocks: AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, BRK-B, JPM, JNJ
    - **Commodities**:
      - Energy futures: CL=F (WTI crude), BZ=F (Brent crude), NG=F (natural gas), RB=F (gasoline), HO=F (heating oil)
      - Metals futures: GC=F (gold), SI=F (silver), HG=F (copper), PL=F (platinum), PA=F (palladium)
      - Agriculture futures: ZW=F (wheat), ZC=F (corn), ZS=F (soybeans), KC=F (coffee), CT=F (cotton)
    - **Crypto**: BTC-USD, ETH-USD, BNB-USD, SOL-USD, UNI-USD, AAVE-USD, LINK-USD
    - **Forex majors**: EURUSD=X, GBPUSD=X, USDJPY=X, USDCHF=X, AUDUSD=X, USDCAD=X
    - **Bonds**: Treasury yield indices (^TNX, ^TYX, ^FVX, ^IRX) and bond ETFs (TLT, IEF, SHY, HYG, LQD)

### Step 3B: Initialize the Price Feed

32. **Write the `__init__` method** initializing:
    - `price_cache`: Empty dictionary to store `{symbol: DataFrame}` pairs
    - `live_prices`: Empty dictionary to store `{symbol: float}` current prices
    - `callbacks`: Empty list of functions to call on price updates
    - `_ws_connections`: Empty list to track active WebSocket connections

### Step 3C: Implement Historical Data Fetching

33. **Write `get_historical`** accepting `symbols`, `period`, and `interval` parameters:
    - Iterate through each symbol
    - Create a `yf.Ticker(symbol)` object
    - Call `.history(period=period, interval=interval)` to get OHLCV data
    - Convert the index to UTC-aware datetime using `pd.to_datetime(df.index, utc=True)`
    - Calculate `returns` as percentage change of Close price
    - Calculate `log_returns` as natural log of price ratio (preferred for statistical analysis)
    - Calculate `volatility_20d` as 20-day rolling standard deviation of returns, annualized by multiplying by `sqrt(252)` (trading days per year)
    - Store each DataFrame in `price_cache`
    - Wrap in try/except to handle invalid symbols gracefully

34. **Understand the period and interval options**:
    - Period: `1d`, `5d`, `1mo`, `3mo`, `6mo`, `1y`, `2y`, `5y`, `10y`, `ytd`, `max`
    - Interval: `1m`, `2m`, `5m`, `15m`, `30m`, `60m`, `90m`, `1h`, `1d`, `5d`, `1wk`, `1mo`
    - Note: Intraday data (1m-90m) is only available for the past 60 days via yfinance

35. **Write `get_all_assets_history`**:
    - Flatten the entire `ASSET_UNIVERSE` dictionary into a single list of symbols
    - Call `get_historical` with that complete list
    - Return the resulting dictionary

### Step 3D: Implement Real-Time WebSocket Streaming

36. **Write `start_crypto_stream`**:
    - Default to BTC, ETH, BNB, and SOL if no symbols are provided
    - Build the Binance stream URL using the format: `wss://stream.binance.com:9443/stream?streams=btcusdt@ticker/ethusdt@ticker`
    - The `@ticker` suffix subscribes to the 24-hour rolling ticker, which includes current price, volume, and price change

37. **Define the `on_message` callback** inside `start_crypto_stream`:
    - Parse the JSON message
    - Extract the nested `data` object (Binance combined stream format wraps data)
    - Get the symbol from `"s"` field and current price from `"c"` (close/last price) field
    - Update `live_prices[symbol]` with the new price
    - Call all registered callbacks with `(symbol, price, full_ticker_data)`

38. **Define the `on_error` callback** to log WebSocket errors without crashing.

39. **Start the WebSocket in a daemon thread**:
    - Create a `websocket.WebSocketApp` with the URL and callbacks
    - Create a `threading.Thread` with `target=ws.run_forever` and `daemon=True`
    - Set `daemon=True` so the thread automatically stops when the main program exits
    - Start the thread and append the WebSocket to `_ws_connections`

40. **Write `register_price_callback`** to append a callable to the `callbacks` list, enabling other system components to react to price updates.

### Step 3E: Implement Technical Indicators

41. **Write `calculate_technical_indicators`** accepting a price DataFrame:

42. **Add trend indicators**:
    - SMA 20, 50, 200: Simple moving averages using `.rolling(n).mean()`
    - EMA 12 and 26: Exponential moving averages using `.ewm(span=n).mean()`
    - MACD: Subtract EMA 26 from EMA 12 (positive = bullish momentum)

43. **Add momentum indicators**:
    - RSI 14: Call the static `_rsi` helper method
    - ROC 10: 10-period rate of change using `.pct_change(10)`

44. **Add volatility indicators**:
    - Bollinger Band Upper: SMA 20 + (2 × 20-day standard deviation)
    - Bollinger Band Lower: SMA 20 - (2 × 20-day standard deviation)
    - ATR 14: Call the static `_atr` helper method

45. **Add volume indicators**:
    - Volume SMA 20: 20-day average volume
    - Volume Ratio: Current volume divided by 20-day average (values > 1.5 indicate unusual activity)

46. **Write the static `_rsi` method**:
    - Calculate price differences using `.diff()`
    - Separate gains (positive differences) and losses (negative differences, made positive)
    - Calculate rolling averages of gains and losses over the period
    - Compute RS (Relative Strength) as average gain divided by average loss
    - Return `100 - (100 / (1 + RS))`
    - Values above 70 indicate overbought; below 30 indicate oversold

47. **Write the static `_atr` method** (Average True Range):
    - Calculate three components of True Range:
      - High minus Low (intraday range)
      - Absolute value of High minus previous Close
      - Absolute value of Low minus previous Close
    - Take the maximum of these three values for each period using `pd.concat(...).max(axis=1)`
    - Return the rolling mean over the specified period
    - ATR measures volatility in price units, useful for position sizing

---

## SECTION 4: PHASE 2A — LLM NEWS ANALYSIS ENGINE

### Step 4A: Set Up the LLM Analyzer

48. **Open** `analysis/llm_analyzer.py` and import required libraries:
    ```python
    import json
    import asyncio
    from openai import AsyncOpenAI
    from anthropic import AsyncAnthropic
    from typing import List, Dict, Optional
    from data.news_aggregator import NewsEvent
    ```

49. **Define the `SYSTEM_PROMPT` class constant**:
    - Establish the LLM's persona as a senior quantitative analyst with 20 years of experience
    - Specify coverage across equities, commodities, forex, and derivatives
    - Embed key analytical principles directly in the prompt:
      - Focus on surprise versus expectation (markets price in consensus)
      - Consider second and third-order cross-asset effects
      - Distinguish between immediate, short-term, and medium-term impacts
      - Be conservative — only flag high-confidence signals
      - Always identify what could invalidate the thesis
      - Consider market microstructure factors (liquidity, time of day, options expiry)

50. **Write the `__init__` method**:
    - Accept `openai_key` (required) and `anthropic_key` (optional)
    - Initialize `AsyncOpenAI` client with the OpenAI key
    - Initialize `AsyncAnthropic` client only if an Anthropic key is provided, otherwise set to `None`

### Step 4B: Implement Core Event Analysis

51. **Write `analyze_event`**:
    - Call `_build_analysis_prompt(event)` to construct the user message
    - Call `openai.chat.completions.create` with:
      - Model: `"gpt-4o"` for complex analysis
      - Messages: system prompt + user prompt
      - `response_format={"type": "json_object"}` to guarantee parseable JSON output
      - `temperature=0.1` for consistent, analytical (not creative) responses
      - `max_tokens=2000` to allow detailed analysis
    - Parse the response content with `json.loads`
    - Add metadata fields: `event_id`, `timestamp`, `source`
    - Return the enriched analysis dictionary

52. **Understand why `temperature=0.1` is used**: Lower temperature makes the model more deterministic and focused. For financial analysis, you want consistent, logical reasoning rather than creative variation. Values near 0 are appropriate for structured analytical tasks.

53. **Write `analyze_batch`** for processing multiple events efficiently:
    - Accept a `max_concurrent` parameter (default 5) to control API rate limiting
    - Create an `asyncio.Semaphore(max_concurrent)` to limit simultaneous API calls
    - Define an inner `analyze_with_limit` coroutine that acquires the semaphore before calling `analyze_event`
    - Wrap each call in try/except, returning `None` on failure
    - Use `asyncio.gather` to run all analyses concurrently within the semaphore limit
    - Filter out `None` results before returning

### Step 4C: Implement Specialized Fed Statement Analysis

54. **Write `analyze_fed_statement`** for Federal Reserve communications:
    - This is the highest-priority analysis function — Fed statements can move all asset classes simultaneously
    - Design the prompt to extract:
      - `policy_stance`: hawkish, dovish, or neutral
      - `stance_change`: whether this represents a shift from the previous statement
      - `rate_path_signal`: hikes, cuts, hold, or uncertain
      - `key_phrases`: specific language that markets will react to
      - `balance_sheet_signal`: quantitative tightening or easing signals
      - `inflation_assessment`: how the Fed characterizes inflation
      - `growth_assessment`: how the Fed characterizes economic growth
      - `asset_impacts`: for each of equities, bonds, gold, USD, and commodities — direction, magnitude, and reasoning
      - `surprise_factor`: how much this deviated from market expectations (0.0 to 1.0)
      - `confidence`: model's confidence in the analysis (0.0 to 1.0)
      - `key_risks`: list of factors that could invalidate the thesis
    - Use `temperature=0.05` — even lower than standard analysis for maximum consistency

55. **Understand the Fed analysis logic**:
    - Hawkish signals (rate hikes, balance sheet reduction): USD up, bonds down, gold down, equities mixed
    - Dovish signals (rate cuts, balance sheet expansion): USD down, bonds up, gold up, equities up
    - The surprise factor is critical — a hawkish statement that was already priced in may cause little movement

### Step 4D: Implement Earnings Analysis

56. **Write `analyze_earnings`** accepting an `earnings_data` dictionary:
    - The prompt should include: company name, ticker, EPS actual vs estimate, revenue actual vs estimate, guidance, and management comments
    - Design the output JSON to include:
      - `eps_surprise_pct`: Percentage by which EPS beat or missed estimates
      - `revenue_surprise_pct`: Percentage by which revenue beat or missed estimates
      - `guidance_change`: raised, lowered, maintained, withdrawn, or none
      - `overall_quality`: beat, miss, or inline
      - `expected_price_reaction`: direction, magnitude percentage, and confidence
      - `sector_read_through`: list of peer companies that may be affected and how
      - `key_risks`: factors that could change the expected reaction
      - `time_horizon`: immediate (pre-market/after-hours) or short-term (next few days)

57. **Understand earnings analysis nuances**:
    - A company can beat EPS but miss revenue and still sell off
    - Guidance is often more important than the actual results
    - "Whisper numbers" (unofficial expectations) often differ from official consensus
    - Sector read-through means: if Apple beats on iPhone sales, it's positive for TSMC, Qualcomm, etc.

### Step 4E: Implement Cross-Asset Effect Analysis

58. **Write `find_cross_asset_effects`** accepting a primary analysis and list of assets:
    - This function identifies second and third-order effects across asset classes
    - Classic examples to encode in the prompt:
      - Oil price spike → Airlines down (fuel costs), Defense contractors up (geopolitical risk), USD up (petrodollar), Gold up (inflation hedge), Emerging market currencies down (oil importers)
      - Strong NFP report → USD up, Bonds down (rate hike expectations), Gold down, Equities mixed
      - China GDP miss → Copper down (industrial demand), Australian dollar down (commodity exports), Luxury goods stocks down
    - The output should rank affected assets by expected impact magnitude
    - Include confidence scores for each cross-asset relationship

---

## SECTION 5: INTEGRATION AND TESTING

### Step 5A: Build Integration Tests

59. **Create** `tests/test_news_aggregator.py`:
    - Test that `NewsEvent` dataclass initializes with correct defaults
    - Test `_extract_assets` with known text containing financial keywords
    - Test `_deduplicate` with a list containing duplicate headlines
    - Mock the HTTP session to test individual fetchers without making real API calls

60. **Create** `tests/test_price_feed.py`:
    - Test `get_historical` with a single well-known symbol (e.g., `"AAPL"`)
    - Verify the returned DataFrame contains expected columns: Open, High, Low, Close, Volume, returns, log_returns, volatility_20d
    - Test `calculate_technical_indicators` and verify all indicator columns are present
    - Test `_rsi` with a known price series and verify output is between 0 and 100

61. **Create** `tests/test_llm_analyzer.py`:
    - Mock the OpenAI API client to avoid real API calls during testing
    - Test that `analyze_event` correctly parses JSON responses
    - Test that `analyze_batch` respects the concurrency limit
    - Test error handling when the API returns an invalid response

### Step 5B: Build a Simple Integration Runner

62. **Create** `main.py` as a simple integration test:
    ```python
    import asyncio
    from data.news_aggregator import NewsAggregator
    from data.price_feed import MultiAssetPriceFeed
    from analysis.llm_analyzer import LLMNewsAnalyzer
    
    async def main():
        # Test news aggregation
        api_keys = {"newsapi": "YOUR_KEY", "finnhub": "YOUR_KEY"}
        async with NewsAggregator(api_keys) as aggregator:
            events = await aggregator.fetch_all()
            print(f"Fetched {len(events)} news events")
            for event in events[:3]:
                print(f"  [{event.source}] {event.headline[:80]}")
        
        # Test price feed
        feed = MultiAssetPriceFeed()
        prices = feed.get_historical(["AAPL", "GC=F", "BTC-USD"], period="1mo")
        for symbol, df in prices.items():
            print(f"  {symbol}: {len(df)} rows, latest close: {df['Close'].iloc[-1]:.2f}")
    
    asyncio.run(main())
    ```

63. **Run the integration test** and verify:
    - News events are being fetched from at least one source
    - Price data is returned with correct columns
    - No unhandled exceptions occur
    - API rate limits are not being exceeded

---

## SECTION 6: OVERSIGHT AND COMPLIANCE LAYER

64. **Create** `oversight/audit_logger.py`:
    - Log every news event received with timestamp and source
    - Log every LLM analysis call with input, output, model used, and cost estimate
    - Log every trading signal generated with full reasoning chain
    - Use structured logging (JSON format) for easy querying
    - Store logs to both file and a database for regulatory compliance

65. **Implement human review checkpoints**:
    - Define a minimum confidence threshold below which signals require human approval
    - Create a review queue where flagged signals wait for human confirmation
    - Set a maximum position size that can be executed without human review
    - Implement a kill switch that immediately halts all automated activity

66. **Set up regulatory reporting**:
    - Research your jurisdiction's requirements for algorithmic trading records
    - Implement trade blotter logging capturing: timestamp, symbol, direction, quantity, price, rationale
    - Store all records for the minimum required retention period (typically 5-7 years)
    - Ensure records are tamper-evident (consider cryptographic hashing of log entries)

67. **Implement circuit breakers**:
    - Daily loss limit: Halt all trading if losses exceed a defined threshold (e.g., 2% of portfolio)
    - Drawdown limit: Halt if portfolio drops more than a defined percentage from peak
    - Volatility circuit breaker: Reduce position sizes or halt during extreme market volatility (VIX > 40)
    - News blackout: Pause trading during scheduled high-impact events until analysis is complete

---

## SECTION 7: NEXT STEPS AND ONGOING DEVELOPMENT

68. **Plan Phase 3 (Decision Layer)** components:
    - Strategy engine: Convert LLM analysis into specific trade parameters (symbol, direction, size, entry, stop, target)
    - Risk management module: Position sizing using Kelly Criterion or fixed fractional methods
    - Portfolio optimizer: Ensure new trades don't create excessive correlation or concentration
    - Backtesting framework: Test strategies on historical data before live deployment

69. **Plan Phase 4 (Execution Layer)** components:
    - Broker API integration: Start with paper trading (simulated) before live trading
    - Order management system: Handle order types (market, limit, stop), partial fills, and rejections
    - Position monitoring: Track open positions, unrealized P&L, and risk metrics in real time
    - Automated risk stops: Implement stop-loss orders that execute without human intervention

70. **Establish a paper trading period**:
    - Run the complete system in simulation mode for a minimum of 3-6 months
    - Track all signals that would have been executed and their hypothetical outcomes
    - Calculate performance metrics: Sharpe ratio, maximum drawdown, win rate, profit factor
    - Only consider live trading after demonstrating consistent paper trading performance

71. **Plan ongoing model maintenance**:
    - Monitor LLM analysis quality by reviewing a random sample of outputs weekly
    - Retrain or fine-tune models as market regimes change
    - Update the `ASSET_KEYWORDS` dictionary as new assets and terminology emerge
    - Review and update the `HIGH_IMPACT_EVENTS` list quarterly
    - Track API costs and optimize model selection (use cheaper models for routine tasks)

72. **Consult professionals before going live**:
    - Hire a securities attorney to review your system for regulatory compliance
    - Consult a licensed financial advisor about appropriate risk parameters
    - Contact your broker's algorithmic trading desk to understand their requirements
    - Consider obtaining relevant licenses if managing money for others (RIA, CPO, etc.)

**Recommended Next Step:** Save the code above and run it!
