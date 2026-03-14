# Question
I would like to have all the ai models search the internet for financial data of how assets go up and down relating to world events and then have ai agents take advantage of that information to make trades

## Final Result from AI Conductor

### Synthesized Plan
# AI-Powered Event-Driven Trading System: Complete Implementation Plan

---

## System Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     ORCHESTRATOR AGENT                       │
│                  (Coordinates all agents)                    │
└──────────┬──────────────┬──────────────┬────────────────────┘
           │              │              │              │
    ┌──────▼──────┐ ┌─────▼──────┐ ┌────▼───────┐ ┌───▼──────────┐
    │    DATA     │ │  ANALYSIS  │ │    RISK    │ │   TRADING    │
    │    AGENT    │ │   AGENT    │ │   AGENT    │ │    AGENT     │
    └─────────────┘ └────────────┘ └────────────┘ └──────────────┘
         │                │               │               │
    News/Markets     AI Analysis     Validate Risk    Execute Orders
    World Events    Multi-Model     Position Size    Stops/Targets
    Macro Data      Sentiment       Loss Limits      Monitor Fills
```

> **⚠️ Critical Warning:** This system can lose real money rapidly. Start with paper trading only. No AI model can consistently predict markets. Treat this as a learning tool first.

---

## Phase 1: Setup & Infrastructure

### 1.1 Environment Setup

```bash
# Core dependencies
pip install openai anthropic

# Data collection
pip install yfinance pandas numpy feedparser
pip install requests beautifulsoup4 tavily-python

# Trading execution
pip install alpaca-trade-api python-dotenv

# Scheduling and utilities
pip install schedule scikit-learn
```

### 1.2 Configuration

```python
# config.py
import os
from dotenv import load_dotenv

load_dotenv()

CONFIG = {
    # AI Models
    "openai_api_key": os.getenv("OPENAI_API_KEY"),
    "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),

    # Data Sources
    "tavily_api_key": os.getenv("TAVILY_API_KEY"),
    "news_api_key": os.getenv("NEWS_API_KEY"),

    # Broker — PAPER TRADING ONLY until validated
    "alpaca_api_key": os.getenv("ALPACA_API_KEY"),
    "alpaca_secret": os.getenv("ALPACA_SECRET"),
    "alpaca_base_url": "https://paper-api.alpaca.markets",  # ← KEEP THIS

    # Hard Risk Limits (never remove these)
    "max_position_pct": 0.05,      # 5% of portfolio per trade
    "max_daily_loss_pct": 0.02,    # 2% max daily loss → system stops
    "max_portfolio_risk": 0.15,    # 15% total capital at risk
    "min_risk_reward": 2.0,        # Minimum 2:1 reward-to-risk ratio
}
```

---

## Phase 2: Data Collection Agent

Pulls three types of data: **breaking news**, **market prices/technicals**, and **macro indicators**.

```python
# agents/data_agent.py
import feedparser
import yfinance as yf
import pandas as pd
from tavily import TavilyClient

class DataCollectionAgent:
    def __init__(self, config):
        self.tavily = TavilyClient(api_key=config["tavily_api_key"])

        # Prioritize high-quality financial sources
        self.news_feeds = [
            "https://feeds.reuters.com/reuters/businessNews",
            "https://feeds.bloomberg.com/markets/news.rss",
            "https://www.ft.com/rss/home",
            "https://feeds.wsj.com/wsj/xml/rss/3_7085.xml",
        ]

    # ── News Collection ────────────────────────────────────────────────
    def get_breaking_news(self) -> list[dict]:
        articles = []
        for url in self.news_feeds:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:10]:
                    articles.append({
                        "title": entry.get("title", ""),
                        "summary": entry.get("summary", ""),
                        "published": entry.get("published", ""),
                        "source": url,
                        "link": entry.get("link", ""),
                    })
            except Exception as e:
                print(f"Feed error {url}: {e}")
        return articles

    def search_event_context(self, query: str) -> dict:
        """Deep search for specific world events affecting markets"""
        return self.tavily.search(
            query=query,
            search_depth="advanced",
            include_domains=[
                "reuters.com", "bloomberg.com", "wsj.com",
                "ft.com", "cnbc.com", "marketwatch.com"
            ],
            max_results=10
        )

    # ── Market Data ────────────────────────────────────────────────────
    def get_asset_data(self, symbols: list[str]) -> dict:
        """Price, volume, RSI, moving averages for each symbol"""
        market_data = {}
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                hist = ticker.history(period="30d")

                hist["SMA_20"] = hist["Close"].rolling(20).mean()
                hist["RSI"] = self._calculate_rsi(hist["Close"])

                market_data[symbol] = {
                    "current_price": info.get("currentPrice", 0),
                    "market_cap": info.get("marketCap", 0),
                    "pe_ratio": info.get("trailingPE"),
                    "52w_high": info.get("fiftyTwoWeekHigh", 0),
                    "52w_low": info.get("fiftyTwoWeekLow", 0),
                    "volume": info.get("volume", 0),
                    "avg_volume": info.get("averageVolume", 0),
                    "beta": info.get("beta", 1),
                    "rsi": float(hist["RSI"].iloc[-1]) if not hist.empty else None,
                    "sma_20": float(hist["SMA_20"].iloc[-1]) if not hist.empty else None,
                    "sector": info.get("sector", "Unknown"),
                }
            except Exception as e:
                market_data[symbol] = {"error": str(e)}
        return market_data

    def get_macro_indicators(self) -> dict:
        """
        Macro context is critical — world events often move these first
        before individual stocks react.
        """
        indices = {
            "SP500": "^GSPC",
            "NASDAQ": "^IXIC",
            "VIX": "^VIX",           # Fear gauge — high VIX = reduce size
            "DXY": "DX-Y.NYB",       # Dollar strength
            "GOLD": "GC=F",          # Safe haven demand
            "OIL": "CL=F",           # Geopolitical sensitivity
            "10Y_TREASURY": "^TNX",  # Rate expectations
        }
        indicators = {}
        for name, symbol in indices.items():
            try:
                hist = yf.Ticker(symbol).history(period="5d")
                if not hist.empty:
                    cur = hist["Close"].iloc[-1]
                    prev = hist["Close"].iloc[-2] if len(hist) > 1 else cur
                    indicators[name] = {
                        "value": cur,
                        "change_pct": ((cur - prev) / prev) * 100
                    }
            except:
                pass
        return indicators

    def _calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
```

---

## Phase 3: Analysis Agent (Multi-Model AI)

Uses **GPT-4o for speed** (sentiment, signal generation) and **Claude for depth** (world event causality analysis). Using two models provides a cross-check and reduces single-model blind spots.

```python
# agents/analysis_agent.py
from openai import OpenAI
import anthropic
import json

class AnalysisAgent:
    def __init__(self, config):
        self.openai = OpenAI(api_key=config["openai_api_key"])
        self.claude = anthropic.Anthropic(api_key=config["anthropic_api_key"])

    # ── Step 1: Sentiment Scan (GPT-4o — fast) ────────────────────────
    def analyze_news_sentiment(self, articles: list[dict]) -> dict:
        """
        Identifies which assets are affected and in which direction.
        GPT-4o used here for speed — this runs on every cycle.
        """
        text = "\n\n".join([
            f"HEADLINE: {a['title']}\nSUMMARY: {a['summary']}"
            for a in articles[:20]
        ])

        response = self.openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a financial analyst. Return JSON only."},
                {"role": "user", "content": f"""Analyze these articles:

{text}

Return JSON:
{{
    "overall_sentiment": "bullish|bearish|neutral",
    "sentiment_score": -100 to 100,
    "affected_assets": [
        {{
            "symbol": "XLE",
            "direction": "up|down|neutral",
            "reason": "...",
            "magnitude": "high|medium|low",
            "timeframe": "intraday|days|weeks"
        }}
    ],
    "sector_outlook": {{"Energy": "bearish", "Technology": "bullish"}},
    "risk_events": ["Fed meeting", "CPI release"],
    "confidence": 70,
    "key_themes": ["inflation", "rate hike expectations"]
}}"""}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)

    # ── Step 2: Causal Event Analysis (Claude — deep) ─────────────────
    def analyze_world_event_impact(self, event: str, market_data: dict) -> dict:
        """
        Claude analyzes historical precedents and causal chains.
        This is where second-order effects are identified —
        e.g., an oil shock hits energy stocks (obvious) but also
        airlines, shipping, and consumer discretionary (less obvious).
        """
        message = self.claude.messages.create(
            model="claude-opus-4-5",
            max_tokens=2000,
            messages=[{"role": "user", "content": f"""Analyze this market event:

EVENT: {event}

MARKET CONTEXT:
{json.dumps(market_data, indent=2, default=str)[:3000]}

Provide:
1. Historical precedents for similar events and their outcomes
2. Direct asset impacts (specific tickers) with reasoning
3. Second-order effects (what else gets affected indirectly)
4. Safe haven asset flows expected
5. Expected timeline of market reaction
6. Where correlation vs. causation risk exists in your analysis

Return JSON:
{{
    "event_type": "geopolitical|economic|natural|political|company",
    "severity": "low|medium|high|extreme",
    "historical_precedents": ["2022 Russia-Ukraine → Oil spiked 30%", "..."],
    "direct_impacts": [{{"symbol": "XOM", "direction": "up", "reason": "..."}}],
    "second_order_effects": [{{"symbol": "DAL", "direction": "down", "reason": "fuel costs"}}],
    "safe_haven_assets": ["GLD", "TLT", "USD"],
    "reaction_timeline": "intraday|1-3 days|1-2 weeks",
    "causation_confidence": "high|medium|low",
    "key_risks_to_thesis": ["event reversal", "already priced in"]
}}"""}]
        )
        try:
            return json.loads(message.content[0].text)
        except:
            return {"raw_analysis": message.content[0].text}

    # ── Step 3: Trade Signal Generation (GPT-4o) ──────────────────────
    def generate_trade_signals(
        self, sentiment: dict, event_analysis: dict,
        market_data: dict, portfolio: dict
    ) -> dict:
        context = {
            "sentiment": sentiment,
            "event_analysis": event_analysis,
            "current_positions": portfolio.get("positions", {}),
            "buying_power": portfolio.get("buying_power", 0),
            "macro": market_data.get("macro", {}),
        }

        response = self.openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a systematic trader. Generate precise signals. "
                        "Always include stop loss. Return JSON only."
                    )
                },
                {"role": "user", "content": f"""Generate trade signals:

{json.dumps(context, indent=2, default=str)[:4000]}

Rules:
- Max 5% of portfolio per position
- Every trade MUST have a stop loss
- Minimum 2:1 reward-to-risk ratio
- Reduce size if VIX > 30

Return JSON:
{{
    "signals": [
        {{
            "symbol": "NVDA",
            "action": "buy|sell|short|cover",
            "quantity": 10,
            "order_type": "market|limit",
            "limit_price": null,
            "stop_loss": 430.00,
            "take_profit": 510.00,
            "rationale": "...",
            "priority": "high|medium|low",
            "risk_reward_ratio": 2.5,
            "max_loss_dollars": 400
        }}
    ],
    "risk_summary": "..."
}}"""}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)

    # ── Step 4: Independent Risk Validation (GPT-4o as second opinion) ─
    def validate_trade(self, trade: dict, portfolio: dict) -> dict:
        """
        A separate AI call acts as a risk officer reviewing the trade.
        Different system prompt = different perspective on same trade.
        """
        response = self.openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a conservative risk officer. "
                        "Your job is to protect capital, not maximize returns. "
                        "Reject or modify any trade that looks risky. "
                        "Return JSON only."
                    )
                },
                {"role": "user", "content": f"""Review this trade:

TRADE: {json.dumps(trade, indent=2)}
PORTFOLIO: {json.dumps(portfolio, indent=2, default=str)}

Check: position size, stop loss adequacy, risk/reward ratio, 
correlation with existing positions, market conditions.

Return:
{{
    "approved": true/false,
    "modified_trade": {{...}} or null,
    "rejection_reason": null or "...",
    "risk_score": 0-100,
    "warnings": ["..."],
    "recommended_quantity": 10
}}"""}
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
```

---

## Phase 4: Trading Execution Agent

```python
# agents/trading_agent.py
import alpaca_trade_api as tradeapi
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class TradingAgent:
    def __init__(self, config):
        self.api = tradeapi.REST(
            config["alpaca_api_key"],
            config["alpaca_secret"],
            config["alpaca_base_url"],
            api_version="v2"
        )
        self.config = config
        self.trade_log = []

    def get_portfolio_state(self) -> dict:
        account = self.api.get_account()
        positions = self.api.list_positions()
        return {
            "equity": float(account.equity),
            "buying_power": float(account.buying_power),
            "portfolio_value": float(account.portfolio_value),
            "daily_pnl": float(account.equity) - float(account.last_equity),
            "positions": {
                p.symbol: {
                    "qty": float(p.qty),
                    "current_price": float(p.current_price),
                    "market_value": float(p.market_value),
                    "unrealized_pnl": float(p.unrealized_pl),
                    "avg_entry_price": float(p.avg_entry_price),
                }
                for p in positions
            },
        }

    def check_hard_risk_limits(self, portfolio: dict) -> tuple[bool, str]:
        """
        These limits are non-negotiable. If hit, trading stops entirely.
        No AI override. No exceptions.
        """
        pv = portfolio["portfolio_value"]

        # Daily loss limit
        daily_loss_pct = abs(min(portfolio["daily_pnl"], 0)) / pv
        if daily_loss_pct >= self.config["max_daily_loss_pct"]:
            return False, f"Daily loss limit hit ({daily_loss_pct:.1%}). System halted."

        # Total portfolio risk
        total_at_risk = sum(
            abs(p["unrealized_pnl"])
            for p in portfolio["positions"].values()
            if p["unrealized_pnl"] < 0
        )
        if total_at_risk / pv >= self.config["max_portfolio_risk"]:
            return False, f"Portfolio risk limit hit ({total_at_risk/pv:.1%}). No new trades."

        return True, "OK"

    def execute_trade(self, signal: dict, portfolio: dict) -> dict:
        ok, reason = self.check_hard_risk_limits(portfolio)
        if not ok:
            logger.warning(reason)
            return {"success": False, "reason": reason}

        symbol = signal["symbol"]
        side = "buy" if signal["action"] in ["buy", "cover"] else "sell"

        try:
            order_params = {
                "symbol": symbol,
                "qty": signal["quantity"],
                "side": side,
                "type": signal.get("order_type", "market"),
                "time_in_force": "day",
            }
            if signal.get("limit_price") and order_params["type"] == "limit":
                order_params["limit_price"] = signal["limit_price"]

            order = self.api.submit_order(**order_params)
            logger.info(f"ORDER: {side} {signal['quantity']} {symbol} | ID: {order.id}")

            # Immediately set stop loss
            if signal.get("stop_loss") and side == "buy":
                self._set_stop_loss(symbol, signal["quantity"], signal["stop_loss"])

            record = {
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "action": signal["action"],
                "quantity": signal["quantity"],
                "order_id": order.id,
                "stop_loss": signal.get("stop_loss"),
                "take_profit": signal.get("take_profit"),
                "rationale": signal.get("rationale", ""),
            }
            self.trade_log.append(record)
            return {"success": True, "order": record}

        except Exception as e:
            logger.error(f"Trade failed for {symbol}: {e}")
            return {"success": False, "reason": str(e)}

    def _set_stop_loss(self, symbol: str, qty: float, stop_price: float):
        try:
            self.api.submit_order(
                symbol=symbol, qty=qty, side="sell",
                type="stop", stop_price=stop_price,
                time_in_force="gtc"
            )
            logger.info(f"Stop loss set: {symbol} @ ${stop_price}")
        except Exception as e:
            logger.error(f"Stop loss failed for {symbol}: {e}")

    def emergency_close_all(self):
        """Use this if the system behaves unexpectedly"""
        logger.warning("⚠️ EMERGENCY CLOSE ALL POSITIONS")
        self.api.close_all_positions()
        self.api.cancel_all_orders()
```

---

## Phase 5: Orchestrator — Main Loop

```python
# orchestrator.py
import asyncio
import schedule
import time
import json
from datetime import datetime
from agents.data_agent import DataCollectionAgent
from agents.analysis_agent import AnalysisAgent
from agents.trading_agent import TradingAgent
from config import CONFIG

class TradingOrchestrator:
    def __init__(self):
        self.data = DataCollectionAgent(CONFIG)
        self.analysis = AnalysisAgent(CONFIG)
        self.trader = TradingAgent(CONFIG)
        self.performance_log = []

        # Start focused — expand after validating the system works
        self.watchlist = [
            "AAPL", "MSFT", "NVDA", "GOOGL",   # Tech
            "JPM", "GS",                          # Finance
            "XOM", "CVX",                         # Energy
            "SPY", "QQQ", "GLD", "TLT", "XLE",  # ETFs
        ]

    async def run_cycle(self):
        cycle_start = datetime.now()
        print(f"\n{'='*60}")
        print(f"CYCLE | {cycle_start.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")

        # ── 1. Collect Data ────────────────────────────────────────────
        print("\n📡 Collecting data...")
        news = self.data.get_breaking_news()
        market_data = self.data.get_asset_data(self.watchlist)
        macro = self.data.get_macro_indicators()
        portfolio = self.trader.get_portfolio_state()

        print(f"   ✓ {len(news)} articles | "
              f"Portfolio: ${portfolio.get('portfolio_value', 0):,.2f} | "
              f"Daily P&L: ${portfolio.get('daily_pnl', 0):+,.2f}")

        # ── 2. Hard Risk Check — stop here if limits are hit ───────────
        ok, reason = self.trader.check_hard_risk_limits(portfolio)
        if not ok:
            print(f"\n⛔ {reason}")
            return

        # Log VIX — high volatility should reduce position sizes
        vix = macro.get("VIX", {}).get("value", 0)
        if vix > 30:
            print(f"\n⚠️  VIX = {vix:.1f} (elevated). Signals will reduce position sizes.")

        # ── 3. GPT-4o: Sentiment Analysis ─────────────────────────────
        print("\n🤖 Analyzing sentiment (GPT-4o)...")
        sentiment = self.analysis.analyze_news_sentiment(news)
        print(f"   Sentiment: {sentiment.get('overall_sentiment', '?').upper()} | "
              f"Score: {sentiment.get('sentiment_score', 0)} | "
              f"Confidence: {sentiment.get('confidence', 0)}%")

        # ── 4. Claude: Deep Event Analysis ────────────────────────────
        key_events = self._extract_significant_events(news)
        event_analysis = {}
        if key_events:
            print(f"\n🌍 Analyzing world event (Claude): {key_events[0][:80]}...")
            event_analysis = self.analysis.analyze_world_event_impact(
                key_events[0],
                {"macro": macro, "assets": market_data}
            )
            print(f"   Severity: {event_analysis.get('severity', '?')} | "
                  f"Type: {event_analysis.get('event_type', '?')}")

        # ── 5. Generate Trade Signals ──────────────────────────────────
        print("\n💡 Generating signals...")
        signals_data = self.analysis.generate_trade_signals(
            sentiment, event_analysis,
            {"macro": macro, "assets": market_data},
            portfolio
        )
        signals = signals_data.get("signals", [])
        print(f"   {len(signals)} signal(s) generated")

        # ── 6. Validate & Execute ──────────────────────────────────────
        executed, rejected = 0, 0

        for signal in signals:
            if signal.get("priority") == "low":
                continue

            print(f"\n   → {signal['action'].upper()} "
                  f"{signal['quantity']} {signal['symbol']} | "
                  f"R/R: {signal.get('risk_reward_ratio', 0):.1f}:1")

            # Independent AI risk check
            validation = self.analysis.validate_trade(signal, portfolio)
            risk_score = validation.get("risk_score", 100)
            approved = validation.get("approved", False)

            print(f"     Risk Score: {risk_score}/100 | Approved: {approved}")
            for w in validation.get("warnings", []):
                print(f"     ⚠️  {w}")

            if approved and risk_score < 70:
                final_signal = validation.get("modified_trade") or signal
                result = self.trader.execute_trade(final_signal, portfolio)
                if result["success"]:
                    print(f"     ✅ EXECUTED")
                    executed += 1
                else:
                    print(f"     ❌ FAILED: {result.get('reason')}")
                    rejected += 1
            else:
                print(f"     🚫 REJECTED: {validation.get('rejection_reason', 'Risk too high')}")
                rejected += 1

        # ── 7. Log ─────────────────────────────────────────────────────
        self.performance_log.append({
            "timestamp": cycle_start.isoformat(),
            "portfolio_value": portfolio.get("portfolio_value", 0),
            "daily_pnl": portfolio.get("daily_pnl", 0),
            "signals": len(signals),
            "executed": executed,
            "rejected": rejected,
            "vix": vix,
        })
        print(f"\n📊 Done | Executed: {executed} | Rejected: {rejected}")

    def _extract_significant_events(self, articles: list) -> list:
        """Filter for genuinely market-moving events"""
        high_impact_keywords = [
            "fed", "interest rate", "inflation", "gdp", "jobs report",
            "earnings beat", "earnings miss", "merger", "acquisition",
            "war", "sanctions", "tariff", "recession", "bankruptcy",
            "crisis", "central bank", "powell", "rate hike", "rate cut",
        ]
        significant = []
        for article in articles:
            text = (article["title"] + " " + article["summary"]).lower()
            if any(kw in text for kw in high_impact_keywords):
                significant.append(article["title"])
        return significant[:3]

    def run(self, interval_minutes: int = 15):
        print("🚀 AI Trading System Starting")
        print(f"   Assets monitored: {len(self.watchlist)}")
        print(f"   Scan interval: {interval_minutes} min")
        print(f"   Max position: {CONFIG['max_position_pct']*100}%")
        print(f"   Daily loss limit: {CONFIG['max_daily_loss_pct']*100}%")
        print("\n   ⚠️  PAPER TRADING MODE — No real money at risk\n")

        asyncio.run(self.run_cycle())
        schedule.every(interval_minutes).minutes.do(
            lambda: asyncio.run(self.run_cycle())
        )
        while True:
            schedule.run_pending()
            time.sleep(60)

if __name__ == "__main__":
    TradingOrchestrator().run(interval_minutes=15)
```

---

## Phase 6: What to Expect (Honest Assessment)

| Factor | Reality |
|--------|---------|
| **Market speed** | Major news is priced in within seconds. At 15-min intervals, you're trading *aftermath*, not the event |
| **Overfitting risk** | AI signals that look great on past data often fail live. Backtest extensively before trusting |
| **Causation vs. correlation** | AI can identify patterns but struggles to distinguish true causation from coincidence |
| **Black swan events** | By definition unpredictable. No model handles 2020-style crashes well |
| **Realistic edge** | Best opportunity: slower-moving macro themes (sector rotation, rate cycle plays) rather than instant news reaction |

---

## Recommended Rollout Sequence

```
Month 1-2:  Paper trade only. Log every signal. Track what the AI
            gets right vs. wrong. Don't execute anything live.

Month 3-4:  Backtest using yfinance historical data. Does the
            event→asset relationship actually hold historically?

Month 5-6:  If paper results are positive (Sharpe > 1.0, max
            drawdown < 10%), consider live trading with $500-1000
            maximum. Treat it as tuition, not investment.

Ongoing:    Retrain/reprompt models regularly. Markets evolve
            and model drift will degrade performance over time.
```

---

## Key Files Reference

```
trading_system/
├── config.py               # API keys and risk parameters
├── orchestrator.py         # Main loop — run this
├── agents/
│   ├── data_agent.py       # News, prices, macro data
│   ├── analysis_agent.py   # GPT-4o + Claude analysis
│   └── trading_agent.py    # Alpaca order execution
└── logs/
    └── performance.json    # Auto-generated trade log
```

The system is deliberately modular — you can swap Alpaca for another broker, replace Tavily with a different search API, or add new data sources (Reddit sentiment, SEC filings, satellite data) by extending the `DataCollectionAgent` without touching the rest of the system.

### Code Agent Output
# AI-Powered Event-Driven Trading System

I'll implement this system with clean, well-documented code that closely follows the architecture plan while adding proper error handling, logging, and safety guardrails throughout.

## Project Structure

```
trading_system/
├── .env                    # API keys (never commit this)
├── config.py               # Configuration and risk parameters
├── orchestrator.py         # Main entry point
├── agents/
│   ├── __init__.py
│   ├── data_agent.py       # Data collection
│   ├── analysis_agent.py   # AI analysis (GPT-4o + Claude)
│   └── trading_agent.py    # Order execution
└── logs/
    └── performance.json    # Auto-generated trade log
```

## config.py

```python
"""
config.py
---------
Central configuration for the trading system.

All risk limits live here. They are referenced by every agent,
so changing a value here propagates system-wide — no need to
hunt through multiple files.

⚠️  PAPER TRADING ONLY until you have validated the system for
    several months and understand its failure modes.
"""

import os
from dotenv import load_dotenv

load_dotenv()


def _require(key: str) -> str:
    """
    Retrieve a required environment variable.

    Raises a clear error at startup rather than failing silently
    mid-run when a missing key causes an API call to blow up.
    """
    value = os.getenv(key)
    if not value:
        raise EnvironmentError(
            f"Missing required environment variable: {key}\n"
            f"Add it to your .env file before starting the system."
        )
    return value


CONFIG = {
    # ── AI Models ───────────────────────────────────────────────────────
    "openai_api_key":    _require("OPENAI_API_KEY"),
    "anthropic_api_key": _require("ANTHROPIC_API_KEY"),

    # ── Data Sources ────────────────────────────────────────────────────
    "tavily_api_key": _require("TAVILY_API_KEY"),

    # ── Broker (Alpaca) ─────────────────────────────────────────────────
    # The URL below points to Alpaca's paper trading environment.
    # Do NOT change this to the live URL until you have completed
    # months of paper trading and backtesting validation.
    "alpaca_api_key":  _require("ALPACA_API_KEY"),
    "alpaca_secret":   _require("ALPACA_SECRET"),
    "alpaca_base_url": "https://paper-api.alpaca.markets",  # ← KEEP THIS

    # ── Hard Risk Limits ────────────────────────────────────────────────
    # These are checked before every trade. Breaching any of them
    # halts new trading for the session. Do not relax these limits
    # until you fully understand what you are doing.
    "max_position_pct":   0.05,   # 5 % of portfolio per trade
    "max_daily_loss_pct": 0.02,   # 2 % daily loss → system stops
    "max_portfolio_risk": 0.15,   # 15 % total capital at risk
    "min_risk_reward":    2.0,    # Minimum 2:1 reward-to-risk ratio

    # ── Behavioural Thresholds ──────────────────────────────────────────
    "vix_caution_threshold": 30,  # Reduce sizes when VIX exceeds this
    "max_risk_score":        70,  # AI risk-officer score above which we skip
}
```

---

## agents/\_\_init\_\_.py

```python
"""agents/__init__.py — marks the directory as a Python package."""
```

---

## agents/data\_agent.py

```python
"""
agents/data_agent.py
--------------------
DataCollectionAgent — the system's sensory layer.

Responsibilities
----------------
1. Pull breaking news from RSS feeds (Reuters, Bloomberg, FT, WSJ).
2. Deep-search a specific event via Tavily for richer context.
3. Fetch price history and compute basic technicals (RSI, SMA-20).
4. Retrieve macro-level indicators (VIX, DXY, Oil, Gold, Treasuries).

Design notes
------------
- Every public method returns plain Python dicts/lists so that
  downstream agents need no knowledge of yfinance or feedparser
  internals — they just consume structured data.
- All network calls are individually try/excepted so a single
  broken feed or ticker does not abort the whole collection run.
- The RSI helper is kept private (leading underscore) because it
  is an implementation detail, not part of the agent's public API.
"""

import feedparser
import yfinance as yf
import pandas as pd
from tavily import TavilyClient


class DataCollectionAgent:
    """Collects news, market data, and macro indicators."""

    # RSS feeds for financial news.
    # Prefer primary sources over aggregators to reduce duplicate stories.
    NEWS_FEEDS = [
        "https://feeds.reuters.com/reuters/businessNews",
        "https://feeds.bloomberg.com/markets/news.rss",
        "https://www.ft.com/rss/home",
        "https://feeds.wsj.com/wsj/xml/rss/3_7085.xml",
    ]

    # Macro tickers and their human-readable names.
    # These move first on macro events — watch them before individual stocks.
    MACRO_SYMBOLS = {
        "SP500":        "^GSPC",     # US equity benchmark
        "NASDAQ":       "^IXIC",     # Tech-heavy benchmark
        "VIX":          "^VIX",      # Implied volatility / fear gauge
        "DXY":          "DX-Y.NYB",  # US Dollar Index
        "GOLD":         "GC=F",      # Safe-haven demand
        "OIL":          "CL=F",      # Geopolitical sensitivity
        "10Y_TREASURY": "^TNX",      # Rate-expectation proxy
    }

    def __init__(self, config: dict) -> None:
        """
        Parameters
        ----------
        config : dict
            Must contain a ``tavily_api_key`` entry.
        """
        self.tavily = TavilyClient(api_key=config["tavily_api_key"])

    # ── Public API ──────────────────────────────────────────────────────

    def get_breaking_news(self) -> list[dict]:
        """
        Parse up to 10 entries from each RSS feed.

        Returns
        -------
        list[dict]
            Each dict has keys: title, summary, published, source, link.
            Articles from all feeds are combined into a single flat list.
        """
        articles: list[dict] = []

        for feed_url in self.NEWS_FEEDS:
            try:
                feed = feedparser.parse(feed_url)
                for entry in feed.entries[:10]:
                    articles.append({
                        "title":     entry.get("title", ""),
                        "summary":   entry.get("summary", ""),
                        "published": entry.get("published", ""),
                        "source":    feed_url,
                        "link":      entry.get("link", ""),
                    })
            except Exception as exc:
                # Log and continue — a broken feed should not stop collection.
                print(f"[DataAgent] Feed error ({feed_url}): {exc}")

        return articles

    def search_event_context(self, query: str) -> dict:
        """
        Deep-search a specific event via Tavily for richer context.

        This is an *optional* enrichment step. If Tavily is unavailable,
        the analysis agent can still operate on the RSS articles alone.

        Parameters
        ----------
        query : str
            A natural-language description of the event to research.

        Returns
        -------
        dict
            Raw Tavily response containing ranked search results.
        """
        return self.tavily.search(
            query=query,
            search_depth="advanced",
            include_domains=[
                "reuters.com", "bloomberg.com", "wsj.com",
                "ft.com", "cnbc.com", "marketwatch.com",
            ],
            max_results=10,
        )

    def get_asset_data(self, symbols: list[str]) -> dict:
        """
        Retrieve price history and basic technicals for a list of tickers.

        Technicals computed
        -------------------
        - SMA-20  : 20-day simple moving average of closing price
        - RSI-14  : 14-period Relative Strength Index

        Parameters
        ----------
        symbols : list[str]
            Ticker symbols recognised by Yahoo Finance (e.g. "AAPL").

        Returns
        -------
        dict
            Keyed by symbol. Each value is a dict of fundamentals and
            technical readings, or {"error": "<message>"} on failure.
        """
        market_data: dict = {}

        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info   = ticker.info
                hist   = ticker.history(period="30d")

                # Compute technicals on the price series.
                hist["SMA_20"] = hist["Close"].rolling(20).mean()
                hist["RSI"]    = self._calculate_rsi(hist["Close"])

                # Safely extract the last non-NaN value for each series.
                rsi_latest   = float(hist["RSI"].iloc[-1])   if not hist.empty else None
                sma_latest   = float(hist["SMA_20"].iloc[-1]) if not hist.empty else None

                market_data[symbol] = {
                    "current_price": info.get("currentPrice", 0),
                    "market_cap":    info.get("marketCap", 0),
                    "pe_ratio":      info.get("trailingPE"),
                    "52w_high":      info.get("fiftyTwoWeekHigh", 0),
                    "52w_low":       info.get("fiftyTwoWeekLow", 0),
                    "volume":        info.get("volume", 0),
                    "avg_volume":    info.get("averageVolume", 0),
                    "beta":          info.get("beta", 1),
                    "rsi":           rsi_latest,
                    "sma_20":        sma_latest,
                    "sector":        info.get("sector", "Unknown"),
                }
            except Exception as exc:
                # Record the error but keep processing remaining symbols.
                market_data[symbol] = {"error": str(exc)}

        return market_data

    def get_macro_indicators(self) -> dict:
        """
        Fetch the latest value and 1-day percentage change for each
        macro indicator defined in ``MACRO_SYMBOLS``.

        Macro data is crucial context for sizing and direction decisions.
        For example, a VIX spike above 30 signals elevated fear — the
        system should reduce position sizes in response.

        Returns
        -------
        dict
            Keyed by indicator name (e.g. "VIX"). Each value contains
            ``{"value": float, "change_pct": float}``.
        """
        indicators: dict = {}

        for name, symbol in self.MACRO_SYMBOLS.items():
            try:
                hist = yf.Ticker(symbol).history(period="5d")
                if hist.empty:
                    continue

                current  = hist["Close"].iloc[-1]
                previous = hist["Close"].iloc[-2] if len(hist) > 1 else current

                indicators[name] = {
                    "value":      float(current),
                    "change_pct": float(((current - previous) / previous) * 100),
                }
            except Exception as exc:
                print(f"[DataAgent] Macro fetch error ({name}): {exc}")

        return indicators

    # ── Private Helpers ─────────────────────────────────────────────────

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        Compute the Relative Strength Index.

        RSI > 70  →  potentially overbought (consider short / avoid buy)
        RSI < 30  →  potentially oversold  (consider long / avoid short)

        The standard Wilder formula uses a rolling mean of gains and
        losses over ``period`` bars. Values at the start of the series
        will be NaN until enough bars have accumulated.

        Parameters
        ----------
        prices : pd.Series
            Closing price series (daily recommended).
        period : int
            Look-back window, default 14.

        Returns
        -------
        pd.Series
            RSI values aligned with the input index.
        """
        delta = prices.diff()

        # Separate gains and losses; replace opposite-direction values with 0.
        gain = delta.where(delta > 0, 0.0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()

        rs  = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi
```

---

## agents/analysis\_agent.py

```python
"""
agents/analysis_agent.py
------------------------
AnalysisAgent — the system's reasoning layer.

This agent wraps two large language models with distinct roles:

GPT-4o  (OpenAI)  — fast, structured JSON output
    • Sentiment analysis of the news batch
    • Trade signal generation
    • Independent risk-officer validation

Claude  (Anthropic) — deeper, longer-context reasoning
    • Causal analysis of world events
    • Historical precedent matching
    • Second-order effect identification

Why two models?
---------------
Using models from different providers with different system prompts
provides a lightweight cross-check. A signal that GPT-4o generates
is then reviewed by GPT-4o *again* but from a deliberately adversarial
"risk officer" perspective. Claude adds a qualitatively different view
on causality that complements GPT-4o's pattern-matching strength.

This is not infallible — both models share training-data biases and
can hallucinate — but it raises the bar before any order is placed.
"""

import json
import logging

from openai import OpenAI
import anthropic

logger = logging.getLogger(__name__)


class AnalysisAgent:
    """Multi-model AI analysis: sentiment → causality → signals → validation."""

    def __init__(self, config: dict) -> None:
        """
        Parameters
        ----------
        config : dict
            Must contain ``openai_api_key`` and ``anthropic_api_key``.
        """
        self.openai = OpenAI(api_key=config["openai_api_key"])
        self.claude = anthropic.Anthropic(api_key=config["anthropic_api_key"])

    # ── Step 1: Sentiment Analysis (GPT-4o — fast) ─────────────────────

    def analyze_news_sentiment(self, articles: list[dict]) -> dict:
        """
        Classify the overall market sentiment of the news batch and
        identify which specific assets are likely to be affected.

        GPT-4o is chosen here for latency — this runs every cycle.

        Parameters
        ----------
        articles : list[dict]
            News articles as returned by DataCollectionAgent.
            Only the first 20 are sent to stay within token limits.

        Returns
        -------
        dict
            Structured sentiment output.  Key fields:

            overall_sentiment : "bullish" | "bearish" | "neutral"
            sentiment_score   : int  (-100 to +100)
            affected_assets   : list of per-symbol dicts
            confidence        : int  (0–100)
            risk_events       : list of upcoming macro events to watch
        """
        # Flatten articles into a single text block for the prompt.
        article_text = "\n\n".join(
            f"HEADLINE: {a['title']}\nSUMMARY: {a['summary']}"
            for a in articles[:20]
        )

        prompt = f"""Analyze these financial news articles and return a JSON object.

{article_text}

Return ONLY valid JSON matching this exact schema:
{{
    "overall_sentiment": "bullish|bearish|neutral",
    "sentiment_score": <integer -100 to 100>,
    "affected_assets": [
        {{
            "symbol":    "<ticker>",
            "direction": "up|down|neutral",
            "reason":    "<one-sentence explanation>",
            "magnitude": "high|medium|low",
            "timeframe": "intraday|days|weeks"
        }}
    ],
    "sector_outlook":  {{"<sector>": "bullish|bearish|neutral"}},
    "risk_events":     ["<event description>"],
    "confidence":      <integer 0-100>,
    "key_themes":      ["<theme>"]
}}"""

        response = self.openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role":    "system",
                    "content": (
                        "You are a senior financial analyst. "
                        "Return only valid JSON — no prose, no markdown."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )

        return json.loads(response.choices[0].message.content)

    # ── Step 2: Causal Event Analysis (Claude — deep) ──────────────────

    def analyze_world_event_impact(
        self, event: str, market_data: dict
    ) -> dict:
        """
        Ask Claude to reason about the causal chain from a world event
        to specific asset prices, including non-obvious second-order effects.

        Claude is used here because of its stronger performance on long,
        multi-step reasoning tasks with historical context.

        Second-order effects are the most valuable output of this step.
        Example: an oil-price shock directly lifts energy stocks (obvious),
        but also raises costs for airlines, shipping, and consumer
        discretionary — effects that a simple sentiment scan misses.

        Parameters
        ----------
        event : str
            A brief description of the event headline.
        market_data : dict
            Current macro and asset data for grounding the analysis.

        Returns
        -------
        dict
            Structured causal analysis.  Key fields:

            severity            : "low" | "medium" | "high" | "extreme"
            direct_impacts      : list of {symbol, direction, reason}
            second_order_effects: list of {symbol, direction, reason}
            safe_haven_assets   : list of tickers
            reaction_timeline   : str
            causation_confidence: "high" | "medium" | "low"
            key_risks_to_thesis : list of strings
        """
        # Truncate market data to avoid exceeding Claude's context window.
        market_data_str = json.dumps(market_data, indent=2, default=str)[:3000]

        prompt = f"""Analyze this market-moving event and its investment implications.

EVENT: {event}

CURRENT MARKET CONTEXT:
{market_data_str}

Provide a thorough analysis covering:
1. Historical precedents for similar events (be specific: dates, magnitudes)
2. Direct asset impacts with ticker symbols and reasoning
3. Second-order effects — what else gets affected indirectly and why
4. Expected safe-haven flows (which assets benefit from risk-off moves)
5. Timeline for the market reaction to play out
6. Where your analysis may confuse correlation with causation

Return ONLY valid JSON matching this schema:
{{
    "event_type":              "geopolitical|economic|natural|political|company",
    "severity":                "low|medium|high|extreme",
    "historical_precedents":   ["<event>: <outcome>"],
    "direct_impacts":          [{{"symbol": "<ticker>", "direction": "up|down", "reason": "<text>"}}],
    "second_order_effects":    [{{"symbol": "<ticker>", "direction": "up|down", "reason": "<text>"}}],
    "safe_haven_assets":       ["<ticker>"],
    "reaction_timeline":       "intraday|1-3 days|1-2 weeks",
    "causation_confidence":    "high|medium|low",
    "key_risks_to_thesis":     ["<risk>"]
}}"""

        message = self.claude.messages.create(
            model="claude-opus-4-5",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )

        raw = message.content[0].text
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            # If Claude wraps JSON in markdown fences, strip them and retry.
            logger.warning("[AnalysisAgent] Claude response was not pure JSON — attempting cleanup.")
            cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError:
                # Return raw text so the orchestrator can log it without crashing.
                logger.error("[AnalysisAgent] Could not parse Claude response as JSON.")
                return {"raw_analysis": raw}

    # ── Step 3: Trade Signal Generation (GPT-4o) ───────────────────────

    def generate_trade_signals(
        self,
        sentiment:      dict,
        event_analysis: dict,
        market_data:    dict,
        portfolio:      dict,
    ) -> dict:
        """
        Synthesise sentiment and event analysis into actionable trade signals.

        Each signal includes a mandatory stop-loss and a take-profit target.
        The model is instructed to enforce the minimum risk/reward ratio and
        to reduce quantity when VIX is elevated.

        Parameters
        ----------
        sentiment      : output of analyze_news_sentiment()
        event_analysis : output of analyze_world_event_impact()
        market_data    : dict with "macro" and "assets" sub-keys
        portfolio      : current portfolio state from TradingAgent

        Returns
        -------
        dict
            Contains a "signals" list and a "risk_summary" string.
            Each signal dict includes: symbol, action, quantity, order_type,
            limit_price, stop_loss, take_profit, rationale, priority,
            risk_reward_ratio, max_loss_dollars.
        """
        # Bundle context into a single object; truncate to stay within tokens.
        context = {
            "sentiment":         sentiment,
            "event_analysis":    event_analysis,
            "current_positions": portfolio.get("positions", {}),
            "buying_power":      portfolio.get("buying_power", 0),
            "macro":             market_data.get("macro", {}),
        }
        context_str = json.dumps(context, indent=2, default=str)[:4000]

        prompt = f"""Generate precise trade signals based on the following context.

{context_str}

Rules you MUST follow:
- Maximum 5 % of portfolio value per position
- Every signal MUST include a stop_loss price
- Minimum risk/reward ratio of 2.0
- If VIX > 30, halve the quantity of every signal
- Only include signals where the thesis is clear and well-supported

Return ONLY valid JSON matching this schema:
{{
    "signals": [
        {{
            "symbol":           "<ticker>",
            "action":           "buy|sell|short|cover",
            "quantity":         <integer shares>,
            "order_type":       "market|limit",
            "limit_price":      <float or null>,
            "stop_loss":        <float>,
            "take_profit":      <float>,
            "rationale":        "<concise explanation>",
            "priority":         "high|medium|low",
            "risk_reward_ratio": <float>,
            "max_loss_dollars": <float>
        }}
    ],
    "risk_summary": "<overall risk assessment for this batch>"
}}"""

        response = self.openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role":    "system",
                    "content": (
                        "You are a systematic quantitative trader. "
                        "Generate only high-conviction signals. "
                        "Return valid JSON only — no markdown, no prose."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )

        return json.loads(response.choices[0].message.content)

    # ── Step 4: Independent Risk Validation (GPT-4o as risk officer) ───

    def validate_trade(self, trade: dict, portfolio: dict) -> dict:
        """
        Run an independent AI review of a proposed trade from the
        perspective of a risk officer whose goal is capital preservation.

        The deliberate change in system prompt ("conservative risk officer"
        vs. "systematic trader") creates a different decision frame that
        can catch risks the signal-generator overlooked.

        Parameters
        ----------
        trade     : a single signal dict from generate_trade_signals()
        portfolio : current portfolio state

        Returns
        -------
        dict
            approved            : bool
            modified_trade      : dict or None  (size-reduced version)
            rejection_reason    : str or None
            risk_score          : int  (0 = safe, 100 = extremely risky)
            warnings            : list[str]
            recommended_quantity: int
        """
        prompt = f"""Review this proposed trade from a risk-management perspective.

PROPOSED TRADE:
{json.dumps(trade, indent=2)}

CURRENT PORTFOLIO:
{json.dumps(portfolio, indent=2, default=str)}

Evaluate:
- Is the position size within safe limits (≤ 5 % of portfolio)?
- Is the stop-loss tight enough to protect capital?
- Is the risk/reward ratio at least 2:1?
- Does this trade create dangerous correlation with existing positions?
- Are market conditions (VIX, trend) supportive?

Return ONLY valid JSON matching this schema:
{{
    "approved":             <true|false>,
    "modified_trade":       <{{...}} or null>,
    "rejection_reason":     "<reason or null>",
    "risk_score":           <integer 0-100>,
    "warnings":             ["<warning>"],
    "recommended_quantity": <integer>
}}"""

        response = self.openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role":    "system",
                    "content": (
                        "You are a conservative risk officer. "
                        "Your primary duty is to protect capital — not maximise returns. "
                        "Reject or reduce any trade that poses excessive risk. "
                        "Return valid JSON only."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )

        return json.loads(response.choices[0].message.content)
```

---

## agents/trading\_agent.py

```python
"""
agents/trading_agent.py
-----------------------
TradingAgent — the system's execution layer.

This agent is the only component that interacts with the brokerage API.
Keeping all order-management code in one place makes it easier to audit,
test, and (eventually) swap brokers without touching any analysis logic.

Hard risk limits
----------------
The ``check_hard_risk_limits`` method enforces non-negotiable circuit
breakers.  If daily loss or total portfolio risk thresholds are breached,
``execute_trade`` refuses to submit any further orders for the session.
These limits are deliberately not overridable by AI output — they exist
precisely because AI models can generate confidently wrong signals.

Stop-loss discipline
--------------------
Every buy order is immediately followed by a GTC stop-loss order.
If the stop submission fails (e.g. API glitch), the failure is logged
loudly rather than silently ignored, and the orchestrator should be
alerted to close the position manually.
"""

import logging
from datetime import datetime

import alpaca_trade_api as tradeapi

logger = logging.getLogger(__name__)


class TradingAgent:
    """Executes trades via Alpaca and enforces hard risk limits."""

    def __init__(self, config: dict) -> None:
        """
        Parameters
        ----------
        config : dict
            Must contain alpaca_api_key, alpaca_secret, alpaca_base_url,
            max_daily_loss_pct, and max_portfolio_risk.
        """
        self.api = tradeapi.REST(
            config["alpaca_api_key"],
            config["alpaca_secret"],
            config["alpaca_base_url"],
            api_version="v2",
        )
        self.config    = config
        self.trade_log: list[dict] = []  # In-memory log; also written to disk by orchestrator

    # ── Portfolio State ─────────────────────────────────────────────────

    def get_portfolio_state(self) -> dict:
        """
        Fetch a snapshot of the account from Alpaca.

        Returns
        -------
        dict
            equity, buying_power, portfolio_value, daily_pnl, positions.
            ``positions`` is keyed by symbol; each value contains qty,
            current_price, market_value, unrealized_pnl, avg_entry_price.
        """
        account   = self.api.get_account()
        positions = self.api.list_positions()

        return {
            "equity":          float(account.equity),
            "buying_power":    float(account.buying_power),
            "portfolio_value": float(account.portfolio_value),
            # Daily P&L = today's equity minus yesterday's closing equity.
            "daily_pnl":       float(account.equity) - float(account.last_equity),
            "positions": {
                p.symbol: {
                    "qty":              float(p.qty),
                    "current_price":    float(p.current_price),
                    "market_value":     float(p.market_value),
                    "unrealized_pnl":   float(p.unrealized_pl),
                    "avg_entry_price":  float(p.avg_entry_price),
                }
                for p in positions
            },
        }

    # ── Risk Gate ───────────────────────────────────────────────────────

    def check_hard_risk_limits(self, portfolio: dict) -> tuple[bool, str]:
        """
        Enforce the two non-negotiable circuit breakers.

        Breaker 1 — Daily loss limit
            If today's losses have consumed ``max_daily_loss_pct`` of
            portfolio value, trading stops for the remainder of the day.

        Breaker 2 — Total portfolio risk
            If the sum of all open unrealised losses exceeds
            ``max_portfolio_risk``, no new positions are opened.

        Parameters
        ----------
        portfolio : dict
            As returned by get_portfolio_state().

        Returns
        -------
        (True, "OK")              — all limits are within bounds
        (False, "<reason string>")— a limit has been breached
        """
        pv = portfolio["portfolio_value"]
        if pv <= 0:
            return False, "Portfolio value is zero or negative. Cannot trade."

        # Breaker 1: Daily loss
        daily_loss     = min(portfolio["daily_pnl"], 0)   # Negative number or zero
        daily_loss_pct = abs(daily_loss) / pv

        if daily_loss_pct >= self.config["max_daily_loss_pct"]:
            msg = (
                f"Daily loss limit reached ({daily_loss_pct:.2%} ≥ "
                f"{self.config['max_daily_loss_pct']:.2%}). "
                f"Trading halted for this session."
            )
            logger.critical(msg)
            return False, msg

        # Breaker 2: Total unrealised risk across all open positions
        total_at_risk = sum(
            abs(pos["unrealized_pnl"])
            for pos in portfolio["positions"].values()
            if pos["unrealized_pnl"] < 0
        )
        portfolio_risk_pct = total_at_risk / pv

        if portfolio_risk_pct >= self.config["max_portfolio_risk"]:
            msg = (
                f"Portfolio risk limit reached ({portfolio_risk_pct:.2%} ≥ "
                f"{self.config['max_portfolio_risk']:.2%}). "
                f

### Planning Agent Output
# AI-Powered Event-Driven Trading System: Detailed Action Steps

---

## PHASE 1: Environment Setup

### Step 1 — Create Project Directory Structure

```bash
mkdir trading_system
cd trading_system
mkdir agents logs tests
touch agents/__init__.py
touch logs/.gitkeep
```

**What you're doing:** Establishing the folder layout the rest of
the system depends on. Every import path in later files assumes
this structure exists.

---

### Step 2 — Create and Activate a Virtual Environment

```bash
python -m venv venv

# Mac/Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

**Why this matters:** Isolates your dependencies from other Python
projects on your machine. Always activate this before working on
the system.

---

### Step 3 — Install All Dependencies

```bash
pip install openai anthropic
pip install yfinance pandas numpy feedparser
pip install requests beautifulsoup4 tavily-python
pip install alpaca-trade-api python-dotenv
pip install schedule scikit-learn
```

**Verification check — run this after installation:**

```python
# Run in terminal: python -c "import openai, anthropic, yfinance,
# alpaca_trade_api; print('All imports OK')"
```

If any import fails, install that package individually and check
for version conflicts with `pip list`.

---

### Step 4 — Gather All Required API Keys

Open accounts and collect keys from each service before writing
any code. You need all of these:

| Service | Purpose | Where to get it |
|---------|---------|-----------------|
| OpenAI | GPT-4o analysis | platform.openai.com |
| Anthropic | Claude analysis | console.anthropic.com |
| Tavily | Web search | tavily.com |
| NewsAPI | RSS backup | newsapi.org |
| Alpaca | Paper trading | alpaca.markets |

**Alpaca-specific steps:**
1. Sign up at alpaca.markets
2. Navigate to Paper Trading (not live)
3. Generate API key and secret from the Paper Trading dashboard
4. Copy the paper trading base URL exactly:
   `https://paper-api.alpaca.markets`

> ⚠️ **Do not use your live Alpaca keys at any point during
> development. The paper URL is the safety mechanism.**

---

### Step 5 — Create the `.env` File

```bash
# In trading_system/ directory:
touch .env
```

Paste this into `.env` and fill in your actual keys:

```env
OPENAI_API_KEY=sk-your-key-here
ANTHROPIC_API_KEY=sk-ant-your-key-here
TAVILY_API_KEY=tvly-your-key-here
NEWS_API_KEY=your-newsapi-key-here
ALPACA_API_KEY=your-alpaca-paper-key
ALPACA_SECRET=your-alpaca-paper-secret
```

---

### Step 6 — Create `.gitignore` to Protect Your Keys

```bash
touch .gitignore
```

Add this content:

```
.env
venv/
__pycache__/
*.pyc
logs/*.json
```

> ⚠️ **Never commit `.env` to Git. API keys exposed in public
> repos get scraped and abused within minutes.**

---

### Step 7 — Create `config.py`

Create `trading_system/config.py` with this content:

```python
# config.py
import os
from dotenv import load_dotenv

load_dotenv()

CONFIG = {
    # AI Models
    "openai_api_key": os.getenv("OPENAI_API_KEY"),
    "anthropic_api_key": os.getenv("ANTHROPIC_API_KEY"),

    # Data Sources
    "tavily_api_key": os.getenv("TAVILY_API_KEY"),
    "news_api_key": os.getenv("NEWS_API_KEY"),

    # Broker — PAPER TRADING ONLY until validated
    "alpaca_api_key": os.getenv("ALPACA_API_KEY"),
    "alpaca_secret": os.getenv("ALPACA_SECRET"),
    "alpaca_base_url": "https://paper-api.alpaca.markets",

    # Hard Risk Limits
    "max_position_pct": 0.05,
    "max_daily_loss_pct": 0.02,
    "max_portfolio_risk": 0.15,
    "min_risk_reward": 2.0,
}
```

**Verification check:**

```python
# python -c "from config import CONFIG;
# print(CONFIG['alpaca_base_url'])"
# Should print: https://paper-api.alpaca.markets
```

If it prints `None` for any key, your `.env` file has a typo or
the variable name doesn't match.

---

## PHASE 2: Data Collection Agent

### Step 8 — Create `agents/data_agent.py`

Create the file and add the full `DataCollectionAgent` class. The
class has four methods. Build them in this order so you can test
each one independently.

```python
# agents/data_agent.py
import feedparser
import yfinance as yf
import pandas as pd
from tavily import TavilyClient

class DataCollectionAgent:
    def __init__(self, config):
        self.tavily = TavilyClient(api_key=config["tavily_api_key"])
        self.news_feeds = [
            "https://feeds.reuters.com/reuters/businessNews",
            "https://feeds.bloomberg.com/markets/news.rss",
            "https://www.ft.com/rss/home",
            "https://feeds.wsj.com/wsj/xml/rss/3_7085.xml",
        ]
```

---

### Step 9 — Add `get_breaking_news()` Method

Add this method inside the class from Step 8:

```python
    def get_breaking_news(self) -> list[dict]:
        articles = []
        for url in self.news_feeds:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:10]:
                    articles.append({
                        "title": entry.get("title", ""),
                        "summary": entry.get("summary", ""),
                        "published": entry.get("published", ""),
                        "source": url,
                        "link": entry.get("link", ""),
                    })
            except Exception as e:
                print(f"Feed error {url}: {e}")
        return articles
```

**Test this method immediately:**

```python
# test_news.py (run from project root)
from agents.data_agent import DataCollectionAgent
from config import CONFIG

agent = DataCollectionAgent(CONFIG)
articles = agent.get_breaking_news()
print(f"Got {len(articles)} articles")
print(articles[0] if articles else "No articles returned")
```

Expected output: 20-40 articles. If you get 0, the RSS feeds may
be blocking requests or temporarily down — this is normal and the
try/except handles it gracefully.

---

### Step 10 — Add `search_event_context()` Method

```python
    def search_event_context(self, query: str) -> dict:
        return self.tavily.search(
            query=query,
            search_depth="advanced",
            include_domains=[
                "reuters.com", "bloomberg.com", "wsj.com",
                "ft.com", "cnbc.com", "marketwatch.com"
            ],
            max_results=10
        )
```

**Test:**

```python
# Add to test_news.py
result = agent.search_event_context("Federal Reserve interest rate decision")
print(f"Search returned {len(result.get('results', []))} results")
```

---

### Step 11 — Add `_calculate_rsi()` Helper Method

Add this before `get_asset_data()` since it's a dependency:

```python
    def _calculate_rsi(self, prices, period=14):
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
```

RSI above 70 suggests overbought, below 30 suggests oversold.
This is a widely-used momentum indicator that feeds into the
signal generation later.

---

### Step 12 — Add `get_asset_data()` Method

```python
    def get_asset_data(self, symbols: list[str]) -> dict:
        market_data = {}
        for symbol in symbols:
            try:
                ticker = yf.Ticker(symbol)
                info = ticker.info
                hist = ticker.history(period="30d")

                hist["SMA_20"] = hist["Close"].rolling(20).mean()
                hist["RSI"] = self._calculate_rsi(hist["Close"])

                market_data[symbol] = {
                    "current_price": info.get("currentPrice", 0),
                    "market_cap": info.get("marketCap", 0),
                    "pe_ratio": info.get("trailingPE"),
                    "52w_high": info.get("fiftyTwoWeekHigh", 0),
                    "52w_low": info.get("fiftyTwoWeekLow", 0),
                    "volume": info.get("volume", 0),
                    "avg_volume": info.get("averageVolume", 0),
                    "beta": info.get("beta", 1),
                    "rsi": float(hist["RSI"].iloc[-1]) if not hist.empty else None,
                    "sma_20": float(hist["SMA_20"].iloc[-1]) if not hist.empty else None,
                    "sector": info.get("sector", "Unknown"),
                }
            except Exception as e:
                market_data[symbol] = {"error": str(e)}
        return market_data
```

**Test with a small symbol list first:**

```python
# test_market.py
from agents.data_agent import DataCollectionAgent
from config import CONFIG

agent = DataCollectionAgent(CONFIG)
data = agent.get_asset_data(["AAPL", "SPY"])
for symbol, d in data.items():
    print(f"{symbol}: ${d.get('current_price')} | RSI: {d.get('rsi'):.1f}")
```

yfinance can be slow on the first call — 5-10 seconds per symbol
is normal. If you get errors, check that the symbol is valid on
Yahoo Finance.

---

### Step 13 — Add `get_macro_indicators()` Method

```python
    def get_macro_indicators(self) -> dict:
        indices = {
            "SP500": "^GSPC",
            "NASDAQ": "^IXIC",
            "VIX": "^VIX",
            "DXY": "DX-Y.NYB",
            "GOLD": "GC=F",
            "OIL": "CL=F",
            "10Y_TREASURY": "^TNX",
        }
        indicators = {}
        for name, symbol in indices.items():
            try:
                hist = yf.Ticker(symbol).history(period="5d")
                if not hist.empty:
                    cur = hist["Close"].iloc[-1]
                    prev = hist["Close"].iloc[-2] if len(hist) > 1 else cur
                    indicators[name] = {
                        "value": cur,
                        "change_pct": ((cur - prev) / prev) * 100
                    }
            except:
                pass
        return indicators
```

**Test:**

```python
# test_macro.py
from agents.data_agent import DataCollectionAgent
from config import CONFIG

agent = DataCollectionAgent(CONFIG)
macro = agent.get_macro_indicators()
for name, data in macro.items():
    print(f"{name}: {data['value']:.2f} ({data['change_pct']:+.2f}%)")
```

VIX is the most important number here. Values above 30 indicate
high fear in the market — the orchestrator uses this to reduce
position sizes automatically.

---

## PHASE 3: Analysis Agent

### Step 14 — Create `agents/analysis_agent.py` with Imports

```python
# agents/analysis_agent.py
from openai import OpenAI
import anthropic
import json

class AnalysisAgent:
    def __init__(self, config):
        self.openai = OpenAI(api_key=config["openai_api_key"])
        self.claude = anthropic.Anthropic(api_key=config["anthropic_api_key"])
```

**Verify API connections before adding any methods:**

```python
# test_apis.py
from agents.analysis_agent import AnalysisAgent
from config import CONFIG

agent = AnalysisAgent(CONFIG)

# Test OpenAI
r = agent.openai.chat.completions.create(
    model="gpt-4o",
    messages=[{"role": "user", "content": "Say OK"}]
)
print("OpenAI:", r.choices[0].message.content)

# Test Anthropic
r = agent.claude.messages.create(
    model="claude-opus-4-5",
    max_tokens=10,
    messages=[{"role": "user", "content": "Say OK"}]
)
print("Claude:", r.content[0].text)
```

Both should print "OK". If either throws an authentication error,
recheck your `.env` file and confirm the key is active in the
respective dashboard.

---

### Step 15 — Add `analyze_news_sentiment()` Method

This is the fastest-running method — it runs every cycle:

```python
    def analyze_news_sentiment(self, articles: list[dict]) -> dict:
        text = "\n\n".join([
            f"HEADLINE: {a['title']}\nSUMMARY: {a['summary']}"
            for a in articles[:20]
        ])

        response = self.openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a financial analyst. Return JSON only."
                },
                {
                    "role": "user",
                    "content": f"""Analyze these articles:

{text}

Return JSON:
{{
    "overall_sentiment": "bullish|bearish|neutral",
    "sentiment_score": -100 to 100,
    "affected_assets": [
        {{
            "symbol": "XLE",
            "direction": "up|down|neutral",
            "reason": "...",
            "magnitude": "high|medium|low",
            "timeframe": "intraday|days|weeks"
        }}
    ],
    "sector_outlook": {{"Energy": "bearish", "Technology": "bullish"}},
    "risk_events": ["Fed meeting", "CPI release"],
    "confidence": 70,
    "key_themes": ["inflation", "rate hike expectations"]
}}"""
                }
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
```

**Test:**

```python
# test_sentiment.py
from agents.data_agent import DataCollectionAgent
from agents.analysis_agent import AnalysisAgent
from config import CONFIG

data_agent = DataCollectionAgent(CONFIG)
analysis_agent = AnalysisAgent(CONFIG)

articles = data_agent.get_breaking_news()
sentiment = analysis_agent.analyze_news_sentiment(articles)

print(f"Sentiment: {sentiment['overall_sentiment']}")
print(f"Score: {sentiment['sentiment_score']}")
print(f"Affected assets: {len(sentiment.get('affected_assets', []))}")
```

Common issue: If you get a JSON parsing error, the model returned
invalid JSON. Add a try/except and print the raw response to debug.

---

### Step 16 — Add `analyze_world_event_impact()` Method

This is the slower, deeper Claude analysis — only called when
significant events are detected:

```python
    def analyze_world_event_impact(self, event: str, market_data: dict) -> dict:
        message = self.claude.messages.create(
            model="claude-opus-4-5",
            max_tokens=2000,
            messages=[{
                "role": "user",
                "content": f"""Analyze this market event:

EVENT: {event}

MARKET CONTEXT:
{json.dumps(market_data, indent=2, default=str)[:3000]}

Provide:
1. Historical precedents for similar events and their outcomes
2. Direct asset impacts (specific tickers) with reasoning
3. Second-order effects (what else gets affected indirectly)
4. Safe haven asset flows expected
5. Expected timeline of market reaction
6. Where correlation vs. causation risk exists in your analysis

Return JSON:
{{
    "event_type": "geopolitical|economic|natural|political|company",
    "severity": "low|medium|high|extreme",
    "historical_precedents": ["2022 Russia-Ukraine: Oil spiked 30%"],
    "direct_impacts": [{{"symbol": "XOM", "direction": "up", "reason": "..."}}],
    "second_order_effects": [{{"symbol": "DAL", "direction": "down", "reason": "fuel costs"}}],
    "safe_haven_assets": ["GLD", "TLT", "USD"],
    "reaction_timeline": "intraday|1-3 days|1-2 weeks",
    "causation_confidence": "high|medium|low",
    "key_risks_to_thesis": ["event reversal", "already priced in"]
}}"""
            }]
        )
        try:
            return json.loads(message.content[0].text)
        except:
            return {"raw_analysis": message.content[0].text}
```

The `default=str` in `json.dumps` handles datetime objects and
other non-serializable types that yfinance sometimes returns.

---

### Step 17 — Add `generate_trade_signals()` Method

```python
    def generate_trade_signals(
        self,
        sentiment: dict,
        event_analysis: dict,
        market_data: dict,
        portfolio: dict
    ) -> dict:
        context = {
            "sentiment": sentiment,
            "event_analysis": event_analysis,
            "current_positions": portfolio.get("positions", {}),
            "buying_power": portfolio.get("buying_power", 0),
            "macro": market_data.get("macro", {}),
        }

        response = self.openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a systematic trader. Generate precise signals. "
                        "Always include stop loss. Return JSON only."
                    )
                },
                {
                    "role": "user",
                    "content": f"""Generate trade signals:

{json.dumps(context, indent=2, default=str)[:4000]}

Rules:
- Max 5% of portfolio per position
- Every trade MUST have a stop loss
- Minimum 2:1 reward-to-risk ratio
- Reduce size if VIX > 30

Return JSON:
{{
    "signals": [
        {{
            "symbol": "NVDA",
            "action": "buy|sell|short|cover",
            "quantity": 10,
            "order_type": "market|limit",
            "limit_price": null,
            "stop_loss": 430.00,
            "take_profit": 510.00,
            "rationale": "...",
            "priority": "high|medium|low",
            "risk_reward_ratio": 2.5,
            "max_loss_dollars": 400
        }}
    ],
    "risk_summary": "..."
}}"""
                }
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
```

---

### Step 18 — Add `validate_trade()` Method

This is an independent AI call with a different system prompt —
it acts as a risk officer reviewing the signal generated in
Step 17:

```python
    def validate_trade(self, trade: dict, portfolio: dict) -> dict:
        response = self.openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a conservative risk officer. "
                        "Your job is to protect capital, not maximize returns. "
                        "Reject or modify any trade that looks risky. "
                        "Return JSON only."
                    )
                },
                {
                    "role": "user",
                    "content": f"""Review this trade:

TRADE: {json.dumps(trade, indent=2)}
PORTFOLIO: {json.dumps(portfolio, indent=2, default=str)}

Check: position size, stop loss adequacy, risk/reward ratio,
correlation with existing positions, market conditions.

Return:
{{
    "approved": true,
    "modified_trade": null,
    "rejection_reason": null,
    "risk_score": 0-100,
    "warnings": ["..."],
    "recommended_quantity": 10
}}"""
                }
            ],
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
```

**Why two separate AI calls matter:** The signal generator is
prompted to find opportunities. The validator is prompted to
protect capital. The same data, opposite objectives — this
provides a built-in cross-check against overconfident signals.

---

## PHASE 4: Trading Execution Agent

### Step 19 — Create `agents/trading_agent.py`

```python
# agents/trading_agent.py
import alpaca_trade_api as tradeapi
from datetime import datetime
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/trading.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class TradingAgent:
    def __init__(self, config):
        self.api = tradeapi.REST(
            config["alpaca_api_key"],
            config["alpaca_secret"],
            config["alpaca_base_url"],
            api_version="v2"
        )
        self.config = config
        self.trade_log = []
```

**Test the Alpaca connection immediately:**

```python
# test_alpaca.py
from agents.trading_agent import TradingAgent
from config import CONFIG

agent = TradingAgent(CONFIG)
account = agent.api.get_account()
print(f"Account status: {account.status}")
print(f"Paper equity: ${float(account.equity):,.2f}")
```

You should see a paper account with $100,000 (Alpaca's default
paper balance). If you get an authentication error, double-check
that you're using paper trading keys, not live keys.

---

### Step 20 — Add `get_portfolio_state()` Method

```python
    def get_portfolio_state(self) -> dict:
        account = self.api.get_account()
        positions = self.api.list_positions()
        return {
            "equity": float(account.equity),
            "buying_power": float(account.buying_power),
            "portfolio_value": float(account.portfolio_value),
            "daily_pnl": float(account.equity) - float(account.last_equity),
            "positions": {
                p.symbol: {
                    "qty": float(p.qty),
                    "current_price": float(p.current_price),
                    "market_value": float(p.market_value),
                    "unrealized_pnl": float(p.unrealized_pl),
                    "avg_entry_price": float(p.avg_entry_price),
                }
                for p in positions
            },
        }
```

---

### Step 21 — Add `check_hard_risk_limits()` Method

```python
    def check_hard_risk_limits(self, portfolio: dict) -> tuple[bool, str]:
        pv = portfolio["portfolio_value"]

        # Check daily loss limit
        daily_loss_pct = abs(min(portfolio["daily_pnl"], 0)) / pv
        if daily_loss_pct >= self.config["max_daily_loss_pct"]:
            return False, (
                f"Daily loss limit hit ({daily_loss_pct:.1%}). "
                f"System halted for today."
            )

        # Check total portfolio risk
        total_at_risk = sum(
            abs(p["unrealized_pnl"])
            for p in portfolio["positions"].values()
            if p["unrealized_pnl"] < 0
        )
        if pv > 0 and (total_at_risk / pv) >= self.config["max_portfolio_risk"]:
            return False, (
                f"Portfolio risk limit hit ({total_at_risk/pv:.1%}). "
                f"No new trades."
            )

        return True, "OK"
```

> ⚠️ **These limits are the most important code in the entire
> system.** Do not remove the daily loss check. Do not add logic
> that bypasses it. The system will halt trading for the day when
> 2% of portfolio value is lost — this is intentional.

---

### Step 22 — Add `_set_stop_loss()` Helper Method

```python
    def _set_stop_loss(self, symbol: str, qty: float, stop_price: float):
        try:
            self.api.submit_order(
                symbol=symbol,
                qty=qty,
                side="sell",
                type="stop",
                stop_price=stop_price,
                time_in_force="gtc"
            )
            logger.info(f"Stop loss set: {symbol} @ ${stop_price:.2f}")
        except Exception as e:
            logger.error(f"CRITICAL: Stop loss failed for {symbol}: {e}")
            # Log but don't raise — main order is already submitted
```

The `gtc` (good-till-cancelled) time in force means this stop
order persists across trading sessions until it fills or you
cancel it manually.

---

### Step 23 — Add `execute_trade()` Method

```python
    def execute_trade(self, signal: dict, portfolio: dict) -> dict:
        # Hard limits checked first — no exceptions
        ok, reason = self.check_hard_risk_limits(portfolio)
        if not ok:
            logger.warning(f"Trade blocked: {reason}")
            return {"success": False, "reason": reason}

        symbol = signal["symbol"]
        side = "buy" if signal["action"] in ["buy", "cover"] else "sell"

        try:
            order_params = {
                "symbol": symbol,
                "qty": signal["quantity"],
                "side": side,
                "type": signal.get("order_type", "market"),
                "time_in_force": "day",
            }

            if signal.get("limit_price") and order_params["type"] == "limit":
                order_params["limit_price"] = signal["limit_price"]

            order = self.api.submit_order(**order_params)
            logger.info(
                f"ORDER SUBMITTED: {side.upper()} {signal['quantity']} "
                f"{symbol} | ID: {order.id}"
            )

            # Set stop loss immediately after order
            if signal.get("stop_loss") and side == "buy":
                self._set_stop_loss(symbol, signal["quantity"], signal["stop_loss"])

            record = {
                "timestamp": datetime.now().isoformat(),
                "symbol": symbol,
                "action": signal["action"],
                "quantity": signal["quantity"],
                "order_id": order.id,
                "stop_loss": signal.get("stop_loss"),
                "take_profit": signal.get("take_profit"),
                "rationale": signal.get("rationale", ""),
            }
            self.trade_log.append(record)
            return {"success": True, "order": record}

        except Exception as e:
            logger.error(f"Trade execution failed for {symbol}: {e}")
            return {"success": False, "reason": str(e)}
```

---

### Step 24 — Add `emergency_close_all()` Method

```python
    def emergency_close_all(self):
        """
        Call this manually if the system behaves unexpectedly.
        Closes all positions and cancels all open orders.
        """
        logger.warning("⚠️  EMERGENCY CLOSE ALL POSITIONS TRIGGERED")
        try:
            self.api.close_all_positions()
            self.api.cancel_all_orders()
            logger.info("All positions closed. All orders cancelled.")
        except Exception as e:
            logger.error(f"Emergency close failed: {e}")
            logger.error("LOG IN TO ALPACA DASHBOARD AND CLOSE MANUALLY")
```

**Test this method now, on paper, before going further:**

```python
# test_emergency.py
from agents.trading_agent import TradingAgent
from config import CONFIG

agent = TradingAgent(CONFIG)
agent.emergency_close_all()  # Safe on paper — nothing to close yet
print("Emergency close tested successfully")
```

Knowing this works before you need it is critical.

---

## PHASE 5: Orchestrator

### Step 25 — Create `orchestrator.py` with Imports and `__init__`

```python
# orchestrator.py
import asyncio
import schedule
import time
import json
from datetime import datetime
from agents.data_agent import DataCollectionAgent
from agents.analysis_agent import AnalysisAgent
from agents.trading_agent import TradingAgent
from config import CONFIG

class TradingOrchestrator:
    def __init__(self):
        self.data = DataCollectionAgent(CONFIG)
        self.analysis = AnalysisAgent(CONFIG)
        self.trader = TradingAgent(CONFIG)
        self.performance_log = []

        self.watchlist = [
            "AAPL", "MSFT", "NVDA", "GOOGL",
            "JPM", "GS",
            "XOM", "CVX",
            "SPY", "QQQ", "GLD", "TLT", "XLE",
        ]
```

---

### Step 26 — Add `_extract_significant_events()` Helper

```python
    def _extract_significant_events(self, articles: list) -> list:
        high_impact_keywords = [
            "fed", "interest rate", "inflation", "gdp", "jobs report",
            "earnings beat", "earnings miss", "merger", "acquisition",
            "war", "sanctions", "tariff", "recession", "bankruptcy",
            "crisis", "central bank", "powell", "rate hike", "rate cut",
        ]
        significant = []
        for article in articles:
            text = (article["title"] + " " + article["summary"]).lower()
            if any(kw in text for kw in high_impact_keywords):
                significant.append(article["title"])
        return significant[:3]
```

This runs locally with no API cost. Add keywords specific to
your trading strategy — if you're focused on energy stocks, add
"OPEC", "pipeline", "refinery" to the list.

---

### Step 27 — Add `run_cycle()` Method — Data Collection Section

Build `run_cycle()` in sections so you can test each phase
independently. Add this first section:

```python
    async def run_cycle(self):
        cycle_start = datetime.now()
        print(f"\n{'='*60}")
        print(f"CYCLE START | {cycle_start.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")

        # ── 1. Collect Data ────────────────────────────────────────────
        print("\n📡 Collecting data...")
        news = self.data.get_breaking_news()
        market_data = self.data.get_asset_data(self.watchlist)
        macro = self.data.get_macro_indicators()
        portfolio = self.trader.get_portfolio_state()

        print(
            f"   ✓ {len(news)}

**Recommended Next Step:** Save the code above and run it!
