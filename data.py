import re
from datetime import datetime, timedelta, timezone

import anthropic
import numpy as np
import pandas as pd
import yfinance as yf
from flask_caching import Cache

cache = Cache()

DEFAULT_TICKERS = "DELL KMI NEE TSM"

# (display label, yfinance info key)
INDICATORS = [
    ("P/E",                "trailingPE"),
    ("Forward P/E",        "forwardPE"),
    ("Price/Book",         "priceToBook"),
    ("Price/Sales",        "priceToSalesTrailing12Months"),
    ("Dividend per Share", "dividendRate"),
    ("EPS",                "trailingEps"),
    ("Forward EPS",        "forwardEps"),
    ("Dividend Yield",     "dividendYield"),
]

TOOLTIPS = {
    "P/E":                "Price-to-Earnings ratio: A company's share price relative to its earnings per share.",
    "Forward P/E":        "Forward Price-to-Earnings ratio: Uses projected earnings for the next 12 months.",
    "Price/Book":         "Price-to-Book ratio: Market price relative to book value per share.",
    "Price/Sales":        "Price-to-Sales ratio: Market cap relative to annual revenue.",
    "Dividend per Share": "Total dividends paid per year divided by shares outstanding.",
    "EPS":                "Earnings Per Share: Net profit divided by shares outstanding.",
    "Forward EPS":        "Estimated earnings per share for the next fiscal year.",
    "Dividend Yield":     "Annual dividends as a percentage of the current share price.",
}

_TICKER_RE = re.compile(r"^[A-Z0-9][A-Z0-9.\-]{0,9}$")


def validate_tickers(raw: str) -> list[str]:
    return [t.upper() for t in (raw or "").split() if _TICKER_RE.match(t.upper())]


@cache.memoize(timeout=300)
def fetch_ticker_info(ticker: str) -> dict:
    return yf.Ticker(ticker).info


@cache.memoize(timeout=300)
def fetch_stock_history(ticker: str, period: str) -> pd.DataFrame:
    return yf.Ticker(ticker).history(period=period).copy()


def fetch_stock_data(tickers: list[str], period: str) -> dict[str, pd.DataFrame]:
    result = {}
    for ticker in tickers:
        try:
            result[ticker] = fetch_stock_history(ticker, period)
        except Exception as e:
            print(f"Error fetching history for {ticker}: {e}")
    return result


def fetch_key_indicators(ticker: str) -> pd.DataFrame:
    info = fetch_ticker_info(ticker)
    return pd.DataFrame({
        "Indicator": [label for label, _ in INDICATORS],
        "Value":     [info.get(key) for _, key in INDICATORS],
    })


def fetch_company_info(ticker: str) -> dict:
    info = fetch_ticker_info(ticker)
    return {
        "name":        info.get("longName", ticker),
        "description": info.get("longBusinessSummary", "No description available."),
    }


def calculate_key_figures(stock_data: dict[str, pd.DataFrame]) -> dict:
    key_figures = {}
    for ticker, data in stock_data.items():
        daily_return = data["Close"].pct_change().dropna()
        vol = daily_return.std()
        key_figures[ticker] = {
            "Volatility":            vol,
            "Annualized Volatility": vol * np.sqrt(252),
        }
    return key_figures


def _normalize_news_item(raw: dict) -> dict | None:
    """Flatten a yfinance news item (nested 'content' format, or the legacy flat one)."""
    content = raw.get("content") or raw
    title = content.get("title")
    if not title:
        return None

    published = None
    if content.get("pubDate"):
        try:
            published = datetime.fromisoformat(content["pubDate"].replace("Z", "+00:00"))
        except ValueError:
            pass
    elif content.get("providerPublishTime"):
        published = datetime.fromtimestamp(content["providerPublishTime"], tz=timezone.utc)

    link = (
        (content.get("clickThroughUrl") or {}).get("url")
        or (content.get("canonicalUrl") or {}).get("url")
        or content.get("link")
    )
    if not link or not link.startswith(("https://", "http://")):
        link = None

    provider = content.get("provider")
    publisher = provider.get("displayName") if isinstance(provider, dict) else content.get("publisher")

    return {"title": title, "published": published, "link": link, "publisher": publisher}


@cache.memoize(timeout=300)
def fetch_company_news(ticker: str, days: int = 7, limit: int = 5) -> list[dict]:
    """Recent news for a ticker; returns [] if Yahoo fails or changes format."""
    try:
        raw_items = yf.Ticker(ticker).get_news(count=20)
    except Exception as e:
        print(f"Error fetching news for {ticker}: {e}")
        return []

    cutoff = datetime.now(timezone.utc) - timedelta(days=days)
    items = [n for n in map(_normalize_news_item, raw_items) if n]
    recent = [n for n in items if n["published"] is None or n["published"] >= cutoff]
    recent.sort(key=lambda n: n["published"] or cutoff, reverse=True)
    return recent[:limit]


_client = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env
    return _client


def generate_claude_analysis(
    ticker: str,
    indicators_df: pd.DataFrame,
    company_info: dict,
    keywords: str,
    stock_data: pd.DataFrame,
    news: list[dict],
) -> str:
    indicators_str = "\n".join(
        f"{row['Indicator']}: {row['Value']}"
        for _, row in indicators_df.iterrows()
    )
    price_change = ((stock_data["Close"].iloc[-1] / stock_data["Close"].iloc[0]) - 1) * 100
    volatility = stock_data["Close"].pct_change().std() * (252 ** 0.5)
    headlines = "\n".join(f"- {item['title']}" for item in news[:3])

    prompt = f"""Analyze {company_info['name']} ({ticker}) based on the following:

1. Company Description:
{company_info['description']}

2. Key Indicators:
{indicators_str}

3. Recent Performance:
- Price change over the analyzed period: {price_change:.2f}%
- Annualized volatility: {volatility * 100:.2f}%

4. Recent News Headlines:
{headlines}

5. Consider these keywords in your analysis: {keywords}

Provide a concise analysis including:
1. Brief company summary
2. Financial health analysis
3. Recent performance interpretation
4. Impact of recent news on the company's outlook
5. Forward-looking statement considering the news and industry trends
6. Investment recommendation (Buy, Hold, or Sell) with brief explanation

Use markdown for formatting. Be concise and informative."""

    message = _get_client().messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        system="You are a financial analyst with expertise in stock market analysis.",
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(b.text for b in message.content if b.type == "text")
