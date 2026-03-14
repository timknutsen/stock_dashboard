import re
from datetime import datetime, timedelta

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


def fetch_company_news(ticker: str, days: int = 7) -> list[dict]:
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    news = yf.Ticker(ticker).news

    def in_range(item):
        return start_date <= datetime.fromtimestamp(item["providerPublishTime"]) <= end_date

    # Prefer articles explicitly tagged with this ticker
    filtered = [
        item for item in news
        if in_range(item)
        and ticker.upper() in [t.upper() for t in item.get("relatedTickers", [])]
    ]
    # Yahoo's backend sometimes omits relatedTickers; fall back to date filter
    if not filtered:
        filtered = [item for item in news if in_range(item)]

    filtered.sort(key=lambda x: x["providerPublishTime"], reverse=True)
    return filtered[:5]


_client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env


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
- Annualized volatility: {volatility:.2f}%

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

    message = _client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=2000,
        system="You are a financial analyst with expertise in stock market analysis.",
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text
