# Stock Performance Dashboard

A small Dash app that pulls stock data from Yahoo Finance (`yfinance`) and uses the Claude API to write a short analysis with a Buy/Hold/Sell view.

## Features
- **Key Indicators**: P/E, EPS, dividend yield, volatility and more, with tooltips.
- **Stock Performance**: closing-price charts per ticker.
- **Company News**: recent headlines per ticker.
- **Claude Analysis**: AI-written analysis built from the indicators, price history and news.

## Setup
1. Install dependencies (Python 3.10+):
   ```bash
   pip install -r requirements.txt
   ```
2. Create a `.env` file with your Anthropic API key:
   ```plaintext
   ANTHROPIC_API_KEY=your_anthropic_api_key
   ```
3. Run the app:
   ```bash
   python app.py
   ```
   Set `DASH_DEBUG=1` to enable Dash debug mode (development only).
4. Open http://127.0.0.1:8050.

## Layout
| File | Purpose |
|---|---|
| `app.py` | App setup and entry point |
| `data.py` | yfinance fetching (cached 5 min), ticker validation, Claude call |
| `layout.py` | Static page layout |
| `callbacks.py` | Dash callbacks and tab renderers |

Enter tickers separated by spaces (e.g. `AAPL MSFT BRK-B`), pick a period, and click **Generate Claude Analysis** for the AI view.

Not financial advice.
