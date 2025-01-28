import os
from dotenv import load_dotenv
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objs as go
import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
from dash import dash_table
import dash_bootstrap_components as dbc
import openai
from dash import callback_context
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Define default tickers
DEFAULT_TICKERS = "DELL KMI NEE TSM"

# Define color palette
colors = {
    'primary': '#007bff',
    'secondary': '#6c757d',
    'success': '#28a745',
    'info': '#17a2b8',
    'warning': '#ffc107',
    'danger': '#dc3545',
    'light': '#f8f9fa',
    'dark': '#343a40',
}

# Fetch stock data with error handling
def fetch_stock_data(tickers, period='1y'):
    stock_data = {}
    for ticker in tickers:
        try:
            stock_data[ticker] = yf.Ticker(ticker).history(period=period)
        except Exception as e:
            print(f"Error fetching data for {ticker}: {str(e)}")
    return stock_data

# Fetch key financial indicators
def fetch_key_indicators(ticker):
    stock = yf.Ticker(ticker)
    indicators = {
        'Indicator': ['P/E', 'Forward P/E', 'Price/Book', 'Price/Sales', 'Dividend per Share', 'EPS', 'Forward EPS', 'Dividend Yield', 'PEG Ratio'],
        'Value': [
            stock.info.get('trailingPE'),
            stock.info.get('forwardPE'),
            stock.info.get('priceToBook'),
            stock.info.get('priceToSalesTrailing12Months'),
            stock.info.get('dividendRate'),
            stock.info.get('trailingEps'),
            stock.info.get('forwardEps'),
            stock.info.get('dividendYield'),
            stock.info.get('pegRatio')
        ]
    }
    return pd.DataFrame(indicators)

# Fetch company info
def fetch_company_info(ticker):
    stock = yf.Ticker(ticker)
    company_info = {
        'name': stock.info.get('longName', ticker),
        'description': stock.info.get('longBusinessSummary', 'No description available.')
    }
    return company_info

# Calculate key figures including volatility
def calculate_key_figures(stock_data):
    key_figures = {}
    for ticker, data in stock_data.items():
        data['Daily Return'] = data['Close'].pct_change()
        data = data.dropna()
        volatility = data['Daily Return'].std()
        annualized_volatility = volatility * np.sqrt(252)
        key_figures[ticker] = {
            'Volatility': volatility,
            'Annualized Volatility': annualized_volatility
        }
    return key_figures

# Fetch company news
def fetch_company_news(ticker, days=7):
    stock = yf.Ticker(ticker)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    news = stock.news
    
    filtered_news = [
        item for item in news
        if start_date <= datetime.fromtimestamp(item['providerPublishTime']) <= end_date
    ]
    
    filtered_news.sort(key=lambda x: x['providerPublishTime'], reverse=True)
    
    return filtered_news[:5]

# Generate GPT analysis
def generate_gpt_analysis(ticker, indicators_df, company_info, keywords, stock_data, news):
    company_name = company_info['name']
    company_description = company_info['description']
    
    indicators_str = "\n".join([f"{row['Indicator']}: {row['Value']}" for _, row in indicators_df.iterrows()])
    
    recent_price_change = ((stock_data['Close'][-1] / stock_data['Close'][0]) - 1) * 100
    
    volatility = stock_data['Close'].pct_change().std() * (252 ** 0.5)
    
    news_headlines = "\n".join([f"- {item['title']}" for item in news[:3]])
    
    client = openai.OpenAI(api_key=openai.api_key)
    messages = [
        {"role": "system", "content": "You are a financial analyst with expertise in stock market analysis."},
        {"role": "user", "content": f"""Analyze {company_name} ({ticker}) based on the following:

1. Company Description:
{company_description}

2. Key Indicators:
{indicators_str}

3. Recent Performance:
- Price change over the analyzed period: {recent_price_change:.2f}%
- Annualized volatility: {volatility:.2f}%

4. Recent News Headlines:
{news_headlines}

5. Consider these keywords in your analysis: {keywords}

Provide a concise analysis including:
1. Brief company summary
2. Financial health analysis
3. Recent performance interpretation
4. Impact of recent news on the company's outlook
5. Forward-looking statement considering the news and industry trends
6. Investment recommendation (Buy, Hold, or Sell) with brief explanation

Use markdown for formatting. Be concise and informative."""}
    ]
    
    response = client.chat.completions.create(
        messages=messages,
        model="gpt-3.5-turbo",
        max_tokens=2000
    )
    gpt_response = response.choices[0].message.content.strip()
    
    return gpt_response

# Input validation function
def validate_tickers(tickers):
    valid_tickers = []
    for ticker in tickers.split():
        if ticker.isalpha():
            valid_tickers.append(ticker.upper())
    return valid_tickers

# Tooltip function
def get_tooltip(indicator):
    tooltips = {
        'P/E': "Price-to-Earnings ratio: A company's share price relative to its earnings per share.",
        'Forward P/E': "Forward Price-to-Earnings ratio: Using projected earnings for the next 12 months.",
        'Price/Book': "Price-to-Book ratio: The ratio of market price to book value per share.",
        'Price/Sales': "Price-to-Sales ratio: The ratio of a company's market cap to its revenue.",
        'Dividend per Share': "The total dividends paid out over an entire year divided by the number of outstanding ordinary shares.",
        'EPS': "Earnings Per Share: The company's profit divided by the number of outstanding shares of its common stock.",
        'Forward EPS': "Forward Earnings Per Share: The estimated earnings per share based on projections for the next fiscal year.",
        'Dividend Yield': "A financial ratio that shows how much a company pays out in dividends each year relative to its stock price.",
        'PEG Ratio': "Price/Earnings to Growth ratio: A stock's P/E ratio divided by the growth rate of its earnings for a specified time period."
    }
    return tooltips.get(indicator, "No explanation available.")

# Initialize Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP, 'https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&family=Montserrat:wght@500;700&display=swap'])

# Add custom CSS
app.css.append_css({"external_url": "/assets/custom.css"})

# Layout of the app
app.layout = dbc.Container([
    html.H1("Stock Performance Analysis", className="text-center mb-4"),
    dbc.Row([
        dbc.Col([
            dbc.Card([
                dbc.CardBody([
                    dcc.Input(id='input-tickers', value=DEFAULT_TICKERS, type='text', placeholder='Enter stock tickers', className="form-control mb-2"),
                    dcc.Dropdown(
                        id='input-period',
                        options=[
                            {'label': '1 Year', 'value': '1y'},
                            {'label': '6 Months', 'value': '6mo'},
                            {'label': '3 Months', 'value': '3mo'},
                            {'label': '1 Month', 'value': '1mo'},
                            {'label': '1 Week', 'value': '1wk'}
                        ],
                        value='1y',
                        className="form-control mb-2"
                    ),
                    dcc.Input(id='input-keywords', value='', type='text', placeholder='Enter keywords for analysis', className="form-control mb-2"),
                    html.Button(id='submit-button', n_clicks=0, children='Submit', className="btn btn-primary"),
                    html.Button(id='gpt-button', n_clicks=0, children='Generate GPT Analysis', className="btn btn-secondary mt-2")
                ])
            ], className="mb-4")
        ], width=4),
    ]),
    dbc.Tabs([
        dbc.Tab(label="Key Indicators", tab_id="key-indicators"),
        dbc.Tab(label="Stock Performance", tab_id="stock-performance"),
        dbc.Tab(label="Company News", tab_id="company-news"),
        dbc.Tab([
            html.Span("GPT Analysis "),
            dbc.Badge("New", color="success", id="gpt-badge", className="ml-1")
        ], tab_id="gpt-analysis"),
    ], id="tabs", active_tab="key-indicators"),
    html.Div(id='tab-content'),
    dcc.Store(id='gpt-analysis-store', storage_type='memory'),
    dcc.Loading(
        id="loading-stock-info",
        type="circle",
        children=html.Div(id="loading-output-stock-info")
    ),
    dcc.Loading(
        id="loading-gpt-analysis",
        type="circle",
        children=html.Div(id="loading-output-gpt-analysis")
    ),
])

@app.callback(
    [Output('gpt-analysis-store', 'data'),
     Output('tabs', 'active_tab'),
     Output('loading-output-gpt-analysis', 'children')],
    [Input('gpt-button', 'n_clicks')],
    [State('input-tickers', 'value'),
     State('input-period', 'value'),
     State('input-keywords', 'value')]
)
def generate_gpt_analysis_callback(n_clicks, tickers, period, keywords):
    if n_clicks == 0:
        return dash.no_update, dash.no_update, ""
    
    if not tickers:
        tickers = DEFAULT_TICKERS
    
    valid_tickers = validate_tickers(tickers)
    if not valid_tickers:
        return {}, dash.no_update, "No valid tickers entered"

    stock_data = fetch_stock_data(valid_tickers, period)
    gpt_analyses = {}
    
    for ticker in valid_tickers:
        indicators_df = fetch_key_indicators(ticker)
        company_info = fetch_company_info(ticker)
        news = fetch_company_news(ticker)
        
        gpt_response = generate_gpt_analysis(ticker, indicators_df, company_info, keywords, stock_data[ticker], news)
        gpt_analyses[ticker] = gpt_response
    
    return gpt_analyses, 'gpt-analysis', ""

@app.callback(
    Output('tab-content', 'children'),
    [Input('tabs', 'active_tab'),
     Input('submit-button', 'n_clicks'),
     Input('gpt-analysis-store', 'data')],
    [State('input-tickers', 'value'),
     State('input-period', 'value')]
)
def render_tab_content(active_tab, submit_clicks, gpt_analyses, tickers, period):
    ctx = dash.callback_context
    if not ctx.triggered:
        tickers = DEFAULT_TICKERS
    elif not tickers:
        tickers = DEFAULT_TICKERS

    valid_tickers = validate_tickers(tickers)
    if not valid_tickers:
        return "No valid tickers entered"

    stock_data = fetch_stock_data(valid_tickers, period)
    key_figures = calculate_key_figures(stock_data)

    if active_tab == 'key-indicators':
        return render_key_indicators(valid_tickers, key_figures)
    elif active_tab == 'stock-performance':
        return render_stock_performance(stock_data)
    elif active_tab == 'company-news':
        return render_company_news(valid_tickers)
    elif active_tab == 'gpt-analysis':
        if not gpt_analyses:
            return html.Div("GPT Analysis not generated yet. Please click the 'Generate GPT Analysis' button.")
        return render_gpt_analysis(gpt_analyses)

    return "Tab content not found"

@app.callback(
    Output('gpt-badge', 'style'),
    [Input('gpt-analysis-store', 'data')]
)
def toggle_gpt_badge(gpt_analyses):
    if gpt_analyses:
        return {'display': 'inline-block'}
    return {'display': 'none'}

def render_key_indicators(valid_tickers, key_figures):
    key_figures_cards = []
    for ticker in valid_tickers:
        indicators_df = fetch_key_indicators(ticker)
        company_info = fetch_company_info(ticker)
        
        ratios = ['P/E', 'Forward P/E', 'Price/Book', 'Price/Sales']
        values = [indicators_df.loc[indicators_df['Indicator'] == ratio, 'Value'].values[0] for ratio in ratios]
        fig = go.Figure(data=[go.Pie(labels=ratios, values=values, hole=.3)])
        fig.update_layout(title_text=f"Financial Ratios for {ticker}")
        
        key_figures_cards.append(
            dbc.Card([
                dbc.CardHeader(f"Key Indicators for {company_info['name']} ({ticker})"),
                dbc.CardBody([
                    dash_table.DataTable(
                        data=[{
                            "Indicator": "Volatility",
                            "Value": f"{key_figures[ticker]['Volatility']:.4f}"
                        }, {
                            "Indicator": "Annualized Volatility",
                            "Value": f"{key_figures[ticker]['Annualized Volatility']:.4f}"
                        }] + indicators_df.to_dict('records'),
                        columns=[{"name": "Indicator", "id": "Indicator"},
                                 {"name": "Value", "id": "Value"}],
                        tooltip_data=[
                            {
                                'Indicator': {'value': get_tooltip(row['Indicator']), 'type': 'markdown'},
                                'Value': {'value': str(row['Value']), 'type': 'text'}
                            } for row in indicators_df.to_dict('records')
                        ],
                        tooltip_duration=None,
                        style_table={'margin': '20px'},
                        style_cell={'textAlign': 'left', 'padding': '5px'},
                        style_header={'backgroundColor': colors['light'], 'fontWeight': 'bold'}
                    ),
                    dcc.Graph(figure=fig)
                ])
            ], className="mb-4")
        )
    return html.Div(key_figures_cards)

def render_stock_performance(stock_data):
    stock_performance_graphs = []
    for ticker, data in stock_data.items():
        fig = px.line(data, x=data.index, y='Close', title=f'{ticker} Closing Prices')
        fig.update_layout(template="plotly_white")
        stock_performance_graphs.append(dcc.Graph(figure=fig, className="mb-4"))
    return html.Div(stock_performance_graphs)

def render_company_news(valid_tickers):
    news_cards = []
    for ticker in valid_tickers:
        news = fetch_company_news(ticker)
        news_items = [
            dbc.Card([
                dbc.CardBody([
                    html.H5(item['title'], className="card-title"),
                    html.P(f"Published: {datetime.fromtimestamp(item['providerPublishTime']).strftime('%Y-%m-%d %H:%M:%S')}"),
                    html.A("Read more", href=item['link'], target="_blank", className="btn btn-primary btn-sm")
                ])
            ], className="mb-3")
            for item in news
        ]
        news_cards.append(
            dbc.Card([
                dbc.CardHeader(f"Recent News for {ticker}"),
                dbc.CardBody(news_items)
            ], className="mb-4")
        )
    return html.Div(news_cards)

def render_gpt_analysis(gpt_analyses):
    gpt_cards = []
    for ticker, analysis in gpt_analyses.items():
        gpt_cards.append(
            dbc.Card([
                dbc.CardHeader(f"GPT Analysis for {ticker}"),
                dbc.CardBody([
                    dcc.Markdown(analysis)
                ])
            ], className="mb-4")
        )
    return html.Div(gpt_cards)

if __name__ == '__main__':
    app.run_server(debug=True)