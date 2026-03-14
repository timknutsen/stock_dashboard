from datetime import datetime

import dash
import dash_bootstrap_components as dbc
import plotly.express as px
import plotly.graph_objs as go
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State

from data import (
    DEFAULT_TICKERS,
    TOOLTIPS,
    calculate_key_figures,
    fetch_company_info,
    fetch_company_news,
    fetch_key_indicators,
    fetch_stock_data,
    generate_claude_analysis,
    validate_tickers,
)

_TABLE_HEADER_STYLE = {"backgroundColor": "#f8f9fa", "fontWeight": "bold"}
_TABLE_CELL_STYLE = {"textAlign": "left", "padding": "5px"}


def register_callbacks(app):

    @app.callback(
        [
            Output("claude-analysis-store", "data"),
            Output("tabs", "active_tab"),
            Output("loading-output-claude", "children"),
        ],
        Input("claude-button", "n_clicks"),
        [
            State("input-tickers", "value"),
            State("input-period", "value"),
            State("input-keywords", "value"),
        ],
        prevent_initial_call=True,
    )
    def run_claude_analysis(n_clicks, tickers, period, keywords):
        valid = validate_tickers(tickers or DEFAULT_TICKERS)
        if not valid:
            return {}, dash.no_update, "No valid tickers entered."

        stock_data = fetch_stock_data(valid, period)
        analyses = {}
        for ticker in valid:
            if ticker not in stock_data:
                continue
            analyses[ticker] = generate_claude_analysis(
                ticker,
                fetch_key_indicators(ticker),
                fetch_company_info(ticker),
                keywords or "",
                stock_data[ticker],
                fetch_company_news(ticker),
            )
        return analyses, "claude-analysis", ""

    @app.callback(
        Output("tab-content", "children"),
        [
            Input("tabs", "active_tab"),
            Input("submit-button", "n_clicks"),
            Input("claude-analysis-store", "data"),
        ],
        [
            State("input-tickers", "value"),
            State("input-period", "value"),
        ],
    )
    def render_tab_content(active_tab, _submit, claude_analyses, tickers, period):
        valid = validate_tickers(tickers or DEFAULT_TICKERS)
        if not valid:
            return dbc.Alert("No valid tickers entered.", color="warning")

        if active_tab == "key-indicators":
            stock_data = fetch_stock_data(valid, period)
            return _render_key_indicators(valid, calculate_key_figures(stock_data))

        if active_tab == "stock-performance":
            return _render_stock_performance(fetch_stock_data(valid, period))

        if active_tab == "company-news":
            return _render_company_news(valid)

        if active_tab == "claude-analysis":
            if not claude_analyses:
                return html.P("Claude Analysis not generated yet. Click 'Generate Claude Analysis'.")
            return _render_claude_analysis(claude_analyses)

        return "Tab not found."

    @app.callback(
        Output("claude-badge", "style"),
        Input("claude-analysis-store", "data"),
    )
    def toggle_badge(analyses):
        return {"display": "inline-block"} if analyses else {"display": "none"}


# ---------------------------------------------------------------------------
# Render helpers
# ---------------------------------------------------------------------------

def _render_key_indicators(valid_tickers, key_figures):
    cards = []
    for ticker in valid_tickers:
        indicators_df = fetch_key_indicators(ticker)
        company_info = fetch_company_info(ticker)

        # Pie chart — skip None values so the chart is never broken
        ratio_names = ["P/E", "Forward P/E", "Price/Book", "Price/Sales"]
        ratio_map = {
            r: indicators_df.loc[indicators_df["Indicator"] == r, "Value"].values[0]
            for r in ratio_names
        }
        valid_ratios = {k: v for k, v in ratio_map.items() if v is not None}
        fig = go.Figure(data=[go.Pie(
            labels=list(valid_ratios.keys()),
            values=list(valid_ratios.values()),
            hole=0.3,
        )])
        fig.update_layout(title_text=f"Financial Ratios for {ticker}", template="plotly_white")

        # Table rows — volatility first, then indicators with N/A for missing values
        vol_rows = [
            {"Indicator": "Volatility",           "Value": f"{key_figures[ticker]['Volatility']:.4f}"},
            {"Indicator": "Annualized Volatility", "Value": f"{key_figures[ticker]['Annualized Volatility']:.4f}"},
        ]
        indicator_rows = [
            {"Indicator": row["Indicator"], "Value": str(row["Value"]) if row["Value"] is not None else "N/A"}
            for row in indicators_df.to_dict("records")
        ]
        table_rows = vol_rows + indicator_rows

        cards.append(dbc.Card([
            dbc.CardHeader(f"Key Indicators for {company_info['name']} ({ticker})"),
            dbc.CardBody([
                dash_table.DataTable(
                    data=table_rows,
                    columns=[
                        {"name": "Indicator", "id": "Indicator"},
                        {"name": "Value",     "id": "Value"},
                    ],
                    tooltip_data=[
                        {"Indicator": {"value": TOOLTIPS.get(row["Indicator"], ""), "type": "markdown"}}
                        for row in table_rows
                    ],
                    tooltip_duration=None,
                    style_table={"margin": "20px"},
                    style_cell=_TABLE_CELL_STYLE,
                    style_header=_TABLE_HEADER_STYLE,
                ),
                dcc.Graph(figure=fig),
            ]),
        ], className="mb-4"))

    return html.Div(cards)


def _render_stock_performance(stock_data):
    graphs = []
    for ticker, data in stock_data.items():
        fig = px.line(data, x=data.index, y="Close", title=f"{ticker} Closing Prices")
        fig.update_layout(template="plotly_white")
        graphs.append(dcc.Graph(figure=fig, className="mb-4"))
    return html.Div(graphs)


def _render_company_news(valid_tickers):
    cards = []
    for ticker in valid_tickers:
        news = fetch_company_news(ticker)
        if not news:
            items = [html.P("No recent news found.", className="text-muted")]
        else:
            items = [
                dbc.Card([
                    dbc.CardBody([
                        html.H5(item["title"], className="card-title"),
                        html.P(
                            f"Published: {datetime.fromtimestamp(item['providerPublishTime']).strftime('%Y-%m-%d %H:%M')}",
                            className="text-muted small",
                        ),
                        html.A("Read more", href=item["link"], target="_blank", className="btn btn-primary btn-sm"),
                    ]),
                ], className="mb-3")
                for item in news
            ]
        cards.append(dbc.Card([
            dbc.CardHeader(f"Recent News for {ticker}"),
            dbc.CardBody(items),
        ], className="mb-4"))
    return html.Div(cards)


def _render_claude_analysis(analyses):
    return html.Div([
        dbc.Card([
            dbc.CardHeader(f"Claude Analysis for {ticker}"),
            dbc.CardBody(dcc.Markdown(analysis)),
        ], className="mb-4")
        for ticker, analysis in analyses.items()
    ])
