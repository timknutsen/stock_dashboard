from dash import dcc, html
import dash_bootstrap_components as dbc

from data import DEFAULT_TICKERS


def create_layout():
    return dbc.Container([
        html.H1("Stock Performance Analysis", className="text-center mb-4"),

        dbc.Row([
            dbc.Col([
                dbc.Card([
                    dbc.CardBody([
                        dcc.Input(
                            id="input-tickers",
                            value=DEFAULT_TICKERS,
                            type="text",
                            placeholder="Enter stock tickers (e.g. AAPL MSFT BRK.B)",
                            className="form-control mb-2",
                        ),
                        dcc.Dropdown(
                            id="input-period",
                            options=[
                                {"label": "1 Year",   "value": "1y"},
                                {"label": "6 Months", "value": "6mo"},
                                {"label": "3 Months", "value": "3mo"},
                                {"label": "1 Month",  "value": "1mo"},
                                {"label": "1 Week",   "value": "1wk"},
                            ],
                            value="1y",
                            clearable=False,
                            className="mb-2",
                        ),
                        dcc.Input(
                            id="input-keywords",
                            value="",
                            type="text",
                            placeholder="Keywords for analysis (optional)",
                            className="form-control mb-2",
                        ),
                        html.Div([
                            html.Button(
                                "Submit",
                                id="submit-button",
                                n_clicks=0,
                                className="btn btn-primary me-2",
                            ),
                            html.Button(
                                "Generate Claude Analysis",
                                id="claude-button",
                                n_clicks=0,
                                className="btn btn-secondary",
                            ),
                        ]),
                    ]),
                ], className="mb-4"),
            ], width=4),
        ]),

        dbc.Tabs([
            dbc.Tab(label="Key Indicators",    tab_id="key-indicators"),
            dbc.Tab(label="Stock Performance", tab_id="stock-performance"),
            dbc.Tab(label="Company News",      tab_id="company-news"),
            dbc.Tab(
                [html.Span("Claude Analysis "), dbc.Badge("New", color="success", id="claude-badge")],
                tab_id="claude-analysis",
            ),
        ], id="tabs", active_tab="key-indicators"),

        html.Div(id="tab-content"),

        dcc.Store(id="claude-analysis-store", storage_type="memory"),

        dcc.Loading(
            id="loading-claude",
            type="circle",
            children=html.Div(id="loading-output-claude"),
        ),
    ], fluid=True)
