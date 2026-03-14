from dotenv import load_dotenv
load_dotenv()

import dash
import dash_bootstrap_components as dbc

import data
from layout import create_layout
from callbacks import register_callbacks

app = dash.Dash(
    __name__,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,
        "https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&family=Montserrat:wght@500;700&display=swap",
    ],
)

data.cache.init_app(app.server, config={"CACHE_TYPE": "SimpleCache", "CACHE_DEFAULT_TIMEOUT": 300})

app.layout = create_layout()
register_callbacks(app)

if __name__ == "__main__":
    app.run_server(debug=True)
