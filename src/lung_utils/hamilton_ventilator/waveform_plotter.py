import sys

import dash
import pandas as pd
import plotly.graph_objs as go
from dash import Input, Output, dcc, html


# ==== Load waveform file ====
def load_waveform_txt(filepath):
    try:
        df = pd.read_csv(
            filepath, sep="\t", engine="python", encoding="latin1"
        )
    except UnicodeDecodeError:
        raise ValueError(
            f"Failed to decode {filepath}. Please check the file encoding."
        )

    df.columns = [col.strip() for col in df.columns]
    df = df.dropna(how="all")
    df["Date_Time"] = pd.to_numeric(df["Date_Time"], errors="coerce")
    df = df.dropna(subset=["Date_Time"])
    df["Time (s)"] = df["Date_Time"] - df["Date_Time"].iloc[0]
    return df


# ==== Create Dash App ====
def create_dash_app(df, file_path):
    app = dash.Dash(__name__)
    app.title = "Hamilton Waveform Viewer"

    # Select waveform columns
    exclude_cols = ["Date_Time", "Time (s)", "Breath Number", "Status"]
    waveform_columns = [col for col in df.columns if col not in exclude_cols]

    app.layout = html.Div(
        [
            html.H2("Hamilton Ventilator Waveform Viewer"),
            html.Div(
                f"Loaded file: {file_path}",
                id="file-info",
                style={"marginBottom": "10px"},
            ),
            html.Label("Select waveform:"),
            dcc.Dropdown(
                id="waveform-dropdown",
                options=[
                    {"label": col, "value": col} for col in waveform_columns
                ],
                value=waveform_columns[0] if waveform_columns else None,
            ),
            dcc.Graph(id="waveform-plot"),
        ]
    )

    @app.callback(
        Output("waveform-plot", "figure"), Input("waveform-dropdown", "value")
    )
    def update_graph(selected_waveform):
        if selected_waveform is None or df.empty:
            return go.Figure()

        y_data = pd.to_numeric(df[selected_waveform], errors="coerce")
        trace = go.Scatter(
            x=df["Time (s)"], y=y_data, mode="lines", name=selected_waveform
        )
        layout = go.Layout(
            xaxis={"title": "Time (s)"},
            yaxis={"title": selected_waveform},
            margin={"l": 50, "r": 10, "t": 40, "b": 50},
            hovermode="closest",
        )
        return {"data": [trace], "layout": layout}

    return app


if __name__ == "__main__":
    # ==== CLI Argument ====
    if len(sys.argv) != 2:
        print(
            "Usage: python "
            "src/lung_utils/hamilton_ventilator/waveform_plotter.py "
            "/path/to/hamilton_file.txt"
        )
        sys.exit(1)

    FILE_PATH = sys.argv[1]

    # Load the file
    df = load_waveform_txt(FILE_PATH)

    # Create and run the app
    app = create_dash_app(df, FILE_PATH)
    app.run(debug=True)
