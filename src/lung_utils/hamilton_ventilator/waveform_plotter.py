import sys

import dash
import pandas as pd
import plotly.graph_objs as go
from dash import Input, Output, dcc, html
from plotly.subplots import make_subplots


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
            html.Label("Select up to 3 waveforms:"),
            dcc.Dropdown(
                id="waveform-dropdown",
                options=[
                    {"label": col, "value": col} for col in waveform_columns
                ],
                value=[waveform_columns[0]] if waveform_columns else [],
                multi=True,
            ),
            dcc.Graph(id="waveform-plot"),
        ]
    )

    @app.callback(
        Output("waveform-plot", "figure"), Input("waveform-dropdown", "value")
    )
    def update_graph(selected_waveforms):
        if not selected_waveforms or df.empty:
            return go.Figure()

        # Enforce maximum of 3 plots
        if isinstance(selected_waveforms, str):
            selected_waveforms = [selected_waveforms]

        selected_waveforms = selected_waveforms[:3]
        num_plots = len(selected_waveforms)

        fig = make_subplots(
            rows=num_plots,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=selected_waveforms,
        )

        for i, waveform in enumerate(selected_waveforms, start=1):
            y_data = pd.to_numeric(df[waveform], errors="coerce")
            trace = go.Scatter(
                x=df["Time (s)"], y=y_data, mode="lines", name=waveform
            )
            fig.add_trace(trace, row=i, col=1)
            fig.update_yaxes(title_text=waveform, row=i, col=1)

        # Calculate dynamic height (approx 300px per plot)
        plot_height = max(400, 300 * num_plots)

        fig.update_layout(
            height=plot_height,
            xaxis={"title": "Time (s)"},
            margin={"l": 50, "r": 10, "t": 40, "b": 50},
            hovermode="x unified",
        )

        # Ensure the bottom-most x-axis has a title
        fig.update_xaxes(title_text="Time (s)", row=num_plots, col=1)

        return fig

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
