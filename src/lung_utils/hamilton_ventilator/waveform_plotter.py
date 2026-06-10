import argparse
from pathlib import Path

import dash
import pandas as pd
import plotly.graph_objs as go
from dash import Input, Output, dcc, html
from plotly.subplots import make_subplots

METADATA_COLUMNS = {
    "Date_Time",
    "Time (s)",
    "Absolute_Time",
    "Breath Number",
    "Status",
}

PARAMETER_DEFAULT_FIELDS = [
    "Mode Name",
    "PEEP/ CPAP /cmH2O",
    "Tidal Volume /ml",
]


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

    # Calculate relative time in seconds from start
    start_time_val = df["Date_Time"].iloc[0]
    df["Time (s)"] = (df["Date_Time"] - start_time_val) * 24 * 3600

    # Convert OLE Automation Date (Excel date) to absolute datetime
    df["Absolute_Time"] = pd.to_datetime(
        df["Date_Time"], unit="D", origin="1899-12-30"
    )

    return df


def find_parameter_file(waveform_filepath):
    """Find a single Hamilton parameter file next to a waveform file."""
    waveform_path = Path(waveform_filepath)
    parameter_files = sorted(waveform_path.parent.glob("P_Hamilton-C6*"))
    if not parameter_files:
        parameter_files = sorted(waveform_path.parent.glob("P_Hamilton*"))

    if not parameter_files:
        raise FileNotFoundError(
            "No Hamilton parameter file matching P_Hamilton-C6* was found in "
            f"{waveform_path.parent}."
        )
    if len(parameter_files) > 1:
        matches = "\n".join(str(path) for path in parameter_files)
        raise ValueError(
            "Multiple Hamilton parameter files were found. Please pass one "
            f"with --parameter-file:\n{matches}"
        )
    return str(parameter_files[0])


def _load_hamilton_txt(filepath, start_time_val=None):
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

    if start_time_val is None:
        start_time_val = df["Date_Time"].iloc[0]
    df["Time (s)"] = (df["Date_Time"] - start_time_val) * 24 * 3600
    df["Absolute_Time"] = pd.to_datetime(
        df["Date_Time"], unit="D", origin="1899-12-30"
    )

    return df


def load_parameter_txt(filepath, start_time_val=None):
    """Load a Hamilton parameter file and align time to waveform start."""
    return _load_hamilton_txt(filepath, start_time_val=start_time_val)


def get_plot_columns(df):
    """Return non-metadata columns suitable for plotting."""
    return [col for col in df.columns if col not in METADATA_COLUMNS]


def get_default_parameter_fields(parameter_columns):
    defaults = [
        field
        for field in PARAMETER_DEFAULT_FIELDS
        if field in parameter_columns
    ]
    return defaults or parameter_columns[:1]


def add_trace_to_subplot(fig, df, field, row, *, is_parameter=False):
    y_data = pd.to_numeric(df[field], errors="coerce")
    original_values = None

    if is_parameter and y_data.notna().sum() == 0:
        category_values = df[field].fillna("--").astype(str)
        codes, categories = pd.factorize(category_values, sort=True)
        y_data = pd.Series(codes, index=df.index)
        original_values = category_values
        fig.update_yaxes(
            tickmode="array",
            tickvals=list(range(len(categories))),
            ticktext=list(categories),
            row=row,
            col=1,
        )

    abs_time_str = df["Absolute_Time"].dt.strftime("%H:%M:%S.%f").str[:-3]
    customdata = (
        original_values if original_values is not None else abs_time_str
    )
    hovertemplate = (
        "<b>%{y}</b><br>"
        "Time: %{x:.2f} s<br>"
        "Abs Time: %{customdata}"
        "<extra></extra>"
    )
    if original_values is not None:
        hovertemplate = (
            "<b>%{customdata}</b><br>" "Time: %{x:.2f} s" "<extra></extra>"
        )

    trace = go.Scatter(
        x=df["Time (s)"],
        y=y_data,
        mode="lines+markers" if is_parameter else "lines",
        name=field,
        customdata=customdata,
        hovertemplate=hovertemplate,
    )
    fig.add_trace(trace, row=row, col=1)
    fig.update_yaxes(title_text=field, row=row, col=1)


# ==== Create Dash App ====
def create_dash_app(
    df, file_path, parameter_df=None, parameter_file_path=None
):
    app = dash.Dash(__name__)
    app.title = "Hamilton Waveform Viewer"

    # Select waveform columns
    waveform_columns = get_plot_columns(df)
    parameter_columns = []
    parameter_default_fields = []
    if parameter_df is not None:
        parameter_columns = get_plot_columns(parameter_df)
        parameter_default_fields = get_default_parameter_fields(
            parameter_columns
        )

    # Get absolute start time
    start_time_str = ""
    if not df.empty and "Absolute_Time" in df.columns:
        start_time_str = (
            df["Absolute_Time"].iloc[0].strftime("%Y-%m-%d %H:%M:%S")
        )

    layout_children = [
        html.H2("Hamilton Ventilator Waveform Viewer"),
        html.Div(
            f"Loaded file: {file_path}",
            id="file-info",
            style={"marginBottom": "5px"},
        ),
        html.Div(
            f"Recording Start Time: {start_time_str}",
            id="start-time-info",
            style={"marginBottom": "10px", "fontWeight": "bold"},
        ),
        html.Label("Select up to 3 waveforms:"),
        dcc.Dropdown(
            id="waveform-dropdown",
            options=[{"label": col, "value": col} for col in waveform_columns],
            value=[waveform_columns[0]] if waveform_columns else [],
            multi=True,
        ),
    ]

    callback_inputs = [Input("waveform-dropdown", "value")]
    if parameter_df is not None:
        layout_children.extend(
            [
                html.Div(
                    f"Loaded parameter file: {parameter_file_path}",
                    id="parameter-file-info",
                    style={"marginTop": "10px", "marginBottom": "5px"},
                ),
                html.Label("Select up to 3 parameter fields:"),
                dcc.Dropdown(
                    id="parameter-dropdown",
                    options=[
                        {"label": col, "value": col}
                        for col in parameter_columns
                    ],
                    value=parameter_default_fields,
                    multi=True,
                ),
            ]
        )
        callback_inputs.append(Input("parameter-dropdown", "value"))

    layout_children.append(dcc.Graph(id="waveform-plot"))
    app.layout = html.Div(layout_children)

    @app.callback(Output("waveform-plot", "figure"), *callback_inputs)
    def update_graph(selected_waveforms, selected_parameters=None):
        if df.empty:
            return go.Figure()

        # Enforce maximum of 3 plots
        if isinstance(selected_waveforms, str):
            selected_waveforms = [selected_waveforms]
        if isinstance(selected_parameters, str):
            selected_parameters = [selected_parameters]

        selected_waveforms = (selected_waveforms or [])[:3]
        selected_parameters = (selected_parameters or [])[:3]
        num_waveform_plots = len(selected_waveforms)
        num_parameter_plots = len(selected_parameters)
        num_plots = num_waveform_plots + num_parameter_plots
        if num_plots == 0:
            return go.Figure()

        fig = make_subplots(
            rows=num_plots,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=selected_waveforms + selected_parameters,
        )

        for i, waveform in enumerate(selected_waveforms, start=1):
            add_trace_to_subplot(fig, df, waveform, i)

        for i, parameter in enumerate(
            selected_parameters, start=num_waveform_plots + 1
        ):
            add_trace_to_subplot(
                fig, parameter_df, parameter, i, is_parameter=True
            )

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="View Hamilton ventilator waveform TXT files."
    )
    parser.add_argument("waveform_file", help="Path to Hamilton waveform.txt")
    parser.add_argument(
        "--include-parameters",
        action="store_true",
        help="Search for a matching P_Hamilton-C6* file and plot fields.",
    )
    parser.add_argument(
        "--parameter-file",
        help="Explicit Hamilton P_Hamilton-C6* parameter file to include.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Load the file
    df = load_waveform_txt(args.waveform_file)

    parameter_df = None
    parameter_file_path = args.parameter_file
    if args.include_parameters or args.parameter_file:
        if parameter_file_path is None:
            parameter_file_path = find_parameter_file(args.waveform_file)
        parameter_df = load_parameter_txt(
            parameter_file_path, start_time_val=df["Date_Time"].iloc[0]
        )

    # Create and run the app
    app = create_dash_app(
        df,
        args.waveform_file,
        parameter_df=parameter_df,
        parameter_file_path=parameter_file_path,
    )
    app.run(debug=True)


if __name__ == "__main__":
    main()
