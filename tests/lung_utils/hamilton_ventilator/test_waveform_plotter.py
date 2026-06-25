from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from dash import Dash
from lung_utils.hamilton_ventilator.waveform_plotter import (
    create_dash_app,
    find_parameter_file,
    load_parameter_txt,
)


@pytest.fixture
def sample_dataframe():
    """Fixture for a sample dataframe."""
    data = {
        "Date_Time": [44917.617693, 44917.617705, 44917.617716, 44917.617728],
        "Time (s)": [0, 1, 2, 3],
        "Waveform1": [10, 20, 30, 40],
        "Waveform2": [5, 15, 25, 35],
        "Absolute_Time": pd.to_datetime(
            [44917.617693, 44917.617705, 44917.617716, 44917.617728],
            unit="D",
            origin="1899-12-30",
        ),
    }
    return pd.DataFrame(data)


@pytest.fixture
def sample_parameter_dataframe(sample_dataframe):
    """Fixture for a sample Hamilton parameter dataframe."""
    return pd.DataFrame(
        {
            "Date_Time": sample_dataframe["Date_Time"],
            "Time (s)": sample_dataframe["Time (s)"],
            "Breath Number": [1, 2, 3, 4],
            "Mode Name": ["(S)CMV", "(S)CMV", "PCV+", "PCV+"],
            "PEEP/ CPAP /cmH2O": [8, 8, 15, 15],
            "Tidal Volume /ml": [350, 350, 350, 350],
            "Absolute_Time": sample_dataframe["Absolute_Time"],
        }
    )


def test_create_dash_app(sample_dataframe):
    """Test the create_dash_app function."""
    file_path = "test_file.txt"

    # Create the Dash app
    app = create_dash_app(sample_dataframe, file_path)

    # Check if the app is an instance of Dash
    assert isinstance(app, Dash)

    # Check if the app title is set correctly
    assert app.title == "Hamilton Waveform Viewer"

    # Check if the layout contains the expected components
    assert (
        "Hamilton Ventilator Waveform Viewer"
        in app.layout.children[0].children
    )
    assert f"Loaded file: {file_path}" in app.layout.children[1].children
    assert app.layout.children[2].id == "start-time-info"
    assert app.layout.children[4].id == "waveform-dropdown"
    assert app.layout.children[5].id == "waveform-plot"


@patch("src.lung_utils.hamilton_ventilator.waveform_plotter.dcc.Dropdown")
@patch("src.lung_utils.hamilton_ventilator.waveform_plotter.dcc.Graph")
def test_create_dash_app_layout(mock_graph, mock_dropdown, sample_dataframe):
    """Test if create_dash_app sets up the layout correctly."""
    file_path = "test_file.txt"

    # Mock the Dropdown and Graph components
    mock_dropdown.return_value = MagicMock()
    mock_graph.return_value = MagicMock()

    # Create the Dash app
    app = create_dash_app(sample_dataframe, file_path)

    # Check if the layout contains the expected components
    assert (
        app.layout.children[0].children
        == "Hamilton Ventilator Waveform Viewer"
    )
    assert app.layout.children[1].children == f"Loaded file: {file_path}"
    assert app.layout.children[2].id == "start-time-info"
    assert "Recording Start Time:" in app.layout.children[2].children
    assert app.layout.children[3].children == "Select up to 3 waveforms:"
    mock_dropdown.assert_called_once_with(
        id="waveform-dropdown",
        options=[
            {"label": "Waveform1", "value": "Waveform1"},
            {"label": "Waveform2", "value": "Waveform2"},
        ],
        value=["Waveform1"],
        multi=True,
    )
    mock_graph.assert_called_once_with(id="waveform-plot")


def test_create_dash_app_with_parameter_data(
    sample_dataframe, sample_parameter_dataframe
):
    """Test layout includes parameter controls when P data is supplied."""
    app = create_dash_app(
        sample_dataframe,
        "waveform.txt",
        parameter_df=sample_parameter_dataframe,
        parameter_file_path="parameters.txt",
    )

    assert isinstance(app, Dash)
    child_ids = [getattr(child, "id", None) for child in app.layout.children]
    assert "parameter-file-info" in child_ids
    assert "parameter-dropdown" in child_ids
    assert child_ids[-1] == "waveform-plot"


def test_find_parameter_file_finds_single_match(tmp_path):
    """Test parameter discovery finds one matching file."""
    waveform_file = tmp_path / "W_Hamilton-C6__example_Waves_001.txt"
    parameter_file = tmp_path / "P_Hamilton-C6__example_All_001.txt"
    waveform_file.write_text("", encoding="utf-8")
    parameter_file.write_text("", encoding="utf-8")

    assert find_parameter_file(waveform_file) == str(parameter_file)


def test_find_parameter_file_raises_for_missing_match(tmp_path):
    """Test parameter discovery raises when no P file exists."""
    waveform_file = tmp_path / "W_Hamilton-C6__example_Waves_001.txt"
    waveform_file.write_text("", encoding="utf-8")

    with pytest.raises(FileNotFoundError):
        find_parameter_file(waveform_file)


def test_find_parameter_file_raises_for_multiple_matches(tmp_path):
    """Test parameter discovery raises when more than one P file exists."""
    waveform_file = tmp_path / "W_Hamilton-C6__example_Waves_001.txt"
    waveform_file.write_text("", encoding="utf-8")
    (tmp_path / "P_Hamilton-C6__example_All_001.txt").write_text(
        "", encoding="utf-8"
    )
    (tmp_path / "P_Hamilton-C6__example_All_002.txt").write_text(
        "", encoding="utf-8"
    )

    with pytest.raises(ValueError, match="Multiple Hamilton parameter files"):
        find_parameter_file(waveform_file)


def test_load_parameter_txt_uses_waveform_start_time(tmp_path):
    """Test P-file times can be aligned to waveform start."""
    parameter_file = tmp_path / "P_Hamilton-C6__example_All_001.txt"
    parameter_file.write_text(
        "Date_Time\tBreath Number\tMode Name\n"
        "44917.000010\t1\t(S)CMV\n"
        "44917.000020\t2\t(S)CMV\n",
        encoding="latin1",
    )

    df = load_parameter_txt(parameter_file, start_time_val=44917.000000)

    assert list(df["Mode Name"]) == ["(S)CMV", "(S)CMV"]
    assert df["Time (s)"].iloc[0] == pytest.approx(0.864)
    assert df["Time (s)"].iloc[1] == pytest.approx(1.728)
