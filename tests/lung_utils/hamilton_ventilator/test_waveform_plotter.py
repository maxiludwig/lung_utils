from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from dash import Dash
from lung_utils.hamilton_ventilator.waveform_plotter import create_dash_app


@pytest.fixture
def sample_dataframe():
    """Fixture for a sample dataframe."""
    data = {
        "Date_Time": [0, 1, 2, 3],
        "Time (s)": [0, 1, 2, 3],
        "Waveform1": [10, 20, 30, 40],
        "Waveform2": [5, 15, 25, 35],
    }
    return pd.DataFrame(data)


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
    assert app.layout.children[3].id == "waveform-dropdown"
    assert app.layout.children[4].id == "waveform-plot"


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
    assert app.layout.children[2].children == "Select waveform:"
    mock_dropdown.assert_called_once_with(
        id="waveform-dropdown",
        options=[
            {"label": "Waveform1", "value": "Waveform1"},
            {"label": "Waveform2", "value": "Waveform2"},
        ],
        value="Waveform1",
    )
    mock_graph.assert_called_once_with(id="waveform-plot")
