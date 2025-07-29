"""Test run procedure."""

from unittest.mock import MagicMock, patch

from munch import munchify
from lung_utils.core.run import run_lung_utils


def test_run_lung_utils() -> None:
    """Test run procedure of LungUtils."""

    mock_config = munchify({"key": "value"})

    mock_run_manager = MagicMock()

    with patch("lung_utils.core.run.RunManager", return_value=mock_run_manager):
        mock_exemplary_function = MagicMock(return_value="Exemplary output")
        with patch(
            "lung_utils.core.run.exemplary_function", mock_exemplary_function
        ):
            run_lung_utils(mock_config)

    mock_run_manager.init_run.assert_called_once()
    mock_exemplary_function.assert_called_once()
    mock_run_manager.finish_run.assert_called_once()
