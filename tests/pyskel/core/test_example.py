"""Test example."""

from lung_utils.core.example import exemplary_function


def test_exemplary_function() -> None:
    """Test exemplary multiply function."""

    assert exemplary_function(2, 3) == 6
