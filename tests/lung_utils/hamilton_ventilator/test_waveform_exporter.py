from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from lung_utils.hamilton_ventilator.waveform_exporter import (
    extract_waveforms,
    format_fourc_linearinterpolation,
    get_waveform_fields,
    main,
    make_variable_name,
    save_waveforms_npy,
)


@pytest.fixture
def sample_dataframe():
    return pd.DataFrame(
        {
            "Date_Time": [44917.0, 44917.00001, 44917.00002],
            "Time (s)": [0.0, 1.0, 2.0],
            "Pressure Waveform": [10.0, 20.0, 30.0],
            "Flow": [0.0, 2.0, 0.0],
            "Status": [0, 0, 0],
        }
    )


def test_get_waveform_fields_excludes_metadata(sample_dataframe):
    assert get_waveform_fields(sample_dataframe) == [
        "Pressure Waveform",
        "Flow",
    ]


def test_make_variable_name_sanitizes_field_name():
    assert make_variable_name("Pressure Waveform") == "pressure_waveform"
    assert make_variable_name("3 Flow [l/min]") == "waveform_3_flow_l_min"


def test_extract_waveforms_uses_original_samples_by_default(
    sample_dataframe,
):
    waveforms = extract_waveforms(
        sample_dataframe,
        fields=["Pressure Waveform", "Flow"],
        start=0.5,
        end=2.0,
    )

    np.testing.assert_allclose(waveforms["time"], [0.5, 1.5])
    np.testing.assert_allclose(waveforms["Pressure Waveform"], [20.0, 30.0])
    np.testing.assert_allclose(waveforms["Flow"], [2.0, 0.0])


def test_extract_waveforms_preserves_original_sample_times(sample_dataframe):
    waveforms = extract_waveforms(
        sample_dataframe,
        fields=["Pressure Waveform"],
        start=0.5,
        end=2.0,
        preserve_time=True,
    )

    np.testing.assert_allclose(waveforms["time"], [1.0, 2.0])
    np.testing.assert_allclose(waveforms["Pressure Waveform"], [20.0, 30.0])


def test_extract_waveforms_resamples_multiple_fields(sample_dataframe):
    waveforms = extract_waveforms(
        sample_dataframe,
        fields=["Pressure Waveform", "Flow"],
        start=0.5,
        end=1.5,
        sampling_rate=2.0,
    )

    np.testing.assert_allclose(waveforms["time"], [0.0, 0.5, 1.0])
    np.testing.assert_allclose(
        waveforms["Pressure Waveform"], [15.0, 20.0, 25.0]
    )
    np.testing.assert_allclose(waveforms["Flow"], [1.0, 2.0, 1.0])


def test_extract_waveforms_rejects_invalid_interval(sample_dataframe):
    with pytest.raises(ValueError, match="End time"):
        extract_waveforms(
            sample_dataframe,
            fields=["Flow"],
            start=1.0,
            end=1.0,
            sampling_rate=10.0,
        )


def test_save_waveforms_npy_uses_separate_arrays_per_field(tmp_path):
    output = tmp_path / "waveforms.npy"
    save_waveforms_npy(
        {
            "time": np.array([0.0, 1.0]),
            "Flow": np.array([0.0, 2.0]),
            "Pressure": np.array([10.0, 20.0]),
        },
        str(output),
    )

    loaded = np.load(output, allow_pickle=True).item()

    assert set(loaded) == {"time", "Flow", "Pressure"}
    np.testing.assert_allclose(loaded["time"], [0.0, 1.0])
    np.testing.assert_allclose(loaded["Flow"], [0.0, 2.0])
    np.testing.assert_allclose(loaded["Pressure"], [10.0, 20.0])


def test_format_fourc_linearinterpolation():
    function_string = format_fourc_linearinterpolation(
        times=np.array([0.0, 0.5, 1.0]),
        values=np.array([10.0, 15.0, 20.0]),
        funct_number=7,
        variable_name="p",
    )

    assert function_string == "\n".join(
        [
            "FUNCT7:",
            "- SYMBOLIC_FUNCTION_OF_TIME: p",
            "- VARIABLE: 0",
            "  NAME: p",
            "  TYPE: linearinterpolation",
            "  NUMPOINTS: 3",
            "  TIMES:",
            "  - 0",
            "  - 0.5",
            "  - 1",
            "  VALUES:",
            "  - 10",
            "  - 15",
            "  - 20",
        ]
    )


def test_main_lists_fields(tmp_path, capsys):
    waveform_file = tmp_path / "waveform.txt"
    waveform_file.write_text(
        "Date_Time\tPressure\tFlow\tStatus\n"
        "44917.0\t10\t0\t0\n"
        "44917.00001\t20\t2\t0\n",
        encoding="latin1",
    )

    with patch(
        "sys.argv",
        [
            "hamilton-waveform-export",
            str(waveform_file),
            "--list-fields",
        ],
    ):
        main()

    assert capsys.readouterr().out == "Pressure\nFlow\n"


def test_main_writes_npy_for_multiple_fields(tmp_path):
    waveform_file = tmp_path / "waveform.txt"
    output_file = tmp_path / "waveforms.npy"
    waveform_file.write_text(
        "Date_Time\tPressure\tFlow\n"
        "44917.0\t10\t0\n"
        "44917.000011574074\t20\t2\n"
        "44917.000023148148\t30\t0\n",
        encoding="latin1",
    )

    with patch(
        "sys.argv",
        [
            "hamilton-waveform-export",
            str(waveform_file),
            "--fields",
            "Pressure",
            "Flow",
            "--start",
            "0",
            "--end",
            "2.00000046752",
            "--output",
            str(output_file),
        ],
    ):
        main()

    loaded = np.load(output_file, allow_pickle=True).item()
    assert set(loaded) == {"time", "Pressure", "Flow"}
    assert len(loaded["time"]) == 3


def test_main_writes_npy_for_all_fields(tmp_path):
    waveform_file = tmp_path / "waveform.txt"
    output_file = tmp_path / "waveforms.npy"
    waveform_file.write_text(
        "Date_Time\tPressure\tFlow\n"
        "44917.0\t10\t0\n"
        "44917.000011574074\t20\t2\n",
        encoding="latin1",
    )

    with patch(
        "sys.argv",
        [
            "hamilton-waveform-export",
            str(waveform_file),
            "--all-fields",
            "--start",
            "0",
            "--end",
            "1.00000023376",
            "--output",
            str(output_file),
        ],
    ):
        main()

    loaded = np.load(output_file, allow_pickle=True).item()
    assert set(loaded) == {"time", "Pressure", "Flow"}


def test_main_prompts_for_multiple_fields_and_writes_npy(tmp_path):
    waveform_file = tmp_path / "waveform.txt"
    output_file = tmp_path / "waveforms.npy"
    waveform_file.write_text(
        "Date_Time\tPressure\tFlow\n"
        "44917.0\t10\t0\n"
        "44917.000011574074\t20\t2\n",
        encoding="latin1",
    )

    with (
        patch(
            "sys.argv",
            [
                "hamilton-waveform-export",
                str(waveform_file),
                "--output",
                str(output_file),
            ],
        ),
        patch("builtins.input", side_effect=["1,2", "0", ""]),
    ):
        main()

    loaded = np.load(output_file, allow_pickle=True).item()
    assert set(loaded) == {"time", "Pressure", "Flow"}


def test_main_copies_fourc_function_to_clipboard(tmp_path, capsys):
    waveform_file = tmp_path / "waveform.txt"
    waveform_file.write_text(
        "Date_Time\tPressure\tFlow\n"
        "44917.0\t10\t0\n"
        "44917.000011574074\t20\t2\n"
        "44917.000023148148\t30\t0\n",
        encoding="latin1",
    )

    with (
        patch(
            "sys.argv",
            [
                "hamilton-waveform-export",
                str(waveform_file),
                "--format",
                "fourc",
                "--fields",
                "Pressure",
                "--start",
                "0",
                "--end",
                "2.00000046752",
                "--funct",
                "2",
                "--variable-name",
                "p",
            ],
        ),
        patch(
            "lung_utils.hamilton_ventilator."
            "waveform_exporter.copy_to_clipboard"
        ) as copy,
    ):
        main()

    copied = copy.call_args.args[0]
    assert "FUNCT2:" in copied
    assert "- SYMBOLIC_FUNCTION_OF_TIME: p" in copied
    assert "  NUMPOINTS: 3" in copied
    assert "Copied 4C linearinterpolation function" in capsys.readouterr().out


def test_main_rejects_multiple_fields_for_fourc(tmp_path):
    waveform_file = tmp_path / "waveform.txt"
    waveform_file.write_text(
        "Date_Time\tPressure\tFlow\n"
        "44917.0\t10\t0\n"
        "44917.000011574074\t20\t2\n",
        encoding="latin1",
    )

    with (
        patch(
            "sys.argv",
            [
                "hamilton-waveform-export",
                str(waveform_file),
                "--format",
                "fourc",
                "--fields",
                "Pressure",
                "Flow",
                "--start",
                "0",
                "--end",
                "1.1",
            ],
        ),
        pytest.raises(SystemExit),
    ):
        main()
