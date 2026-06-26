"""Export Hamilton waveform data to downstream formats."""

import argparse
import re

import numpy as np
import pandas as pd
from lung_utils.hamilton_ventilator.waveform_plotter import load_waveform_txt

EXCLUDED_WAVEFORM_COLUMNS = {
    "Date_Time",
    "Time (s)",
    "Absolute_Time",
    "Breath Number",
    "Status",
}


def get_waveform_fields(df: pd.DataFrame) -> list[str]:
    """Return data columns that can be exported as waveforms."""
    return [
        column
        for column in df.columns
        if column not in EXCLUDED_WAVEFORM_COLUMNS
    ]


def make_variable_name(field_name: str) -> str:
    """Create a valid simple variable name from a waveform field name."""
    variable_name = re.sub(r"\W+", "_", field_name.strip().lower()).strip("_")
    if not variable_name:
        return "waveform"
    if variable_name[0].isdigit():
        return f"waveform_{variable_name}"
    return variable_name


def extract_waveforms(
    df: pd.DataFrame,
    fields: list[str],
    start: float,
    end: float,
    sampling_rate: float | None = None,
    preserve_time: bool = False,
) -> dict[str, np.ndarray]:
    """Extract one or more waveform fields into a common time basis."""
    if not fields:
        raise ValueError("At least one waveform field must be selected.")
    missing_fields = [field for field in fields if field not in df.columns]
    if missing_fields:
        raise ValueError(
            "Waveform field(s) not found: " + ", ".join(missing_fields)
        )
    if start < 0:
        raise ValueError("Start time must be non-negative.")
    if end <= start:
        raise ValueError("End time must be larger than start time.")
    if sampling_rate is not None and sampling_rate <= 0:
        raise ValueError("Sampling rate must be positive.")

    time = pd.to_numeric(df["Time (s)"], errors="coerce")
    values_by_field = {
        field: pd.to_numeric(df[field], errors="coerce") for field in fields
    }
    valid = time.notna()
    for values in values_by_field.values():
        valid &= values.notna()

    time_values = time[valid].to_numpy(dtype=float)
    if len(time_values) < 2:
        raise ValueError("At least two valid waveform samples are required.")

    waveform_values_by_field = {
        field: values[valid].to_numpy(dtype=float)
        for field, values in values_by_field.items()
    }

    tolerance = 1.0e-6
    if abs(start - time_values[0]) <= tolerance:
        start = time_values[0]
    if abs(end - time_values[-1]) <= tolerance:
        end = time_values[-1]

    if start < time_values[0] or end > time_values[-1]:
        raise ValueError(
            "Requested interval must be within the available waveform time "
            f"range [{time_values[0]:.12g}, {time_values[-1]:.12g}]."
        )

    if sampling_rate is None:
        in_interval = (time_values >= start) & (time_values <= end)
        sample_times = time_values[in_interval]
        if len(sample_times) == 0:
            raise ValueError(
                "No original waveform samples found in the requested interval."
            )
        output = {
            field: values[in_interval]
            for field, values in waveform_values_by_field.items()
        }
    else:
        step = 1.0 / sampling_rate
        sample_times = np.arange(start, end + step * 0.5, step)
        sample_times = sample_times[sample_times <= end]
        if sample_times[-1] < end:
            sample_times = np.append(sample_times, end)
        output = {
            field: np.interp(sample_times, time_values, values)
            for field, values in waveform_values_by_field.items()
        }

    output_time = sample_times if preserve_time else sample_times - start
    return {"time": output_time, **output}


def save_waveforms_npy(waveforms: dict[str, np.ndarray], output: str) -> None:
    """Save extracted waveform arrays to a numpy file."""
    np.save(output, waveforms, allow_pickle=True)


def format_fourc_linearinterpolation(
    times: np.ndarray,
    values: np.ndarray,
    funct_number: int,
    variable_name: str,
) -> str:
    """Format sampled points as a 4C linearinterpolation function block."""
    if len(times) != len(values):
        raise ValueError("Times and values must have the same length.")
    if len(times) == 0:
        raise ValueError("At least one interpolation point is required.")
    if funct_number <= 0:
        raise ValueError("Function number must be positive.")

    lines = [
        f"FUNCT{funct_number}:",
        f"- SYMBOLIC_FUNCTION_OF_TIME: {variable_name}",
        "- VARIABLE: 0",
        f"  NAME: {variable_name}",
        "  TYPE: linearinterpolation",
        f"  NUMPOINTS: {len(times)}",
        "  TIMES:",
    ]
    lines.extend(f"  - {_format_number(time)}" for time in times)
    lines.append("  VALUES:")
    lines.extend(f"  - {_format_number(value)}" for value in values)
    return "\n".join(lines)


def save_waveforms_fourc(content: str, output: str) -> None:
    """Write a 4C linearinterpolation function block to a YAML file."""
    with open(output, "w") as f:
        f.write(content)
        f.write("\n")


def choose_waveform_fields(fields: list[str]) -> list[str]:
    """Ask the user to choose waveform fields interactively."""
    if not fields:
        raise ValueError("No waveform fields found in input file.")

    print("Available waveform fields:")
    for index, field in enumerate(fields, start=1):
        print(f"  {index}: {field}")

    while True:
        selected = input(
            "Choose waveform field numbers separated by commas, or 'all': "
        ).strip()
        if selected.lower() == "all":
            return fields

        try:
            selected_indices = [
                int(part.strip())
                for part in selected.split(",")
                if part.strip()
            ]
        except ValueError:
            print("Please enter valid numbers or 'all'.")
            continue

        if selected_indices and all(
            1 <= selected_index <= len(fields)
            for selected_index in selected_indices
        ):
            return [
                fields[selected_index - 1]
                for selected_index in selected_indices
            ]
        print(f"Please enter numbers between 1 and {len(fields)}.")


def prompt_float(label: str, default: float | None = None) -> float:
    """Ask the user to enter a floating point value."""
    default_text = (
        f" [{_format_number(default)}]" if default is not None else ""
    )
    while True:
        entered = input(f"{label}{default_text}: ").strip()
        if not entered and default is not None:
            return default
        try:
            return float(entered)
        except ValueError:
            print("Please enter a valid number.")


def build_parser() -> argparse.ArgumentParser:
    """Create the command line parser."""
    parser = argparse.ArgumentParser(
        description="Export Hamilton waveform to npy or 4C YAML formats."
    )
    parser.add_argument("waveform_file", help="Path to Hamilton waveform.txt")
    parser.add_argument(
        "--list-fields",
        action="store_true",
        help="List waveform fields in the input file and exit.",
    )
    parser.add_argument(
        "--fields",
        nargs="+",
        help="Waveform fields to export. Quote names that contain spaces.",
    )
    parser.add_argument(
        "--all-fields",
        action="store_true",
        help="Export all available waveform fields.",
    )
    parser.add_argument(
        "--start",
        type=float,
        help="Start time in seconds relative to the waveform recording start.",
    )
    parser.add_argument(
        "--end",
        type=float,
        help="End time in seconds relative to the waveform recording start.",
    )
    parser.add_argument(
        "--sampling-rate",
        type=float,
        help=(
            "Sampling rate in Hz for resampling. If omitted, exact original "
            "waveform samples are used."
        ),
    )
    parser.add_argument(
        "--format",
        choices=("npy", "fourc"),
        default="npy",
        help="Output format. Defaults to npy.",
    )
    parser.add_argument(
        "--output",
        help="Output file path. Required for all formats.",
    )
    parser.add_argument(
        "--funct",
        type=int,
        default=1,
        help="4C function number for --format fourc.",
    )
    parser.add_argument(
        "--variable-name",
        help="Variable name used inside the 4C function block.",
    )
    parser.add_argument(
        "--preserve-time",
        action="store_true",
        help="Keep original recording times instead of resetting to t=0.",
    )
    parser.add_argument(
        "--print",
        action="store_true",
        help="Print generated output summary or 4C function string to stdout.",
    )
    return parser


def main() -> None:
    """Run the Hamilton waveform export CLI."""
    parser = build_parser()
    args = parser.parse_args()

    df = load_waveform_txt(args.waveform_file)
    available_fields = get_waveform_fields(df)

    if args.list_fields:
        for field in available_fields:
            print(field)
        return

    selected_fields = _get_selected_fields(parser, args, available_fields)
    if args.format == "fourc" and len(selected_fields) != 1:
        parser.error(
            "--format fourc requires exactly one selected waveform field."
        )
    if not args.output:
        parser.error("--output is required.")

    available_start = float(df["Time (s)"].min())
    available_end = float(df["Time (s)"].max())
    print(
        "Available time range: "
        f"{_format_number(available_start)} s to "
        f"{_format_number(available_end)} s"
    )

    start = args.start
    if start is None:
        start = prompt_float("Start time in seconds", default=available_start)

    end = args.end
    if end is None:
        end = prompt_float("End time in seconds", default=available_end)

    if args.sampling_rate is None:
        print(
            "Using original waveform samples. Pass --sampling-rate to "
            "resample."
        )

    waveforms = extract_waveforms(
        df,
        fields=selected_fields,
        start=start,
        end=end,
        sampling_rate=args.sampling_rate,
        preserve_time=args.preserve_time,
    )

    if args.format == "npy":
        save_waveforms_npy(waveforms, args.output)
        print(
            f"Saved {len(selected_fields)} waveform field(s) with "
            f"{len(waveforms['time'])} samples to {args.output}."
        )
        return

    field = selected_fields[0]
    variable_name = args.variable_name or make_variable_name(field)
    function_string = format_fourc_linearinterpolation(
        times=waveforms["time"],
        values=waveforms[field],
        funct_number=args.funct,
        variable_name=variable_name,
    )
    save_waveforms_fourc(function_string, args.output)
    print(
        f"Saved 4C linearinterpolation function "
        f"for '{field}' with {len(waveforms['time'])} points to {args.output}."
    )
    if args.print:
        print(function_string)


def _get_selected_fields(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    available_fields: list[str],
) -> list[str]:
    if args.fields and args.all_fields:
        parser.error("Use either --fields or --all-fields, not both.")

    if args.all_fields:
        selected_fields = available_fields
    elif args.fields:
        selected_fields = args.fields
    else:
        selected_fields = choose_waveform_fields(available_fields)

    unknown_fields = [
        field for field in selected_fields if field not in available_fields
    ]
    if unknown_fields:
        parser.error(
            "Unknown waveform field(s): "
            f"{', '.join(unknown_fields)}. Available fields: "
            f"{', '.join(available_fields)}"
        )
    return selected_fields


def _format_number(value: float) -> str:
    return f"{value:.12g}"


if __name__ == "__main__":  # pragma: no cover
    main()
