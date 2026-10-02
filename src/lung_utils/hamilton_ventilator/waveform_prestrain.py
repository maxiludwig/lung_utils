"""Prepend a prestraining flow interval to Hamilton waveform npy/yaml pairs."""

import argparse
from pathlib import Path

import numpy as np
from lung_utils.hamilton_ventilator.waveform_exporter import (
    format_fourc_linearinterpolation,
    make_variable_name,
    save_waveforms_fourc,
)

_INTERVAL_TOLERANCE = 1e-3


def load_npy(path: str) -> dict[str, np.ndarray]:
    data = np.load(path, allow_pickle=True).item()
    if "time" not in data:
        raise ValueError(f"npy file '{path}' has no 'time' key.")
    return data


def load_fourc_yaml(path: str) -> dict:
    """Parse a 4C linearinterpolation yaml and return its components."""
    import yaml

    with open(path) as f:
        doc = yaml.safe_load(f)

    funct_key = next(iter(doc))
    if not funct_key.startswith("FUNCT"):
        raise ValueError(f"Expected FUNCT<n> key, got '{funct_key}'.")
    funct_number = int(funct_key.removeprefix("FUNCT"))

    entries = doc[funct_key]
    symbolic = entries[0]["SYMBOLIC_FUNCTION_OF_TIME"]
    var_block = entries[1]
    variable_name = var_block["NAME"]
    times = np.array(var_block["TIMES"], dtype=float)
    values = np.array(var_block["VALUES"], dtype=float)

    return {
        "funct_number": funct_number,
        "symbolic": symbolic,
        "variable_name": variable_name,
        "times": times,
        "values": values,
    }


def detect_flow_field(
    npy_data: dict[str, np.ndarray], yaml_variable_name: str
) -> str:
    """Return the npy key whose make_variable_name matches the yaml
    variable."""
    matches = [
        key
        for key in npy_data
        if key != "time" and make_variable_name(key) == yaml_variable_name
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            "Multiple npy fields map to yaml variable "
            f"'{yaml_variable_name}': "
            + ", ".join(matches)
            + ". Use --flow-field to disambiguate."
        )
    raise ValueError(
        f"No npy field maps to yaml variable '{yaml_variable_name}'. "
        "Use --flow-field to specify the flow channel."
    )


def validate_interval_match(
    npy_times: np.ndarray,
    yaml_times: np.ndarray,
) -> None:
    for label, npy_t, yaml_t in (
        ("start", npy_times[0], yaml_times[0]),
        ("end", npy_times[-1], yaml_times[-1]),
    ):
        if abs(npy_t - yaml_t) > _INTERVAL_TOLERANCE:
            raise ValueError(
                f"Time interval mismatch at {label}: "
                f"npy={npy_t:.6f}s, yaml={yaml_t:.6f}s "
                f"(tolerance={_INTERVAL_TOLERANCE}s)."
            )


def make_cosine_flow_pulse(
    volume_ml: float, duration_s: float, n_points: int
) -> tuple[np.ndarray, np.ndarray]:
    """Raised-cosine (Hann) flow pulse integrating to volume_ml over
    duration_s.

    flow(t) = A * (1 - cos(2π t / T)) / 2,  A = 2 * volume_ml / duration_s
    """
    times = np.linspace(0.0, duration_s, n_points)
    amplitude = 2.0 * volume_ml / duration_s
    values = amplitude * (1.0 - np.cos(2.0 * np.pi * times / duration_s)) / 2.0
    return times, values


def build_prestrained_npy(
    npy_data: dict[str, np.ndarray],
    flow_field: str,
    pre_times: np.ndarray,
    pre_flow: np.ndarray,
) -> dict[str, np.ndarray]:
    measured_times = npy_data["time"]
    shift = pre_times[-1] + (measured_times[1] - measured_times[0])
    shifted_times = measured_times - measured_times[0] + shift

    n_pre = len(pre_times)
    result: dict[str, np.ndarray] = {}
    result["time"] = np.concatenate([pre_times, shifted_times])
    for key, values in npy_data.items():
        if key == "time":
            continue
        if key == flow_field:
            result[key] = np.concatenate([pre_flow, values])
        else:
            result[key] = np.concatenate([np.full(n_pre, np.nan), values])
    return result


def build_prestrained_yaml(
    yaml_data: dict,
    pre_times: np.ndarray,
    pre_flow: np.ndarray,
    measured_times: np.ndarray,
) -> str:
    shift = pre_times[-1] + (measured_times[1] - measured_times[0])
    shifted_times = measured_times - measured_times[0] + shift

    combined_times = np.concatenate([pre_times, shifted_times])
    combined_values = np.concatenate([pre_flow, yaml_data["values"]])

    return format_fourc_linearinterpolation(
        times=combined_times,
        values=combined_values,
        funct_number=yaml_data["funct_number"],
        variable_name=yaml_data["variable_name"],
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepend a prestraining flow interval to a Hamilton waveform "
            "npy/yaml pair for 4C simulations."
        )
    )
    parser.add_argument(
        "npy_file", help="Path to Hamilton waveform .npy file."
    )
    parser.add_argument(
        "yaml_file", help="Path to 4C linearinterpolation .yaml file."
    )
    parser.add_argument(
        "--volume",
        type=float,
        required=True,
        help="Volume delta in ml. Negative = outflow (deflation).",
    )
    parser.add_argument(
        "--duration",
        type=float,
        required=True,
        help="Prestraining interval length in seconds.",
    )
    parser.add_argument(
        "--sampling-rate",
        type=float,
        default=50.0,
        help="Sampling rate in Hz for the prestraining curve (default: 50).",
    )
    parser.add_argument(
        "--flow-field",
        help=(
            "Name of the flow field in the npy file. Auto-detected from the "
            "yaml variable name if not specified."
        ),
    )
    parser.add_argument(
        "--output-npy",
        help=(
            "Output npy path. "
            "Default: <npy_stem>_prestrained.npy in same directory."
        ),
    )
    parser.add_argument(
        "--output-yaml",
        help=(
            "Output yaml path. "
            "Default: <yaml_stem>_prestrained.yaml in same directory."
        ),
    )
    return parser


def main() -> None:
    """Run the Hamilton waveform prestraining CLI."""
    parser = build_parser()
    args = parser.parse_args()

    if args.volume == 0.0:
        parser.error("--volume must be non-zero.")
    if args.duration <= 0.0:
        parser.error("--duration must be positive.")
    if args.sampling_rate <= 0.0:
        parser.error("--sampling-rate must be positive.")

    npy_data = load_npy(args.npy_file)
    yaml_data = load_fourc_yaml(args.yaml_file)

    validate_interval_match(npy_data["time"], yaml_data["times"])

    flow_field = args.flow_field or detect_flow_field(
        npy_data, yaml_data["variable_name"]
    )
    if flow_field not in npy_data:
        parser.error(
            f"Flow field '{flow_field}' not found in npy file. "
            "Available fields: "
            + ", ".join(k for k in npy_data if k != "time")
        )

    n_points = max(2, round(args.duration * args.sampling_rate))
    pre_times, pre_flow = make_cosine_flow_pulse(
        volume_ml=args.volume,
        duration_s=args.duration,
        n_points=n_points,
    )

    prestrained_npy = build_prestrained_npy(
        npy_data, flow_field, pre_times, pre_flow
    )
    prestrained_yaml = build_prestrained_yaml(
        yaml_data, pre_times, pre_flow, npy_data["time"]
    )

    npy_path = Path(args.npy_file)
    yaml_path = Path(args.yaml_file)
    output_npy = args.output_npy or str(
        npy_path.parent / f"{npy_path.stem}_prestrained.npy"
    )
    output_yaml = args.output_yaml or str(
        yaml_path.parent / f"{yaml_path.stem}_prestrained.yaml"
    )

    np.save(output_npy, prestrained_npy, allow_pickle=True)
    save_waveforms_fourc(prestrained_yaml, output_yaml)

    n_pre = len(pre_times)
    n_meas = len(npy_data["time"])
    print(
        f"Prestraining: {n_pre} points over {args.duration}s, "
        f"volume delta {args.volume:+.1f} ml, flow field '{flow_field}'."
    )
    print(
        f"Combined: {n_pre + n_meas} points "
        f"({n_pre} prestraining + {n_meas} measured)."
    )
    print(f"Saved npy  → {output_npy}")
    print(f"Saved yaml → {output_yaml}")
