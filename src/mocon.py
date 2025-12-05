#!/usr/bin/env python3
"""
mocon - Motion Capture Converter

Convert motion capture files between formats.
Currently supports: FBX → AMASS (ASAP-compatible)

Usage:
    mocon input.fbx                     # Convert to AMASS format
    mocon input.fbx -o output.npz       # Specify output
    mocon input.fbx --info              # Show file info only
    mocon input.fbx --mapping cmu       # Use CMU joint mapping
"""
import argparse
import sys
from pathlib import Path

from src.core.registry import FormatRegistry
from src.core.motion import MotionData
from src.core.mapping import JointMapping, SMPL_JOINTS

# Import readers/writers to register them
from src.readers import fbx_reader  # noqa: F401
from src.writers import amass_writer  # noqa: F401


def convert(
    input_path: str,
    output_path: str = None,
    output_format: str = None,
    target_fps: float = None,
    mapping_name: str = None,
    info_only: bool = False,
    show_mapping: bool = False,
    scale: float = None,
) -> Path:
    """
    Convert a motion file from one format to another.

    Args:
        input_path: Input file path
        output_path: Output file path (auto-generated if not specified)
        output_format: Output format name (default: amass)
        target_fps: Resample to this frame rate
        mapping_name: Joint mapping name or path to mapping file
        info_only: Only print info, don't convert
        show_mapping: Show joint mapping details
        scale: Scale factor for translation (e.g., 0.01 to reduce by 100x)

    Returns:
        Path to output file (or None if info_only)
    """
    input_path = Path(input_path)

    # Load mapping
    try:
        mapping = JointMapping.from_name(mapping_name) if mapping_name else JointMapping.default()
    except (FileNotFoundError, ValueError) as e:
        print(f"Error loading mapping: {e}")
        sys.exit(1)

    # Get reader
    reader_class = FormatRegistry.get_reader(extension=input_path.suffix)
    if reader_class is None:
        print(f"Error: No reader for format: {input_path.suffix}")
        print(f"Supported formats: {FormatRegistry.list_readers()}")
        sys.exit(1)

    reader = reader_class()

    # Read motion
    print(f"Reading: {input_path}")
    try:
        motion = reader.read(input_path)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)

    # Print info
    motion.print_info()

    # Show mapping if requested
    if show_mapping:
        _print_mapping_report(motion, mapping)

    if info_only:
        return None

    # Determine output format and path
    if output_format is None:
        output_format = "amass"  # Default for ASAP

    # Get writer
    writer_class = FormatRegistry.get_writer(format_name=output_format)
    if writer_class is None:
        print(f"Error: No writer for format: {output_format}")
        print(f"Supported formats: {FormatRegistry.list_writers()}")
        sys.exit(1)

    # Create writer with mapping
    writer = writer_class(mapping=mapping)

    # Determine output path
    if output_path is None:
        ext = writer.get_extension()
        output_path = input_path.with_suffix(ext)
    else:
        output_path = Path(output_path)

    # Write
    print(f"\nWriting: {output_path}")
    kwargs = {}
    if target_fps:
        kwargs["target_fps"] = target_fps
    if scale:
        kwargs["scale"] = scale
        print(f"[mocon] Applying translation scale: {scale}")

    result_path = writer.write(motion, output_path, **kwargs)

    print(f"\n{'=' * 60}")
    print("✓ Conversion complete!")
    print(f"{'=' * 60}")
    print(f"\nOutput: {result_path}")
    print(f"\nTo use with ASAP:")
    print(f"  1. Copy to: ASAP/humanoidverse/data/motions/raw_tairantestbed_smpl/")
    print(
        f"  2. Run: python scripts/data_process/fit_smpl_motion.py +robot=g1/g1_29dof_anneal_23dof"
    )

    return result_path


def _print_mapping_report(motion: MotionData, mapping: JointMapping):
    """Print detailed joint mapping report."""
    from src.writers.amass_writer import AMASSWriter

    writer = AMASSWriter(mapping=mapping)
    report = writer.get_mapping_report(motion)

    print(f"\n{'=' * 60}")
    print(f"JOINT MAPPING REPORT ({mapping.name})")
    print(f"{'=' * 60}")

    if mapping.description:
        print(f"  {mapping.description}")

    mapped = []
    unmapped = []

    for source, target in report.items():
        if target == "NOT MAPPED":
            unmapped.append(source)
        else:
            mapped.append((source, target))

    print(f"\nMapped ({len(mapped)}/{len(motion.joint_names)}):")
    for source, target in mapped:
        print(f"  {source:30s} → {target}")

    if unmapped:
        print(f"\nUnmapped ({len(unmapped)}):")
        for source in unmapped:
            print(f"  {source:30s} → ?")

    # Show which SMPL joints are covered
    covered_smpl = set(target for _, target in mapped)
    missing_smpl = set(SMPL_JOINTS) - covered_smpl

    if missing_smpl:
        print(f"\nMissing SMPL joints ({len(missing_smpl)}):")
        for joint in missing_smpl:
            print(f"  {joint}")


def list_mappings():
    """List all available joint mappings."""
    available = JointMapping.list_available()

    print("Available joint mappings:")
    print()

    for name in available:
        try:
            m = JointMapping.from_name(name)
            print(f"  {name:20s} - {m.description or 'No description'}")
        except Exception:
            print(f"  {name:20s} - (error loading)")

    print()
    print("Usage:")
    print("  mocon input.fbx --mapping default")
    print("  mocon input.fbx --mapping /path/to/custom.json")


def main():
    parser = argparse.ArgumentParser(
        prog="mocon",
        description="mocon - Motion Capture Converter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  mocon input.fbx                          # Convert FBX to AMASS
  mocon input.fbx -o output.npz            # Specify output path
  mocon input.fbx --info                   # Print file info only
  mocon input.fbx --fps 30                 # Resample to 30 FPS
  mocon input.fbx --show-mapping           # Show joint mapping details
  mocon input.fbx --mapping cmu            # Use CMU joint mapping
  mocon input.fbx --mapping ./my_rig.json  # Use custom mapping file
  mocon --list-mappings                    # List available mappings
        """,
    )

    parser.add_argument("input", nargs="?", help="Input motion file (e.g., animation.fbx)")
    parser.add_argument("-o", "--output", help="Output file path")
    parser.add_argument("-f", "--format", default="amass", help="Output format (default: amass)")

    # Mapping options
    parser.add_argument(
        "-m",
        "--mapping",
        help="Joint mapping name (e.g., 'default', 'cmu') or path to custom .json file",
    )
    parser.add_argument(
        "--list-mappings", action="store_true", help="List all available joint mappings and exit"
    )

    # Info options
    parser.add_argument("--info", action="store_true", help="Print file info without converting")
    parser.add_argument(
        "--show-mapping", action="store_true", help="Show detailed joint mapping report"
    )

    # Processing options
    parser.add_argument("--fps", type=float, help="Resample to target FPS")
    parser.add_argument(
        "--scale",
        type=float,
        help="Scale factor for XY translation (e.g., 0.1 to reduce movement by 10x)",
    )

    args = parser.parse_args()

    # Handle --list-mappings
    if args.list_mappings:
        list_mappings()
        sys.exit(0)

    # Require input
    if not args.input:
        parser.print_help()
        sys.exit(1)

    # Run conversion
    convert(
        input_path=args.input,
        output_path=args.output,
        output_format=args.format,
        target_fps=args.fps,
        mapping_name=args.mapping,
        info_only=args.info,
        show_mapping=args.show_mapping,
        scale=args.scale,
    )


if __name__ == "__main__":
    main()
