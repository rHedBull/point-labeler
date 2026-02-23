"""Unified CLI for the industrial point labeler pipeline."""

import sys


def main():
    commands = {
        "segment": ("industrial_point_labeler.segmentation.ransac", "RANSAC primitive segmentation"),
        "label": ("industrial_point_labeler.labeler.server", "Interactive GT annotation tool"),
        "doubles": ("industrial_point_labeler.doubles.server", "Doubles dissection tool"),
        "build-graph": ("industrial_point_labeler.graph.builder", "Build pipe-endpoint connectivity graph"),
        "review-graph": ("industrial_point_labeler.graph.server", "Interactive graph review tool"),
        "describe-equipment": ("industrial_point_labeler.equipment.server", "Equipment description tool"),
    }

    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("usage: ipl <command> [options]\n")
        print("Industrial Point Labeler — interactive annotation toolkit\n")
        print("commands:")
        for cmd, (_, desc) in commands.items():
            print(f"  {cmd:<20s} {desc}")
        print(f"\nRun 'ipl <command> --help' for command-specific options.")
        sys.exit(0)

    cmd = sys.argv[1]
    if cmd not in commands:
        print(f"Unknown command: {cmd}")
        print(f"Available: {', '.join(commands)}")
        sys.exit(1)

    module_path, _ = commands[cmd]
    # Strip the command name so each module's argparse sees only its own args
    sys.argv = [f"ipl {cmd}"] + sys.argv[2:]

    import importlib
    mod = importlib.import_module(module_path)
    mod.main()


if __name__ == "__main__":
    main()
