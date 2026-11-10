import argparse
import sys

from . import engines, models
from .env import load_dotenv
from .storage import human_size


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="local-llms")
    sub = parser.add_subparsers(dest="command", required=True)

    dl = sub.add_parser("download", help="download model weights or an inference engine")
    what = dl.add_mutually_exclusive_group(required=True)
    what.add_argument("-m", "--model", help="Hugging Face repo id, e.g. org/name")
    what.add_argument("-e", "--engine", help=f"inference engine: {', '.join(engines.ENGINES)}")
    dl.add_argument("-q", "--quantization", help="model quantization, e.g. Q4_K_M")
    dl.add_argument("--version", help="engine version, e.g. b12138 (default: latest)")
    dl.add_argument(
        "--build", help="engine build, e.g. cpu, cuda, vulkan (default: detected from hardware)"
    )

    ls = sub.add_parser("list", help="list downloaded assets")
    ls.add_argument("--models", action="store_true", help="list downloaded models")
    ls.add_argument("--engines", action="store_true", help="list installed engines")

    rm = sub.add_parser("remove", help="remove a downloaded model or engine")
    what = rm.add_mutually_exclusive_group(required=True)
    what.add_argument("-m", "--model", help="model id to remove")
    what.add_argument("-e", "--engine", help="engine to remove")
    rm.add_argument(
        "-q", "--quantization", help="remove only this quantization (default: all)"
    )
    rm.add_argument("--version", help="remove only this engine version (default: all)")

    up = sub.add_parser("update", help="update an installed engine to the latest release")
    up.add_argument("-e", "--engine", required=True, help="engine to update")
    return parser


def print_table(rows: list[tuple[str, ...]]) -> None:
    widths = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    for r in rows:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip())


def cmd_list_models() -> None:
    entries = models.list_models()
    if not entries:
        print("No models downloaded.")
        return
    print_table(
        [("MODEL", "QUANT", "SIZE", "SOURCE")]
        + [(e.model, e.quantization or "-", human_size(e.size), e.source) for e in entries]
    )


def cmd_list_engines() -> None:
    entries = engines.list_engines()
    if not entries:
        print("No engines installed.")
        return
    print_table(
        [("ENGINE", "VERSION", "BUILD", "SIZE", "PATH")]
        + [(e.engine, e.version, e.build, human_size(e.size), e.path) for e in entries]
    )


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    if getattr(args, "model", None) and (getattr(args, "version", None) or getattr(args, "build", None)):
        parser.error("--version and --build only apply to engines (-e)")
    if getattr(args, "engine", None) and getattr(args, "quantization", None):
        parser.error("--quantization only applies to models (-m)")
    try:
        if args.command == "download":
            if args.model:
                models.download(args.model, args.quantization)
            else:
                engines.install(args.engine, args.version, args.build)
        elif args.command == "list":
            if not (args.models or args.engines):
                print("Specify what to list, e.g. --models or --engines", file=sys.stderr)
                sys.exit(2)
            if args.models:
                cmd_list_models()
            if args.models and args.engines:
                print()
            if args.engines:
                cmd_list_engines()
        elif args.command == "remove":
            if args.model:
                for e in models.remove(args.model, args.quantization):
                    print(f"Removed {e.model} ({e.quantization or 'full'}), freed {human_size(e.size)}")
            else:
                for e in engines.remove(args.engine, args.version):
                    print(f"Removed {e.engine} {e.version} ({e.build}), freed {human_size(e.size)}")
        elif args.command == "update":
            engines.update(args.engine)
    except (models.ModelError, engines.EngineError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
