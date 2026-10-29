import argparse
import sys

from . import models
from .storage import human_size


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="local-llms")
    sub = parser.add_subparsers(dest="command", required=True)

    dl = sub.add_parser("download", help="download model weights from Hugging Face")
    dl.add_argument("-m", "--model", required=True, help="Hugging Face repo id, e.g. org/name")
    dl.add_argument("-q", "--quantization", help="quantization, e.g. Q4_K_M")

    ls = sub.add_parser("list", help="list downloaded assets")
    ls.add_argument("--models", action="store_true", help="list downloaded models")

    rm = sub.add_parser("remove", help="remove a downloaded model")
    rm.add_argument("-m", "--model", required=True, help="model id to remove")
    rm.add_argument(
        "-q", "--quantization", help="remove only this quantization (default: all)"
    )
    return parser


def cmd_list() -> None:
    entries = models.list_models()
    if not entries:
        print("No models downloaded.")
        return
    rows = [("MODEL", "QUANT", "SIZE", "SOURCE")] + [
        (e.model, e.quantization or "-", human_size(e.size), e.source) for e in entries
    ]
    widths = [max(len(r[i]) for r in rows) for i in range(4)]
    for r in rows:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip())


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "download":
            models.download(args.model, args.quantization)
        elif args.command == "list":
            if not args.models:
                print("Specify what to list, e.g. --models", file=sys.stderr)
                sys.exit(2)
            cmd_list()
        elif args.command == "remove":
            for e in models.remove(args.model, args.quantization):
                print(f"Removed {e.model} ({e.quantization or 'full'}), freed {human_size(e.size)}")
    except models.ModelError as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
