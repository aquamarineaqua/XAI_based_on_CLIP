from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np

from database_access import (
    access_batch,
    access_sample_by_index,
    clear_all_templates,
    fetch_template_features,
    get_template_names,
    summarize_h5,
)


def _parse_indices(arg: Sequence[str]) -> Sequence[int] | slice | None:
    if not arg:
        return None
    if len(arg) == 1 and ":" in arg[0]:
        start_str, end_str, *rest = arg[0].split(":") + [""]
        start = int(start_str) if start_str else None
        end = int(end_str) if end_str else None
        step = int(rest[0]) if rest and rest[0] else None
        return slice(start, end, step)
    return [int(x) for x in arg]


def _print_array(arr: np.ndarray) -> None:
    if arr.ndim == 0:
        print(arr.item())
    else:
        print(arr)


def cmd_summarize(args: argparse.Namespace) -> None:
    summarize_h5(args.path, max_templates=args.max_templates)


def cmd_sample(args: argparse.Namespace) -> None:
    data = access_sample_by_index(args.index, path=args.path)
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            print(f"{key}: shape={value.shape}")
        else:
            print(f"{key}: {value}")


def cmd_batch(args: argparse.Namespace) -> None:
    indices = _parse_indices(args.idx)
    result = access_batch(
        split=args.split,
        idx=indices,
        feature=args.feature,
        path=args.path,
    )
    _print_array(result)


def cmd_templates(args: argparse.Namespace) -> None:
    names = get_template_names(args.path)
    print("templates:")
    for name in names:
        print("  -", name)


def cmd_clear(args: argparse.Namespace) -> None:
    clear_all_templates(args.path)


def cmd_fetch(args: argparse.Namespace) -> None:
    template_ids = args.template or None
    features = fetch_template_features(
        template_id=template_ids,
        is_concept=args.concept,
        concept_list=args.concept_list,
        is_label=args.label,
        feature=args.feature,
        path=args.path,
    )
    if isinstance(features, dict):
        summary = {key: value.shape for key, value in features.items()}
        print(json.dumps(summary, indent=2))
    else:
        print(features.shape)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ConceptCLIP H5 access helper")
    parser.add_argument("--path", default="./conceptclip_features.h5")
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_sum = subparsers.add_parser("summarize", help="Print dataset overview")
    p_sum.add_argument("--max-templates", type=int, default=3)
    p_sum.set_defaults(func=cmd_summarize)

    p_sample = subparsers.add_parser("sample", help="Inspect a single sample")
    p_sample.add_argument("index", type=int)
    p_sample.set_defaults(func=cmd_sample)

    p_batch = subparsers.add_parser("batch", help="Fetch a batch of features")
    p_batch.add_argument("--split", default="all")
    p_batch.add_argument("--feature", choices=["image", "patches", "label"], default="image")
    p_batch.add_argument("--idx", nargs="*", default=[])
    p_batch.set_defaults(func=cmd_batch)

    p_templates = subparsers.add_parser("templates", help="List template identifiers")
    p_templates.set_defaults(func=cmd_templates)

    p_clear = subparsers.add_parser("clear-templates", help="Remove all template groups")
    p_clear.set_defaults(func=cmd_clear)

    p_fetch = subparsers.add_parser("fetch", help="Fetch template features")
    p_fetch.add_argument("--template", nargs="*", default=None)
    p_fetch.add_argument("--concept", action="store_true", default=False)
    p_fetch.add_argument("--label", action="store_true", default=False)
    p_fetch.add_argument("--concept-list", nargs="*", default=None)
    p_fetch.add_argument("--feature", choices=["text", "tokens"], default="text")
    p_fetch.set_defaults(func=cmd_fetch)

    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
