from __future__ import annotations

import argparse
import os
from typing import Mapping, Sequence

from database_preparation import prepare_bloodmnist_artifacts
from feature_extraction import (
    encode_and_store_image_features,
    run_full_pipeline,
    store_prompt_features,
)
from model_deployment import load_conceptclip_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ConceptCLIP feature store pipeline")
    parser.add_argument("--hf-token", default=os.getenv("HF_TOKEN"))
    parser.add_argument("--output-path", default="./conceptclip_features.h5")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--no-download", action="store_true", default=False)
    parser.add_argument("--skip-text", action="store_true", default=False)
    parser.add_argument("--skip-image", action="store_true", default=False)
    parser.add_argument("--no-resume", action="store_true", default=False)
    parser.add_argument("--no-progress", action="store_true", default=False)
    parser.add_argument("--data-root", default=None)
    return parser


def run_pipeline(args: argparse.Namespace) -> None:
    loaders, prompts = prepare_bloodmnist_artifacts(
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        download=not args.no_download,
        data_root=args.data_root,
    )
    artifacts = load_conceptclip_model(hf_token=args.hf_token)
    metadata: Mapping[str, object] = {"dataset": "bloodmnist"}
    if args.skip_text and args.skip_image:
        print("Nothing to do. At least one of text or image stages must run.")
        return
    if not args.skip_text and not args.skip_image:
        summary = run_full_pipeline(
            artifacts,
            prompts,
            loaders,
            output_path=args.output_path,
            resume=not args.no_resume,
            show_progress=not args.no_progress,
            extra_metadata=metadata,
        )
        print("Stored text templates:")
        for template_id, count in summary.text_templates.items():
            print(f"  - {template_id}: {count}")
        print("Image samples appended:")
        for split_name, count in summary.image_samples_appended.items():
            print(f"  - {split_name}: {count}")
        print(f"Total new samples: {summary.total_new_samples}")
        return
    if not args.skip_text:
        stored = store_prompt_features(
            artifacts,
            prompts,
            output_path=args.output_path,
            extra_metadata=metadata,
        )
        print("Stored text templates:")
        for template_id, count in stored.items():
            print(f"  - {template_id}: {count}")
    if not args.skip_image:
        appended = encode_and_store_image_features(
            artifacts,
            loaders,
            output_path=args.output_path,
            resume=not args.no_resume,
            show_progress=not args.no_progress,
            extra_metadata=metadata,
        )
        print("Image samples appended:")
        for split_name, count in appended.items():
            print(f"  - {split_name}: {count}")
        print(f"Total new samples: {sum(appended.values())}")


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    run_pipeline(args)


if __name__ == "__main__":
    main()
