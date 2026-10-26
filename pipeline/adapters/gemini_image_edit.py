"""Edit a manifest of images with Gemini's low-cost image model."""

from __future__ import annotations

import argparse
import base64
import csv
import mimetypes
from pathlib import Path

from pipeline.stage1.gemini_common import build_gemini_client


def edit_manifest(manifest: Path, image_root: Path, output_root: Path, model: str, api_key_env: str, captions_csv: Path | None = None):
    client = build_gemini_client(api_key_env)
    with manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    captions = {}
    if captions_csv:
        with captions_csv.open(newline="", encoding="utf-8") as handle:
            captions = {row["file"]: row.get("PUBLIC_CAPTION", row.get("public_caption", "")) for row in csv.DictReader(handle)}
    for row in rows:
        relative = Path(row["file"])
        source = image_root / relative
        target = output_root / relative.with_suffix(".png")
        target.parent.mkdir(parents=True, exist_ok=True)
        with source.open("rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode("ascii")
        instruction = row.get("EDIT_INSTRUCTION", row.get("edit_instruction", ""))
        caption = row.get("PUBLIC_CAPTION", row.get("public_caption", captions.get(row["file"], "")))
        prompt = (
            "Edit the supplied photo to match this privacy-preserving scene description and edit instruction. "
            "Preserve the original framing, useful non-sensitive content, and photorealistic style. "
            "Do not add identifying personal details.\n"
            f"Public description: {caption}\nEdit instruction: {instruction}"
        )
        response = client.interactions.create(
            model=model,
            input=[
                {"type": "text", "text": prompt},
                {"type": "image", "data": image_data, "mime_type": mimetypes.guess_type(source.name)[0] or "image/jpeg"},
            ],
            response_format={"type": "image", "image_size": "1K"},
        )
        if not response.output_image:
            raise RuntimeError(f"Gemini returned no edited image for {relative}")
        target.write_bytes(base64.b64decode(response.output_image.data))
        print(f"{relative} -> {target}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="CSV with file, PUBLIC_CAPTION, EDIT_INSTRUCTION columns")
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--captions-csv", type=Path, help="CSV from generate_captions, used to add PUBLIC_CAPTION")
    parser.add_argument("--model", default="gemini-3.1-flash-lite-image")
    parser.add_argument("--api-key-env", default="GEMINI_API_KEY")
    args = parser.parse_args()
    edit_manifest(args.manifest, args.image_root, args.output_root, args.model, args.api_key_env, args.captions_csv)


if __name__ == "__main__":
    main()
