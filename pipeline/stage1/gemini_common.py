"""Small Gemini API helpers used by Stage 1 and image editing."""

from __future__ import annotations

import os
from pathlib import Path

from PIL import Image


def build_gemini_client(api_key_env: str = "GEMINI_API_KEY"):
    from google import genai

    api_key = os.environ.get(api_key_env) or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError(f"Set {api_key_env} (or GOOGLE_API_KEY) before running Gemini jobs")
    return genai.Client(api_key=api_key)


def run_gemini_batch(client, model_id: str, messages_batch: list, image_paths=None, max_new_tokens=512) -> list[str]:
    outputs = []
    for i, messages in enumerate(messages_batch):
        system_prompt, user_prompt = (message["content"] for message in messages)
        contents = [f"{system_prompt}\n\n{user_prompt}"]
        if image_paths is not None:
            contents.append(Image.open(Path(image_paths[i])).convert("RGB"))
        response = client.models.generate_content(
            model=model_id,
            contents=contents,
            config={"max_output_tokens": max_new_tokens},
        )
        outputs.append((response.text or "").strip())
    return outputs
