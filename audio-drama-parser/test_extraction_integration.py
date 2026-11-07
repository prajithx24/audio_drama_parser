
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from audio_drama_parser.extraction import (
    ExtractionCompletenessError,
    ExtractionError,
    extract_characters_and_scenes,
)
from audio_drama_parser.ingestion import load_script
from audio_drama_parser.preprocessing import preprocess_text


MODEL_NAME = "gemini-3.5-flash-lite"
SCRIPT_PATH = PROJECT_ROOT / "data" / "raw" / "Modern_Shanmugi.txt"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "Modern_Shanmugi_extraction.json"


def main() -> None:
    try:
        raw_script = load_script(str(SCRIPT_PATH))
        cleaned_script = preprocess_text(raw_script)
        result = extract_characters_and_scenes(
            cleaned_script,
            model_name=MODEL_NAME,
        )
    except (ExtractionError, FileNotFoundError, UnicodeDecodeError, ValueError) as error:
        print(f"Extraction integration test failed: {type(error).__name__}: {error}")
        if isinstance(error, ExtractionCompletenessError):
            print("Completeness: FAILED")
        else:
            print("Completeness: NOT EVALUATED")
        return

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(
        json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    event_count = sum(len(scene.events) for scene in result.scenes)
    print(f"Model: {result.model_name}")
    print(f"Characters: {len(result.characters)}")
    print(f"Scenes: {len(result.scenes)}")
    print(f"Events: {event_count}")
    print(f"Warnings: {len(result.warnings)}")
    print(f"Character names: {', '.join(character.name for character in result.characters)}")
    print("Scene summaries:")
    for index, scene in enumerate(result.scenes, start=1):
        print(f"  {index}. {scene.summary}")
    print("Completeness: PASSED")
    print(f"Saved validated extraction: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
