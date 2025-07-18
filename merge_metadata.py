import os
import json
from pathlib import Path

TRANSCRIPTS_DIR = "transcripts"
OUTPUT_FILE = "Metadata.json"

def merge_metadata_files():
    transcripts_path = Path(TRANSCRIPTS_DIR)
    if not transcripts_path.exists():
        print(f"❌ Folder '{TRANSCRIPTS_DIR}' not found.")
        return

    metadata_files = list(transcripts_path.glob("*_meta.json"))
    if not metadata_files:
        print("⚠️ No metadata files found to merge.")
        return

    merged_metadata = []

    for meta_file in metadata_files:
        try:
            with open(meta_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)

                # Add filename info for traceability
                metadata["source_file"] = meta_file.name

                merged_metadata.append(metadata)
        except Exception as e:
            print(f"❌ Failed to read {meta_file.name}: {e}")

    # Write merged file
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as out_file:
        json.dump(merged_metadata, out_file, indent=2, ensure_ascii=False)

    print(f"\n✅ Merged {len(merged_metadata)} metadata files into: {OUTPUT_FILE}")

if __name__ == "__main__":
    merge_metadata_files()
