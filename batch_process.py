import json
import os
from pathlib import Path
from process import ParliamentaryTranscriptProcessor

TRANSCRIPTS_DIR = "transcripts"  # or the folder where your transcripts are stored

def main():
    metadata_file = 'Metadata.json'  # or the path to your JSON file
    
    # Load the metadata file
    if not Path(metadata_file).exists():
        print(f"Error: Metadata file '{metadata_file}' does not exist.")
        return
    
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    if not isinstance(metadata, list):
        print("Error: Metadata JSON must be a list of video records.")
        return
    
    processor = ParliamentaryTranscriptProcessor()
        

    for video in metadata:  # ✅ Corrected loop
        video_id = video.get("video_id")
        title = video.get("title", "[Untitled]")
        if not video_id:
            print(f"⚠️ Skipping metadata entry with no video_id: {video}")
            continue

        transcript_path = Path(TRANSCRIPTS_DIR) / f"{video_id}.json"
        if not transcript_path.exists():
            print(f"❌ Transcript file missing for video ID: {video_id}")
            continue

        print(f"\n📺 Processing: {title} (ID: {video_id})")
        try:
            processor.process_transcript(str(transcript_path))
        except Exception as e:
            print(f"❌ Error processing '{video_id}': {e}")

    print("\n✅ All transcripts processed!")
if __name__ == "__main__":
    main()
