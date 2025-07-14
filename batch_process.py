import json
import os
from pathlib import Path
from process import ParliamentaryTranscriptProcessor

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
    
    for video in metadata:
        transcript_info = video.get("transcript", {})
        
        has_transcript = transcript_info.get("hasTranscript", False)
        transcript_filename = transcript_info.get("formattedContent")
        
        if not has_transcript or not transcript_filename:
            print(f"Skipping video '{video.get('Video_title')}' - No transcript.")
            continue
        
        if not Path(transcript_filename).exists():
            print(f"Transcript file '{transcript_filename}' does not exist. Skipping.")
            continue
        
        print(f"\n=== Processing Video: {video.get('Video_title')} ===")
        try:
            processor.process_transcript(transcript_filename)
        except Exception as e:
            print(f"Error processing {transcript_filename}: {e}")
    
    print("\n✅ All transcripts processed!")

if __name__ == "__main__":
    main()