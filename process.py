
"""
Parliamentary Transcript Processor

This script processes raw AI-transcribed parliamentary session data from India
using LangChain and Google's Gemini model to clean, correct, and format the text
for Knowledge Graph extraction.

Requirements:
- langchain-google-genai
- python-dotenv (optional, for environment variables)

Usage:
    python transcript_processor.py input_file.txt output_file.txt
"""

import sys
import time
import os
import json
from pathlib import Path

try:
    from langchain_google_genai import ChatGoogleGenerativeAI
    from langchain.schema import HumanMessage, SystemMessage
    from dotenv import load_dotenv
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Please install required packages:")
    print("pip install langchain-google-genai python-dotenv")
    sys.exit(1)

# Load environment variables
load_dotenv()

class ParliamentaryTranscriptProcessor:
    def __init__(self, api_key: str = None):
        """
        Initialize the processor with Gemini model.
        
        Args:
            api_key: Google API key. If None, will try to get from environment.
        """
        if api_key is None:
            api_key = os.getenv('GOOGLE_API_KEY')
            
        if not api_key:
            raise ValueError(
                "Google API key is required. Set GOOGLE_API_KEY environment variable "
                "or pass api_key parameter."
            )
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash-preview-05-20",
            google_api_key=api_key,
            temperature=1.0,
            thinking_budget=0  # Disable thinking mode
        )
        
        self.system_prompt = self._get_system_prompt()
    
    def estimate_tokens(self, text: str) -> int:
        """
        Roughly estimate token count: ~4 chars = 1 token (English).
        """
        return len(text) // 4

    def _get_system_prompt(self) -> str:
        """Return the comprehensive system prompt for transcript processing."""
        return """**Objective:**
The primary objective of this task is to transform raw, AI-transcribed parliamentary session data from India's into a clean, accurate, and time-aligned textual format suitable for subsequent Knowledge Graph (KG) extraction. This involves precise sentence segmentation, robust transcription correction, and careful handling of names and parliamentary conventions.

**Processing Instructions:**

1. **Comprehensive Transcription Correction:**

   - **Error Identification:** Carefully examine all text segments for transcription errors. These include spelling mistakes, grammatical inaccuracies, missing punctuation, incorrect sentence boundaries, and mishearings (such as homophones or fragmented/miscombined words).

   - **Clarity & Accuracy:** Correct every identified error to ensure the text is grammatically correct, semantically clear, and reflective of the speaker’s intent. The final output must be natural-sounding, highly readable, and preserve the original meaning of the spoken content.

   - **Indian English & Parliamentary Context:** Pay close attention to nuances of Indian English (e.g., formal tone, idiomatic usage, localized vocabulary) and terminology common in Indian parliamentary sessions. Use Indian spellings and conventions (e.g., "Honourable" not "Honorable", "licence" not "license").

   - **Formal Diction:** Retain the formal and respectful tone typically used in the Indian Parliament. Ensure phrases such as “with due respect,” “Honourable Speaker,” or “through you, Sir/Madam” are preserved or corrected for fluency if needed.

   - **Preserve Parliamentary Register:** Where applicable, keep formal speech patterns typical of Lok Sabha or Rajya Sabha discussions, such as:
     - “I rise to speak on...”
     - “I would like to bring to the attention of the House...”
     - “Through you, Speaker Sir...”

2. **Strict Name & Entity Handling (Indian Parliament Context):**

   - **Extreme Caution with Proper Nouns:** Exercise the utmost caution with proper nouns, especially names of individuals such as Members of Parliament (MPs), Union Ministers, Chief Ministers, or citizens mentioned in the proceedings. Also, take care with names of political parties, specific legislations (e.g., "Women's Reservation Bill"), states, constituencies, and government bodies (e.g., NITI Aayog, Ministry of External Affairs).

   - **Ambiguity Resolution:** If there is *any* doubt regarding the correct spelling, pronunciation, or identity of a proper noun, or if the transcribed name sounds ambiguous or phonetically inaccurate, **replace it with `[unknown]`**. This ensures factual accuracy for downstream Knowledge Graph processing.

   - **Parliamentary Titles:** Accurately identify and capitalize titles and forms of address frequently used in the Indian Parliament, such as:
     - "Honourable Member of Parliament"
     - "Honourable Speaker"
     - "Madam Chairperson"
     - "Honourable Minister of Finance"
     - "Prime Minister"
     - "Leader of the Opposition"
     - "Chief Minister"
     - "Shri" / "Smt." / "Dr." / "Prof." (where clearly spoken and identifiable)

     Do *not* use `[unknown]` for such titles or roles if the context is unambiguous, even if the specific name is not clearly transcribed.

   - **Avoid Political Bias:** Maintain neutrality while editing political statements. Preserve the original meaning and tone of the speaker without inserting or deleting politically sensitive content.

3. **Accurate Sentence Segmentation (NLP-driven):**
   - **Complete Sentences:** Using advanced Natural Language Processing (NLP) techniques, accurately identify and segment complete sentences. This will often involve merging text segments from multiple input JSON objects to form a single, coherent sentence, or occasionally splitting a single text segment if it contains more than one complete sentence.
   - **Logical Flow:** Ensure that each output line represents a complete, grammatically correct, and logically coherent sentence. Avoid creating fragmented sentences or run-on sentences.
   - **Punctuation:** Insert appropriate punctuation marks (periods, question marks, exclamation points, commas, etc.) to ensure sentence clarity and correctness.

4. **Timecode Assignment:**
   - **Sentence Start Time:** For each identified complete sentence, assign the start time (in integer seconds) of the *first* input JSON segment that contributes to that sentence.
   - **Rounding:** Convert the start time from the original string (e.g., "103.840") to an integer number of seconds, rounding *down* to the nearest whole second (e.g., "103.840" becomes 103, "109.360" becomes 109).

**Output Format:**
The output should be plain text, where each line represents a single, complete, time-aligned, and corrected sentence. The format for each line must be:

`[integer_seconds] [Complete and corrected sentence]`

**Key Considerations:**
- **Precision is Paramount:** Every correction and segmentation decision impacts the quality of the downstream Knowledge Graph.
- **Contextual Understanding:** Leverage the context of parliamentary proceedings to inform decisions, especially regarding names and formal language.
- **Balance of Correction and Preservation:** While fixing errors, avoid altering the fundamental meaning or intent of the speaker's original words.
- **No Interpretation:** Do not summarize, interpret, or add information not explicitly stated. Focus solely on cleaning and structuring the transcript.

Process the following transcript data and return ONLY the formatted output with no additional commentary or explanation:"""

    def load_transcript(self, file_path: str) -> str:
        """
        Load transcript data from a file as raw text.
        
        Args:
            file_path: Path to the input file containing transcript data
            
        Returns:
            Raw file content as string
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return content
        except FileNotFoundError:
            raise FileNotFoundError(f"Input file not found: {file_path}")
        except Exception as e:
            raise Exception(f"Error loading transcript: {str(e)}")

    def split_large_transcript_file(self, input_file, max_lines=5000):
        """
        Split JSON transcript file into smaller parts if needed.
        """
        input_path = Path(input_file)
        with open(input_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError("Expected JSON to be a list.")

        total_segments = len(data)
        if total_segments <= max_lines:
            return []

        chunks = [
            data[i:i + max_lines]
            for i in range(0, total_segments, max_lines)
        ]

        base_name = input_path.stem
        new_files = []
        for idx, chunk in enumerate(chunks, start=1):
            output_file = input_path.parent / f"{base_name}_part{idx}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(chunk, f, ensure_ascii=False, indent=2)
            new_files.append(str(output_file))
            print(f"Created chunk: {output_file} ({len(chunk)} segments)")
        return new_files

    def process_transcript(self, input_file: str):
        """
        Process the entire transcript file using Gemini's large context window.
        
        Args:
            input_file: Path to input JSON transcript file
        """
        # Generate output filename by replacing .json with .txt
        input_path = Path(input_file)
        if input_path.suffix.lower() != '.json':
            raise ValueError(f"Input file must be a .json file, got: {input_file}")
        
        # ✅ NEW: Ensure output folder exists
        output_folder = Path("first_clean")
        output_folder.mkdir(exist_ok=True)

        # ✅ NEW: Output filename inside first_clean folder with .txt extension
        output_filename = input_path.stem + ".txt"  # e.g., "OXV3AKneBtE.txt"
        output_file = output_folder / output_filename
        
        print(f"Loading transcript from: {input_file}")
        content = self.load_transcript(input_file)
        print(f"Loaded transcript file ({len(content)} characters)")

        est_tokens = self.estimate_tokens(content)
        print(f"Estimated input tokens: {est_tokens}")

        MAX_SAFE_TOKENS = 200_000  # keep a safety margin for Gemini free tier

        if est_tokens > MAX_SAFE_TOKENS:
            print(f"⚡ Input too large ({est_tokens} tokens). Splitting...")
            chunk_files = self.split_large_transcript_file(input_file, max_lines=5000)
            for chunk_file in chunk_files:
                print(f"Reprocessing chunk: {chunk_file}")
                self.process_transcript(chunk_file)
                time.sleep(10)  # prevent quota bursts
            return
        
        # Create messages for the conversation
        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=f"Please process this transcript data:\n\n{content}")
        ]
        
        print("Processing transcript with Gemini model...")
        try:
            # Get response from the model
            response = self.llm.invoke(messages)
            
            # Debug: Print response details
            print(f"Response type: {type(response)}")
            print(f"Response content length: {len(response.content) if response.content else 'None'}")
            
            result = response.content.strip() if response.content else ""
            
            if not result:
                print("Warning: Model returned empty content!")
                print("Trying to split the file and process in smaller chunks...")

                chunk_files = split_large_transcript_file(input_file, max_lines=5000)
                if not chunk_files:
                    print("No chunks created. Possibly input is empty or already small.")
                    return

                for chunk_file in chunk_files:
                    print(f"\nReprocessing chunk: {chunk_file}")
                    self.process_transcript(chunk_file)
                return
            
            # Save to output file
            print(f"Saving processed transcript to: {output_file}")
            print(f"Content preview (first 200 chars): {result[:200]}...")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(result)
            
            print(f"Processing complete! Output saved to: {output_file}")
            print(f"Output file size: {len(result)} characters")
            
        except Exception as e:
            print(f"Exception details: {e}")
            print(f"Exception type: {type(e)}")
            raise Exception(f"Error processing with Gemini model: {str(e)}")

def main():
    """Main function to run the script."""
    if len(sys.argv) != 2:
        print("Usage: python transcript_processor.py <input_file.json>")
        print("\nExample:")
        print("python transcript_processor.py transcript.json")
        print("Output will be saved as: transcript.txt")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    # Check if input file exists
    if not Path(input_file).exists():
        print(f"Error: Input file '{input_file}' does not exist.")
        sys.exit(1)
    
    try:
        # Initialize processor
        processor = ParliamentaryTranscriptProcessor()
        
        # Process the transcript
        processor.process_transcript(input_file)
        
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("\nTo set up Google API key:")
        print("1.Get an Api key from google ai studio")
        print("2. Set environment variable: export GOOGLE_API_KEY='your-api-key'")
        print("3. Or create a .env file with: GOOGLE_API_KEY=your-api-key")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()