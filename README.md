# Video to SRT Generator

A command-line tool that generates SRT subtitle files from video files using OpenAI's Whisper API. It automatically detects the source language and can translate to any target language.

## Features

- Interactive CLI menu for easier usage
- Automatic language detection for source audio
- Translation to any language supported by OpenAI's Whisper API
- Support for absolute file paths and paths with special characters (including quoted paths)
- Intelligent timestamp adjustment based on translated text length
- **Content-aware overlap resolution** to fix overlapping subtitles
- **Long subtitle splitting** for improved readability
- **Short subtitle extension** for better readability and YouTube compatibility
- Detailed console output of timestamp adjustments and fixes
- Audio compression and chunking for large files
- Parallel processing for faster transcription
- Retry logic for API calls
- Subtitle validation
- **Automatic removal of [inaudible] markers** for cleaner subtitles

## Prerequisites

- Python 3.7+
- ffmpeg (for video to audio conversion)
- OpenAI API key
- Required Python packages:
  ```
  pip install -r requirements.txt
  ```

## Installation

1. Clone this repository or download the script
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Install ffmpeg:
   - On macOS: `brew install ffmpeg`
   - On Ubuntu: `sudo apt-get install ffmpeg`
   - On Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html)
4. Set up your OpenAI API key:
   - Create a `.env` file in the same directory as the script
   - Add your OpenAI API key to the `.env` file:
     ```
     OPENAI_API_KEY='your_actual_api_key_here'
     ```
   - You can get an API key from [OpenAI's platform](https://platform.openai.com/api-keys)

## Usage

### Interactive CLI Menu (Recommended)

The easiest way to use the tool is with the interactive CLI menu:

```
python video_to_srt.py
```

This will guide you through the process with prompts for:

- Input file path
- Output name
- Target language

The output directory is automatically set to `output_srt_files` for simplicity.

### Command Line Arguments

#### Basic Usage

```
python video_to_srt.py input_file output_name --target-language target_lang
```

#### Examples

Transcribe audio to English subtitles:

```
python video_to_srt.py video.mp4 output_name --target-language en
```

Transcribe audio to Spanish subtitles:

```
python video_to_srt.py video.mp4 output_name --target-language es
```

#### Using Paths with Special Characters

You can use quotes for paths with special characters:

```
python video_to_srt.py '/path/to/your/video with #special characters.mp4' output_name --target-language es
```

#### Custom Output Directory

```
python video_to_srt.py video.mp4 output_name --target-language es --output-dir /path/to/custom/directory
```

## Handling Mixed Language Content

The tool now automatically detects the language of the audio content. For mixed language content:

1. The tool will automatically detect the dominant language
2. Choose your desired `--target-language` for the output subtitles
3. All content will be translated to the target language

This approach ensures consistent translation without language switching in the output.

Example for mixed language audio:

```
python video_to_srt.py mixed_audio.mp4 output_name --target-language es
```

## How It Works

### Translation Process

1. **Audio Extraction**: Converts video to audio using ffmpeg
2. **Transcription with Auto-detection**: Uses OpenAI's Whisper API to automatically detect and transcribe audio
3. **Translation**: Translates text to the target language using GPT-3.5 Turbo with strict output formatting
4. **Timestamp Adjustment**: Adjusts subtitle durations based on text length
5. **Long Subtitle Splitting**: Intelligently splits long subtitles at natural language boundaries
6. **Overlap Resolution**: Intelligently fixes overlapping subtitles based on content length and word count
7. **Short Subtitle Extension**: Extends subtitles that are too short for comfortable reading
8. **Validation**: Checks for overlapping subtitles and other issues

### Large File Handling

For files larger than 25 MB (OpenAI's limit):

1. Audio is compressed to 64k mono MP3
2. If still too large, audio is split into 10-minute chunks
3. Each chunk is processed in parallel
4. Results are merged with proper timestamp adjustments

### Overlap Resolution

The script includes a sophisticated algorithm to fix overlapping subtitles:

1. Detects overlaps between adjacent subtitles
2. Analyzes content length and word count to make intelligent adjustments
3. Prioritizes longer/more complex subtitles when resolving conflicts
4. Maintains minimum duration requirements for readability
5. Provides detailed console output of all changes made

### Long Subtitle Splitting

The script automatically splits long subtitles for improved readability:

1. Calculates a threshold based on mean subtitle length and standard deviation
2. Identifies subtitles exceeding this threshold
3. Splits long subtitles at natural language boundaries (sentences, clauses, or words)
4. Distributes timing proportionally based on text length
5. Maintains proper chronological order and subtitle numbering

This feature significantly improves subtitle readability by breaking long blocks of text into manageable segments while preserving the original meaning and timing.

### Short Subtitle Extension

The script intelligently extends subtitles that are too short for comfortable reading:

1. Calculates a minimum duration threshold based on mean duration and standard deviation (at least 500ms)
2. Identifies subtitles shorter than this threshold
3. Analyzes available space before and after each short subtitle
4. Intelligently extends duration by:
   - Shifting start time earlier when space is available
   - Extending end time when space is available
   - Distributing time adjustments proportionally when space exists in both directions
   - Removing empty subtitles to create more space when needed
5. Ensures all subtitles have sufficient duration for viewers to read

This feature significantly improves subtitle readability and YouTube compatibility by ensuring no subtitle appears too briefly on screen. YouTube often rejects SRT files with extremely short subtitles (under 500ms), and this feature automatically fixes such issues.

### Inaudible Marker Removal

The script automatically removes [inaudible] markers from subtitles:

1. Completely removes standalone [inaudible] subtitles (preserving empty timestamps)
2. Removes [inaudible] markers from within mixed content subtitles
3. Maintains all subtitle timing and numbering
4. Creates cleaner, more readable subtitles for viewers

This feature ensures that viewers aren't distracted by [inaudible] markers while watching videos, while still preserving the proper timing and structure of the subtitles.

## Supported Languages

The script supports all languages supported by OpenAI's Whisper model. Common language codes:

- English: `en`
- Spanish: `es`
- French: `fr`
- German: `de`
- Japanese: `ja`
- Chinese: `zh`
- Russian: `ru`
- Portuguese: `pt`
- Italian: `it`

## Important Notes

- **Language detection is automatic** - no need to specify the source language
- The tool will always translate to the specified target language
- The maximum audio file size supported by OpenAI's API is 25 MB (handled automatically)
- The quality of transcription depends on the audio quality and the Whisper model's capabilities
- Translation is strictly formatted to return only translations or audio cues in brackets
- Inaudible or noise sections are marked with [inaudible] during processing but removed in the final output

## Cost Considerations

- **Whisper API**: $0.006 per minute (or $0.36 per hour)
- **GPT-3.5 Turbo** (for translation): ~$0.01-0.02 for an hour of speech

## YouTube Integration

### Recommendation for YouTube Uploads

If your primary concern is making your content accessible to a global audience with minimal effort:

1. Use `video_to_srt.py` to generate a clean, consistent English SRT file (especially if your audio has mixed languages)
2. Upload this English SRT to YouTube
3. Let viewers use YouTube's auto-translate feature as needed

This approach balances your workload with accessibility, while still addressing concerns about mixed language content in the original SRT file.

### How YouTube Translation Works for Viewers

When you upload an English SRT file to YouTube:

- Viewers can click on the settings (gear icon) in the player
- Select "Subtitles/CC"
- Choose "Auto-translate"
- Select their preferred language from the list

This provides accessibility in all languages YouTube supports without requiring you to create multiple SRT files.

## Logging and Debugging

The script provides detailed console output to help troubleshoot any issues:

- **Timestamp Fixes**: Displays all adjustments made to fix overlapping subtitles
- **Subtitle Splitting**: Shows statistics on long subtitles that were split
- **Console Output**: Provides real-time progress updates and statistics on overlaps fixed and [inaudible] markers removed

These logs are particularly useful for identifying and resolving issues with subtitle timing, overlaps, and readability.

## License

This project is licensed under the MIT License - see below for details:

```
MIT License

Copyright (c) 2024

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
