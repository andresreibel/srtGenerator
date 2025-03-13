# Video to SRT Generator

A command-line tool that generates SRT subtitle files from video files using OpenAI's Whisper API. It can transcribe audio in its original language or translate it to a target language.

## Features

- Convert video files to audio using ffmpeg
- Transcribe audio in its original language
- Translate audio to any language supported by OpenAI's Whisper API
- Improved two-step translation process for non-English languages that preserves timestamps and subtitle structure
- Intelligent timestamp adjustment based on translated text length
- Automatically handle file paths in the correct directories
- Generate properly formatted SRT subtitle files

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
4. Set up your OpenAI API key as an environment variable:
   ```
   export OPENAI_API_KEY='your-api-key'
   ```
   Or add it to a `.env` file in the same directory.

## Directory Structure

- `input_video_files/`: Place your video files in this directory
- `output_srt_files/`: Generated SRT files will be saved here

## Usage

### Basic Transcription

To transcribe a video file in its original language:

```
python video_to_srt.py video.mp4 output_name
```

### Translation to English

To translate the audio to English:

```
python video_to_srt.py video.mp4 output_name --translate
```

### Translation to Other Languages

To translate the audio to a specific language (e.g., Spanish):

```
python video_to_srt.py video.mp4 output_name --translate --target-language es
```

### Specify Source Language

If you know the source language, you can specify it for better accuracy:

```
python video_to_srt.py video.mp4 output_name --language ja
```

### Timestamp Adjustment

To adjust subtitle timestamps based on translated text length:

```
python video_to_srt.py video.mp4 output_name --translate --target-language es --adjust-timestamps
```

## Translation Approaches

The script uses different approaches depending on the target language:

1. **English Translation**: Uses OpenAI's dedicated translations API endpoint, which directly translates all audio content to English.

2. **Non-English Translation**: Uses an improved two-step approach:
   - First transcribes the audio with timestamps to get an SRT file in the original language(s)
   - Parses the SRT file to extract subtitle entries with their timestamps
   - Translates the text of each subtitle entry to the target language using GPT-3.5 Turbo
   - Optionally adjusts timestamps based on translated text length
   - Reconstructs the SRT file with the original or adjusted timestamps and translated text

The improved two-step approach ensures that all content is preserved when translating to non-English languages, maintaining the original subtitle structure and timing.

## Timestamp Adjustment

The timestamp adjustment feature modifies subtitle durations based on the ratio between original and translated text lengths. This helps ensure that:

- Longer translations have more time on screen
- Shorter translations don't stay on screen too long
- Subtitles don't overlap
- A minimum gap (100ms) is maintained between consecutive subtitles

This feature is particularly useful for languages that are significantly longer or shorter than the source language when written.

## Supported Languages

The script supports all languages supported by OpenAI's Whisper model. Some common language codes:

- English: `en`
- Spanish: `es`
- French: `fr`
- German: `de`
- Japanese: `ja`
- Chinese: `zh`
- Russian: `ru`
- Portuguese: `pt`
- Italian: `it`

## Cost Considerations

- **Whisper API**: $0.006 per minute (or $0.36 per hour)
- **GPT-3.5 Turbo** (used for text translation): Very low cost (approximately $0.01-0.02 for an hour of speech)

For a 1-hour video:

- English translation: ~$0.36
- Non-English translation: ~$0.378 (slightly higher due to the additional text translation step)

## Notes

- The maximum audio file size supported by OpenAI's API is 25 MB
- The quality of transcription and translation depends on the audio quality and the Whisper model's capabilities
