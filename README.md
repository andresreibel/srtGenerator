# Video to SRT Generator

A command-line tool that generates SRT subtitle files from video files using OpenAI's Whisper API. It can transcribe audio in its original language or translate it to a target language.

## Features

- Interactive CLI menu for easier usage
- Transcribe audio in its original language
- Translate audio to any language supported by OpenAI's Whisper API
- Support for absolute file paths and paths with special characters (including quoted paths)
- Intelligent timestamp adjustment based on translated text length
- Audio compression and chunking for large files
- Parallel processing for faster transcription
- Retry logic for API calls
- Subtitle validation

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
- Source language
- Target language
- Output directory

### Command Line Arguments

#### Basic Usage

```
python video_to_srt.py input_file output_name --language source_lang --target-language target_lang
```

#### Examples

Transcribe English audio to English subtitles:

```
python video_to_srt.py video.mp4 output_name --language en --target-language en
```

Transcribe English audio to Spanish subtitles:

```
python video_to_srt.py video.mp4 output_name --language en --target-language es
```

Transcribe Japanese audio to English subtitles:

```
python video_to_srt.py video.mp4 output_name --language ja --target-language en
```

#### Using Paths with Special Characters

You can use quotes for paths with special characters:

```
python video_to_srt.py '/path/to/your/video with #special characters.mp4' output_name --language en --target-language es
```

#### Custom Output Directory

```
python video_to_srt.py video.mp4 output_name --language en --target-language es --output-dir /path/to/custom/directory
```

## Handling Mixed Language Content

When working with audio that contains multiple languages:

1. **Identify the primary language** in your audio content
2. Specify this as the `--language` parameter (or select it in the interactive menu)
3. Choose your desired `--target-language` for the output subtitles

This approach ensures that Whisper will treat all speech as if it were in the specified source language, resulting in consistent transcription without language switching in the output.

Example for mixed English/Spanish audio where English is dominant:

```
python video_to_srt.py mixed_audio.mp4 output_name --language en --target-language es
```

## How It Works

### Translation Process

1. **Audio Extraction**: Converts video to audio using ffmpeg
2. **Transcription**: Uses OpenAI's Whisper API to transcribe audio in the source language
3. **Translation** (if needed): Translates text to the target language using GPT-3.5 Turbo
4. **Timestamp Adjustment**: Adjusts subtitle durations based on text length
5. **Validation**: Checks for overlapping subtitles and other issues

### Large File Handling

For files larger than 25 MB (OpenAI's limit):

1. Audio is compressed to 64k mono MP3
2. If still too large, audio is split into 10-minute chunks
3. Each chunk is processed in parallel
4. Results are merged with proper timestamp adjustments

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

- **Always specify the source language** for best results
- **For mixed language content, specify the primary/dominant language** to avoid inconsistent transcription
- The default source and target language is English if not specified
- The maximum audio file size supported by OpenAI's API is 25 MB (handled automatically)
- The quality of transcription depends on the audio quality and the Whisper model's capabilities

## Cost Considerations

- **Whisper API**: $0.006 per minute (or $0.36 per hour)
- **GPT-3.5 Turbo** (for translation): ~$0.01-0.02 for an hour of speech
