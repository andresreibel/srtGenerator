# Video to SRT Generator

A command-line tool that generates SRT subtitle files from video files using OpenAI's Whisper API. It automatically detects the source language and can translate to any target language.

## Features

- Interactive CLI menu for easier usage
- Automatic language detection for source audio
- Translation to any language supported by OpenAI's Whisper API
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
- Target language
- Output directory

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
3. **Translation**: Translates text to the target language using GPT-3.5 Turbo
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

- **Language detection is automatic** - no need to specify the source language
- The tool will always translate to the specified target language
- The maximum audio file size supported by OpenAI's API is 25 MB (handled automatically)
- The quality of transcription depends on the audio quality and the Whisper model's capabilities

## Cost Considerations

- **Whisper API**: $0.006 per minute (or $0.36 per hour)
- **GPT-3.5 Turbo** (for translation): ~$0.01-0.02 for an hour of speech

## YouTube Integration

### Recommendation for YouTube Uploads

If your primary concern is making your content accessible to a global audience with minimal effort:

1. Use `video_new.py` to generate a clean, consistent English SRT file (especially if your audio has mixed languages)
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
