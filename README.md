# Video to SRT Generator

A command-line tool that generates SRT subtitle files from video files using OpenAI's Whisper API. It can transcribe audio in its original language or translate it to a target language.

## Features

- Convert video files to audio using ffmpeg
- Transcribe audio in its original language
- Translate audio to any language supported by OpenAI's Whisper API
- Improved two-step translation process for non-English languages that preserves timestamps and subtitle structure
- Intelligent timestamp adjustment based on translated text length (enabled by default)
- Support for absolute file paths and paths with special characters
- Custom output directory option
- Audio compression for files larger than 25 MB (enabled by default)
- Automatic chunking for very large files that exceed OpenAI's 25 MB limit
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
4. Set up your OpenAI API key:
   - Create a `.env` file in the same directory as the script (copy from `.env.example`)
   - Add your OpenAI API key to the `.env` file:
     ```
     OPENAI_API_KEY='your_actual_api_key_here'
     ```
   - You can get an API key from [OpenAI's platform](https://platform.openai.com/api-keys)
   - Alternatively, you can set it as an environment variable:
     ```
     export OPENAI_API_KEY='your_actual_api_key_here'
     ```

## Directory Structure

- `output_srt_files/`: Generated SRT files will be saved here by default

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

### Disable Timestamp Adjustment

Timestamp adjustment is enabled by default. To disable it:

```
python video_to_srt.py video.mp4 output_name --translate --target-language es --no-adjust-timestamps
```

### Using Absolute Paths

You can use absolute paths for input files:

```
python video_to_srt.py /path/to/your/video.mp4 output_name --translate
```

### Custom Output Directory

To specify a custom output directory for the SRT files:

```
python video_to_srt.py video.mp4 output_name --output-dir /path/to/custom/directory
```

### Handling Large Files

For files larger than 25 MB (OpenAI's limit), compression is applied automatically.
To disable automatic compression:

```
python video_to_srt.py large_video.mp4 output_name --no-compress
```

You can also force compression for smaller files:

```
python video_to_srt.py video.mp4 output_name --force-compress
```

For extremely large files that exceed 25 MB even after compression, the script will automatically split the audio into chunks, process each chunk separately, and then merge the results.

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

## Large File Handling

The script includes advanced features for handling large files:

### Automatic Compression

- Files larger than 25 MB are automatically compressed
- Uses mono audio with 64k bitrate for efficient size reduction
- Compression can be disabled with the `--no-compress` flag

### Automatic Chunking

For files that still exceed 25 MB after compression:

1. The audio is split into 10-minute chunks
2. Each chunk is processed separately with the OpenAI API
3. The resulting SRT files are merged with proper timestamp adjustments
4. This allows processing files of any size, regardless of OpenAI's 25 MB limit

This chunking process is completely automatic and requires no additional user input.

## Timestamp Adjustment

The timestamp adjustment feature (enabled by default) modifies subtitle durations based on the ratio between original and translated text lengths. This helps ensure that:

- Longer translations have more time on screen
- Shorter translations don't stay on screen too long
- Subtitles don't overlap
- A minimum gap (100ms) is maintained between consecutive subtitles

This feature is particularly useful for languages that are significantly longer or shorter than the source language when written.

## Audio Compression

The script includes audio compression features to handle files larger than OpenAI's 25 MB limit:

- For standard compression: Uses mono MP3 with 64k bitrate
- For forced compression: Uses mono MP3 with 64k bitrate
- Automatically applied to files larger than 25 MB
- Can reduce file sizes by 5-10x while maintaining good speech quality for transcription

The compression is optimized for voice content and maintains sufficient quality for accurate transcription.

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
- Compression is automatically applied to files larger than 25 MB
- Files that exceed 25 MB even after compression will be automatically chunked
- Timestamp adjustment is enabled by default for translated content
- The quality of transcription and translation depends on the audio quality and the Whisper model's capabilities
