import argparse
import os
import subprocess
import tempfile
import re
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def convert_video_to_audio(video_path, audio_output_path):
    """Convert video file to audio using ffmpeg."""
    try:
        subprocess.run(['ffmpeg', '-version'], check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        command = [
            'ffmpeg',
            '-i', video_path,
            '-vn',
            '-acodec', 'mp3',
            '-ab', '192k',
            '-y',
            audio_output_path
        ]
        subprocess.run(command, check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error converting video to audio: {e}")
        return False
    except FileNotFoundError:
        print("Error: ffmpeg is not installed. Please install ffmpeg first.")
        print("On Ubuntu: sudo apt-get install ffmpeg")
        print("On macOS: brew install ffmpeg")
        return False


def translate_text(client, text, target_language):
    """Translate text to the target language using GPT-3.5-turbo."""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"You are a professional translator. Translate the following text to {target_language}. Preserve the original meaning, tone, and style as much as possible. Only respond with the translated text, no explanations or additional text."},
                {"role": "user", "content": text}
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error translating text: {e}")
        return text


def parse_srt(srt_content):
    """Parse SRT content into a list of subtitle entries with original text preserved."""
    subtitle_blocks = re.split(r'\n\s*\n', srt_content.strip())
    subtitles = []
    for block in subtitle_blocks:
        if not block.strip():
            continue
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue
        subtitle_number = lines[0].strip()
        timestamp = lines[1].strip()
        text = '\n'.join(lines[2:]).strip()
        subtitles.append({
            'number': subtitle_number,
            'timestamp': timestamp,
            'text': text,
            'original_text': text  # Store original text for timestamp adjustment
        })
    return subtitles


def translate_subtitles(client, subtitles, target_language):
    """Translate the text of each subtitle entry."""
    all_text = '\n'.join([subtitle['text'] for subtitle in subtitles])
    translated_text = translate_text(client, all_text, target_language)
    translated_lines = translated_text.strip().split('\n')
    if len(translated_lines) != len(subtitles):
        print(
            f"Warning: Number of translated lines ({len(translated_lines)}) doesn't match original subtitles ({len(subtitles)})")
        print("Translating each subtitle individually...")
        for subtitle in subtitles:
            subtitle['text'] = translate_text(
                client, subtitle['text'], target_language)
    else:
        for i, line in enumerate(translated_lines):
            subtitles[i]['text'] = line
    return subtitles


def srt_time_to_ms(srt_time):
    """Convert SRT time format to milliseconds."""
    hours, minutes, seconds = srt_time.split(':')
    seconds, ms = seconds.split(',')
    total_ms = (int(hours) * 3600 + int(minutes) *
                60 + int(seconds)) * 1000 + int(ms)
    return total_ms


def ms_to_srt_time(ms):
    """Convert milliseconds to SRT time format."""
    hours = ms // 3600000
    ms %= 3600000
    minutes = ms // 60000
    ms %= 60000
    seconds = ms // 1000
    ms %= 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{ms:03d}"


def adjust_timestamps(subtitles):
    """Adjust subtitle timestamps based on translated text length to improve accuracy."""
    for i, subtitle in enumerate(subtitles):
        if 'original_text' not in subtitle or not subtitle['original_text']:
            continue
        original_length = len(subtitle['original_text'])
        translated_length = len(subtitle['text'])
        ratio = translated_length / original_length if original_length > 0 else 1
        start_str, end_str = subtitle['timestamp'].split(' --> ')
        start_ms = srt_time_to_ms(start_str)
        end_ms = srt_time_to_ms(end_str)
        original_duration = end_ms - start_ms
        new_duration = int(original_duration * ratio)
        if i < len(subtitles) - 1:
            next_start_str = subtitles[i + 1]['timestamp'].split(' --> ')[0]
            next_start_ms = srt_time_to_ms(next_start_str)
            max_end_ms = next_start_ms - 100  # Ensure 100ms gap
        else:
            max_end_ms = None
        new_end_ms = start_ms + new_duration
        if max_end_ms is not None and new_end_ms > max_end_ms:
            new_end_ms = max_end_ms
        subtitle['timestamp'] = f"{ms_to_srt_time(start_ms)} --> {ms_to_srt_time(new_end_ms)}"
    return subtitles


def format_srt(subtitles):
    """Format a list of subtitle entries back into SRT format."""
    srt_content = ""
    for subtitle in subtitles:
        srt_content += f"{subtitle['number']}\n{subtitle['timestamp']}\n{subtitle['text']}\n\n"
    return srt_content


def main():
    parser = argparse.ArgumentParser(
        description="Generate an SRT file from a video or audio file using OpenAI Whisper API.")
    parser.add_argument(
        "input", help="Path to the input file (e.g., video.mp4, audio.mp3, audio.wav)")
    parser.add_argument(
        "output_name", help="Name for the output SRT file (without extension, e.g., 'subtitles')")
    parser.add_argument("--translate", action="store_true",
                        help="Translate the audio to the specified target language")
    parser.add_argument(
        "--language", help="Language of the audio for transcription (e.g., 'en', 'es')")
    parser.add_argument("--target-language", default="en",
                        help="Target language for translation. Default is English")
    parser.add_argument("--adjust-timestamps", action="store_true",
                        help="Adjust timestamps based on translated text length")
    args = parser.parse_args()

    # Verify OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY_ANDRES") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: Please set the OPENAI_API_KEY or OPENAI_API_KEY_ANDRES environment variable.")
        exit(1)

    # Create input and output directories
    os.makedirs("input_video_files", exist_ok=True)
    os.makedirs("output_srt_files", exist_ok=True)

    # Determine input file path
    input_file = args.input
    if not os.path.isabs(input_file) and not input_file.startswith('./'):
        input_video_path = os.path.join("input_video_files", input_file)
        if os.path.exists(input_video_path):
            input_file = input_video_path

    # Determine output SRT file path
    target_lang_suffix = f"_{args.target_language}" if args.translate else ""
    timestamp_suffix = "_adjusted" if args.adjust_timestamps else ""
    srt_filename = f"{args.output_name}{target_lang_suffix}{timestamp_suffix}.srt"
    srt_path = os.path.join("output_srt_files", srt_filename)

    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        print("Make sure the file exists or place it in the 'input_video_files' directory.")
        exit(1)

    # Check if output SRT file already exists
    if os.path.exists(srt_path):
        overwrite = input(f"'{srt_path}' already exists. Overwrite? (y/n): ")
        if overwrite.lower() != "y":
            print("Operation cancelled.")
            exit(0)

    # Determine if input is video or audio
    video_extensions = ['.mp4', '.mkv', '.avi', '.mov']
    input_ext = os.path.splitext(input_file)[1].lower()
    audio_file_path = input_file

    if input_ext in video_extensions:
        print(f"Converting video '{input_file}' to audio...")
        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_audio:
            audio_file_path = tmp_audio.name
            if not convert_video_to_audio(input_file, audio_file_path):
                print("Failed to convert video to audio.")
                exit(1)
        print(f"Video converted to temporary audio file: {audio_file_path}")

    try:
        # Check file size (OpenAI limit is 25 MB)
        audio_size = os.path.getsize(audio_file_path)
        if audio_size > 25 * 1024 * 1024:
            print("Warning: Audio file exceeds 25 MB. The OpenAI API may reject it.")

        # Initialize OpenAI client
        client = OpenAI(api_key=api_key)

        if args.translate:
            target_lang = args.target_language
            if target_lang.lower() == "en":
                print("Translating audio directly to English...")
                with open(audio_file_path, "rb") as audio_file:
                    response = client.audio.translations.create(
                        model="whisper-1",
                        file=audio_file,
                        response_format="srt"
                    )
                print(f"Saving SRT file to {srt_path}...")
                with open(srt_path, "w") as srt_file:
                    srt_file.write(response)
            else:
                print(
                    f"Using two-step approach for translation to {target_lang}...")

                # Step 1: Get SRT with timestamps from original audio
                print("Step 1: Transcribing audio with timestamps...")
                with open(audio_file_path, "rb") as audio_file:
                    srt_with_timestamps = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=audio_file,
                        response_format="srt"
                    )

                # Step 2: Parse the SRT content
                print("Step 2: Parsing SRT content...")
                subtitles = parse_srt(srt_with_timestamps)

                # Step 3: Translate the subtitle text
                print(f"Step 3: Translating subtitles to {target_lang}...")
                translated_subtitles = translate_subtitles(
                    client, subtitles, target_lang)

                # Step 4: Adjust timestamps if requested
                if args.adjust_timestamps:
                    print(
                        "Step 4: Adjusting timestamps based on translated text length...")
                    translated_subtitles = adjust_timestamps(
                        translated_subtitles)

                # Step 5: Format back to SRT
                print("Step 5: Formatting translated content back to SRT...")
                final_srt = format_srt(translated_subtitles)

                # Save the final SRT file
                print(f"Saving SRT file to {srt_path}...")
                with open(srt_path, "w") as srt_file:
                    srt_file.write(final_srt)
        else:
            print(
                f"Transcribing audio{' in ' + args.language if args.language else ''}...")
            with open(audio_file_path, "rb") as audio_file:
                transcription_args = {
                    "model": "whisper-1",
                    "file": audio_file,
                    "response_format": "srt"
                }
                if args.language:
                    transcription_args["language"] = args.language
                response = client.audio.transcriptions.create(
                    **transcription_args)

            # Save the SRT file
            print(f"Saving SRT file to {srt_path}...")
            with open(srt_path, "w") as srt_file:
                srt_file.write(response)

        print(f"SRT file saved to '{srt_path}'.")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Clean up temporary audio file if it was created
        if audio_file_path != input_file and os.path.exists(audio_file_path):
            os.remove(audio_file_path)
            print(f"Cleaned up temporary file: {audio_file_path}")


if __name__ == "__main__":
    main()
