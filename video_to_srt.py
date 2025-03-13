import argparse
import os
import subprocess
import tempfile
import re
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def convert_to_audio(input_path, output_path, bitrate="128k"):
    """Convert video or audio to mono MP3 with specified bitrate using ffmpeg."""
    try:
        subprocess.run(['ffmpeg', '-version'], check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        command = [
            'ffmpeg',
            '-i', input_path,
            '-vn',  # No video
            '-ac', '1',  # Mono
            '-acodec', 'mp3',
            '-ab', bitrate,
            '-y',
            output_path
        ]
        subprocess.run(command, check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error converting to audio: {e}")
        return False
    except FileNotFoundError:
        print("Error: ffmpeg is not installed. Please install ffmpeg first.")
        print("On Ubuntu: sudo apt-get install ffmpeg")
        print("On macOS: brew install ffmpeg")
        return False


def get_duration(input_path):
    """Get the duration of an audio or video file in seconds using ffmpeg."""
    try:
        result = subprocess.run(
            ['ffmpeg', '-i', input_path], stderr=subprocess.PIPE, text=True)
        duration_line = [line for line in result.stderr.split(
            '\n') if 'Duration' in line][0]
        duration = duration_line.split(',')[0].split(' ')[1]
        h, m, s = duration.split(':')
        total_seconds = int(h) * 3600 + int(m) * 60 + float(s)
        return total_seconds
    except Exception as e:
        print(f"Error getting duration: {e}")
        return None


def chunk_audio(input_path, output_dir, chunk_duration=600):
    """Split audio into chunks of specified duration (in seconds)."""
    duration = get_duration(input_path)
    if duration is None:
        return []

    chunk_paths = []
    num_chunks = int(duration // chunk_duration) + \
        (1 if duration % chunk_duration else 0)
    for i in range(num_chunks):
        start_time = i * chunk_duration
        output_path = os.path.join(output_dir, f"chunk_{i}.mp3")
        command = [
            'ffmpeg',
            '-i', input_path,
            '-ss', str(start_time),
            '-t', str(chunk_duration),
            '-ac', '1',
            '-acodec', 'mp3',
            '-ab', '64k',
            '-y',
            output_path
        ]
        subprocess.run(command, check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        chunk_paths.append(output_path)
    return chunk_paths


def merge_srt_files(srt_contents, time_offsets):
    """Merge multiple SRT files with adjusted timestamps."""
    merged_srt = ""
    subtitle_counter = 1
    for srt_content, offset in zip(srt_contents, time_offsets):
        subtitles = parse_srt(srt_content)
        for subtitle in subtitles:
            start_str, end_str = subtitle['timestamp'].split(' --> ')
            start_ms = srt_time_to_ms(start_str) + offset * 1000
            end_ms = srt_time_to_ms(end_str) + offset * 1000
            subtitle['number'] = str(subtitle_counter)
            subtitle['timestamp'] = f"{ms_to_srt_time(start_ms)} --> {ms_to_srt_time(end_ms)}"
            merged_srt += f"{subtitle['number']}\n{subtitle['timestamp']}\n{subtitle['text']}\n\n"
            subtitle_counter += 1
    return merged_srt


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
            'original_text': text
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
    """Adjust subtitle timestamps based on translated text length."""
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
            max_end_ms = next_start_ms - 100  # 100ms gap
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
        description="Generate an SRT file from a video or audio file using OpenAI Whisper API with compression and chunking.")
    parser.add_argument(
        "input", help="Path to the input file (e.g., video.mp4, audio.mp3)")
    parser.add_argument(
        "output_name", help="Name for the output SRT file (without extension)")
    parser.add_argument("--translate", action="store_true",
                        help="Translate audio to target language")
    parser.add_argument(
        "--language", help="Language of the audio for transcription (e.g., 'en', 'es')")
    parser.add_argument("--target-language", default="en",
                        help="Target language for translation (default: English)")
    parser.add_argument("--no-adjust-timestamps", action="store_false", dest="adjust_timestamps",
                        help="Disable timestamp adjustment based on translated text length")
    parser.add_argument(
        "--output-dir", help="Custom output directory for SRT files (default: output_srt_files)")
    parser.add_argument("--no-compress", action="store_true",
                        help="Disable automatic compression for large files")
    parser.add_argument("--force-compress", action="store_true",
                        help="Force compression for all files")
    parser.set_defaults(adjust_timestamps=True)
    args = parser.parse_args()

    # Verify OpenAI API key
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: Please set the OPENAI_API_KEY environment variable.")
        exit(1)

    # Create output directory
    output_dir = args.output_dir if args.output_dir else "output_srt_files"
    os.makedirs(output_dir, exist_ok=True)

    input_file = args.input
    target_lang_suffix = f"_{args.target_language}" if args.translate else ""
    timestamp_suffix = "" if not args.adjust_timestamps else "_adjusted"
    srt_filename = f"{args.output_name}{target_lang_suffix}{timestamp_suffix}.srt"
    srt_path = os.path.join(output_dir, srt_filename)

    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        exit(1)

    if os.path.exists(srt_path):
        overwrite = input(f"'{srt_path}' already exists. Overwrite? (y/n): ")
        if overwrite.lower() != "y":
            print("Operation cancelled.")
            exit(0)

    # Determine input type
    video_extensions = ['.mp4', '.mkv', '.avi', '.mov', '.MOV']
    audio_extensions = ['.mp3', '.wav', '.m4a', '.aac', '.flac', '.ogg']
    input_ext = os.path.splitext(input_file)[1].lower()

    # Prepare audio file
    temp_files = []
    audio_file_path = None
    try:
        if input_ext in video_extensions:
            bitrate = "64k" if args.force_compress or not args.no_compress else "128k"
            print(f"Converting video to audio with bitrate {bitrate}...")
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_audio:
                audio_file_path = tmp_audio.name
                if not convert_to_audio(input_file, audio_file_path, bitrate=bitrate):
                    print("Failed to convert video to audio.")
                    exit(1)
                temp_files.append(audio_file_path)
        elif input_ext in audio_extensions:
            file_size = os.path.getsize(input_file) / (1024 * 1024)
            print(f"Audio file size: {file_size:.2f} MB")
            if args.force_compress or (not args.no_compress and file_size > 25):
                bitrate = "64k"
                print(f"Compressing audio file to bitrate {bitrate}...")
                with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp_audio:
                    audio_file_path = tmp_audio.name
                    if not convert_to_audio(input_file, audio_file_path, bitrate=bitrate):
                        print("Failed to compress audio.")
                        exit(1)
                    temp_files.append(audio_file_path)
                    compressed_size = os.path.getsize(
                        audio_file_path) / (1024 * 1024)
                    print(
                        f"Compressed audio to {compressed_size:.2f} MB (reduction: {(1 - compressed_size/file_size)*100:.1f}%)")
            else:
                audio_file_path = input_file
        else:
            print(f"Unsupported file format: {input_ext}")
            exit(1)

        # Check if audio file needs chunking
        audio_size = os.path.getsize(audio_file_path) / (1024 * 1024)
        print(f"Prepared audio size: {audio_size:.2f} MB")
        client = OpenAI(api_key=api_key)

        if audio_size > 25:
            print("Audio exceeds 25 MB after compression. Splitting into chunks...")
            with tempfile.TemporaryDirectory() as tmp_dir:
                chunk_paths = chunk_audio(audio_file_path, tmp_dir)
                if not chunk_paths:
                    print("Failed to chunk audio.")
                    exit(1)
                temp_files.extend(chunk_paths)
                print(f"Created {len(chunk_paths)} chunks")

                srt_contents = []
                time_offsets = []
                for i, chunk_path in enumerate(chunk_paths):
                    chunk_size = os.path.getsize(chunk_path) / (1024 * 1024)
                    print(
                        f"Processing chunk {i+1}/{len(chunk_paths)} ({chunk_size:.2f} MB)...")
                    if chunk_size > 25:
                        print(
                            f"Warning: Chunk {i+1} still exceeds 25 MB ({chunk_size:.2f} MB).")
                        should_continue = input("Continue anyway? (y/n): ")
                        if should_continue.lower() != "y":
                            print("Operation cancelled.")
                            exit(0)
                    with open(chunk_path, "rb") as chunk_file:
                        if args.translate and args.target_language.lower() == "en":
                            response = client.audio.translations.create(
                                model="whisper-1",
                                file=chunk_file,
                                response_format="srt"
                            )
                        else:
                            transcription_args = {
                                "model": "whisper-1",
                                "file": chunk_file,
                                "response_format": "srt"
                            }
                            if args.language:
                                transcription_args["language"] = args.language
                            response = client.audio.transcriptions.create(
                                **transcription_args)
                        srt_contents.append(response)
                        time_offsets.append(i * 600)  # 10-minute chunks

                # Merge SRT files
                final_srt = merge_srt_files(srt_contents, time_offsets)
                if args.translate and args.target_language.lower() != "en":
                    subtitles = parse_srt(final_srt)
                    translated_subtitles = translate_subtitles(
                        client, subtitles, args.target_language)
                    if args.adjust_timestamps:
                        translated_subtitles = adjust_timestamps(
                            translated_subtitles)
                    final_srt = format_srt(translated_subtitles)

                with open(srt_path, "w") as srt_file:
                    srt_file.write(final_srt)
                print(f"SRT file saved to '{srt_path}'.")
        else:
            print("Audio is under 25 MB. Processing directly...")
            with open(audio_file_path, "rb") as audio_file:
                if args.translate:
                    if args.target_language.lower() == "en":
                        response = client.audio.translations.create(
                            model="whisper-1",
                            file=audio_file,
                            response_format="srt"
                        )
                    else:
                        srt_with_timestamps = client.audio.transcriptions.create(
                            model="whisper-1",
                            file=audio_file,
                            response_format="srt",
                            language=args.language if args.language else None
                        )
                        subtitles = parse_srt(srt_with_timestamps)
                        translated_subtitles = translate_subtitles(
                            client, subtitles, args.target_language)
                        if args.adjust_timestamps:
                            translated_subtitles = adjust_timestamps(
                                translated_subtitles)
                        response = format_srt(translated_subtitles)
                else:
                    transcription_args = {
                        "model": "whisper-1",
                        "file": audio_file,
                        "response_format": "srt"
                    }
                    if args.language:
                        transcription_args["language"] = args.language
                    response = client.audio.transcriptions.create(
                        **transcription_args)
                with open(srt_path, "w") as srt_file:
                    srt_file.write(response)
                print(f"SRT file saved to '{srt_path}'.")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                print(f"Cleaned up temporary file: {temp_file}")


if __name__ == "__main__":
    main()
