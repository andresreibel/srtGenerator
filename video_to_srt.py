import argparse
import os
import subprocess
import tempfile
import re
import time
import sys
from openai import OpenAI
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
from concurrent.futures import ThreadPoolExecutor

# Load environment variables from .env file
load_dotenv()

# Common language codes for reference
COMMON_LANGUAGES = {
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'it': 'Italian',
    'pt': 'Portuguese',
    'ru': 'Russian',
    'zh': 'Chinese',
    'ja': 'Japanese',
    'ko': 'Korean',
    'ar': 'Arabic',
    'hi': 'Hindi'
}


def convert_to_audio(input_path, output_path, bitrate="128k"):
    """Convert video or audio to mono MP3 with specified bitrate using ffmpeg."""
    print(f"Converting to audio with bitrate {bitrate}...")
    command = [
        'ffmpeg', '-i', input_path, '-vn', '-ac', '1', '-acodec', 'mp3', '-ab', bitrate, '-y', output_path
    ]
    try:
        subprocess.run(command, check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        audio_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"Audio conversion complete: {audio_size:.2f} MB")
        return True
    except subprocess.CalledProcessError as e:
        print(f"FFmpeg error: {e.stderr.decode('utf-8', errors='replace')}")
        return False


def get_duration(input_path):
    """Get the duration of an audio or video file in seconds using ffprobe."""
    print("Detecting file duration...")
    command = [
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', input_path
    ]
    try:
        result = subprocess.run(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        duration = float(result.stdout.strip()
                         ) if result.stdout.strip() else None
        if duration:
            print(f"Duration detected: {duration:.2f} seconds")
        return duration
    except Exception as e:
        print(f"Error getting duration: {e}")
        return None


def chunk_audio(input_path, output_dir, chunk_duration=600, overlap=5):
    """Split audio into chunks with overlap to avoid splitting sentences."""
    print(f"Chunking audio with {overlap}s overlap...")
    duration = get_duration(input_path)
    if not duration:
        return []

    chunk_paths = []
    start_time = 0
    chunk_index = 0
    total_chunks = int((duration - overlap) // (chunk_duration - overlap)) + 1
    print(f"Will create approximately {total_chunks} chunks")

    while start_time < duration:
        output_path = os.path.join(output_dir, f"chunk_{chunk_index}.mp3")
        extract_duration = min(chunk_duration + overlap, duration - start_time)
        print(
            f"Creating chunk {chunk_index+1}/{total_chunks}: {start_time:.1f}s to {start_time+extract_duration:.1f}s")
        command = [
            'ffmpeg', '-i', input_path, '-ss', str(
                start_time), '-t', str(extract_duration),
            '-ac', '1', '-acodec', 'mp3', '-ab', '64k', '-y', output_path
        ]
        try:
            subprocess.run(command, check=True,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            chunk_paths.append((output_path, start_time))
            chunk_index += 1
        except subprocess.CalledProcessError as e:
            print(
                f"Error creating chunk {chunk_index}: {e.stderr.decode('utf-8', errors='replace')}")
            break
        start_time += chunk_duration - overlap

    print(f"Created {len(chunk_paths)} chunks successfully")
    return chunk_paths


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def transcribe_audio(client, audio_file, language=None):
    """Transcribe audio using OpenAI Whisper API with retry logic."""
    transcription_args = {"model": "whisper-1",
                          "file": audio_file, "response_format": "srt"}
    if language:
        transcription_args["language"] = language
    return client.audio.transcriptions.create(**transcription_args)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def translate_text(client, text, target_language):
    """Translate text to the target language using GPT-3.5-turbo with retry logic."""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": f"Translate the following text to {target_language}. Preserve the original meaning and structure."},
            {"role": "user", "content": text}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content.strip()


def parse_srt(srt_content):
    """Parse SRT content into a list of subtitle entries."""
    subtitles = []
    for block in re.split(r'\n\s*\n', srt_content.strip()):
        if not block.strip():
            continue
        lines = block.strip().split('\n')
        if len(lines) < 3:
            continue
        subtitles.append({
            'number': lines[0].strip(),
            'timestamp': lines[1].strip(),
            'text': '\n'.join(lines[2:]).strip(),
            'original_text': '\n'.join(lines[2:]).strip()
        })
    return subtitles


def translate_subtitle(client, subtitle, target_language):
    """Translate a single subtitle, used for parallel processing."""
    try:
        subtitle['text'] = translate_text(
            client, subtitle['text'], target_language)
    except Exception as e:
        print(
            f"Error translating subtitle {subtitle['number']}: {e}. Using original text.")
        subtitle['text'] = subtitle['original_text']
    return subtitle


def translate_subtitles(client, subtitles, target_language):
    """Translate subtitles in parallel."""
    print(f"Translating {len(subtitles)} subtitles to {target_language}...")
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(
            translate_subtitle, client, subtitle, target_language) for subtitle in subtitles]
        subtitles = [future.result() for future in futures]
    print(f"Translation completed in {time.time() - start_time:.1f} seconds")
    return subtitles


def srt_time_to_ms(srt_time):
    """Convert SRT time format to milliseconds."""
    h, m, s = srt_time.split(':')
    s, ms = s.split(',')
    return (int(h) * 3600 + int(m) * 60 + int(s)) * 1000 + int(ms)


def ms_to_srt_time(ms):
    """Convert milliseconds to SRT time format."""
    h = ms // 3600000
    ms %= 3600000
    m = ms // 60000
    ms %= 60000
    s = ms // 1000
    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def adjust_timestamps(subtitles, cps=17, min_time=1.0, gap=0.1):
    """Adjust subtitle timestamps based on translated text length."""
    print(f"Adjusting timestamps for {len(subtitles)} subtitles...")
    adjusted_count = 0
    for i, subtitle in enumerate(subtitles):
        start_str, end_str = subtitle['timestamp'].split(' --> ')
        start_ms = srt_time_to_ms(start_str)
        end_ms = srt_time_to_ms(end_str)
        current_duration = (end_ms - start_ms) / 1000.0  # in seconds
        translated_text = subtitle['text']
        char_count = len(translated_text)
        required_time = max(min_time, char_count / cps)
        desired_end_ms = start_ms + int(required_time * 1000)
        if i < len(subtitles) - 1:
            next_start_str = subtitles[i + 1]['timestamp'].split(' --> ')[0]
            next_start_ms = srt_time_to_ms(next_start_str)
            max_end_ms = next_start_ms - int(gap * 1000)
        else:
            max_end_ms = desired_end_ms
        new_end_ms = min(max(end_ms, desired_end_ms), max_end_ms)
        if new_end_ms != end_ms:
            adjusted_count += 1
        if new_end_ms <= start_ms:
            new_end_ms = start_ms + 100  # Minimum 0.1 second
        new_end_str = ms_to_srt_time(new_end_ms)
        subtitle['timestamp'] = f"{start_str} --> {new_end_str}"
    print(f"Adjusted {adjusted_count} subtitle timestamps")
    return subtitles


def subtitles_to_srt(subtitles):
    """Convert subtitle list to SRT string."""
    srt_content = ""
    for subtitle in subtitles:
        srt_content += f"{subtitle['number']}\n{subtitle['timestamp']}\n{subtitle['text']}\n\n"
    return srt_content


def merge_srt_files(srt_contents, time_offsets):
    """Merge SRT files with timestamp adjustments and duplicate removal."""
    print("Merging subtitle chunks...")
    all_subtitles = []
    for srt_content, offset in zip(srt_contents, time_offsets):
        subtitles = parse_srt(srt_content)
        for subtitle in subtitles:
            start_str, end_str = subtitle['timestamp'].split(' --> ')
            start_ms = srt_time_to_ms(start_str) + int(offset * 1000)
            end_ms = srt_time_to_ms(end_str) + int(offset * 1000)
            subtitle['timestamp'] = f"{ms_to_srt_time(start_ms)} --> {ms_to_srt_time(end_ms)}"
            all_subtitles.append(subtitle)

    print(f"Total subtitles before deduplication: {len(all_subtitles)}")
    all_subtitles.sort(key=lambda x: srt_time_to_ms(
        x['timestamp'].split(' --> ')[0]))
    unique_subtitles = []
    seen = set()
    for subtitle in all_subtitles:
        key = (subtitle['timestamp'], subtitle['text'])
        if key not in seen:
            seen.add(key)
            unique_subtitles.append(subtitle)
    for i, subtitle in enumerate(unique_subtitles, 1):
        subtitle['number'] = str(i)
    print(f"Unique subtitles after deduplication: {len(unique_subtitles)}")
    return unique_subtitles


def validate_srt(srt_content):
    """Validate SRT for sequential timestamps and no overlaps."""
    print("Validating subtitles...")
    subtitles = parse_srt(srt_content)
    overlap_count = 0
    for i in range(len(subtitles) - 1):
        current_end = srt_time_to_ms(
            subtitles[i]['timestamp'].split(' --> ')[1])
        next_start = srt_time_to_ms(
            subtitles[i + 1]['timestamp'].split(' --> ')[0])
        if current_end > next_start:
            overlap_count += 1
            print(
                f"Warning: Overlapping subtitles at {subtitles[i]['number']} and {subtitles[i+1]['number']}")
    print(f"Validation complete: {overlap_count} overlapping subtitles found")


def transcribe_chunk(client, chunk_file, language):
    """Helper function for parallel transcription."""
    chunk_name = os.path.basename(chunk_file)
    print(f"Transcribing {chunk_name}...")
    try:
        with open(chunk_file, "rb") as f:
            result = transcribe_audio(client, f, language=language)
            print(f"Completed transcription of {chunk_name}")
            return result
    except Exception as e:
        print(
            f"Warning: Failed to transcribe {chunk_name}: {e}. Skipping.")
        return ""


def display_menu():
    """Display the interactive CLI menu."""
    print("\n" + "="*60)
    print("🎬 SRT Generator - Interactive Menu 🎬".center(60))
    print("="*60)
    print("\nThis tool generates SRT subtitle files from video/audio files.")
    print("All processing features are enabled by default:")
    print("  ✅ Chunking for large files")
    print("  ✅ Parallel processing")
    print("  ✅ Retry logic for API calls")
    print("  ✅ Timestamp adjustment")
    print("  ✅ Subtitle validation")

    # Get input file
    while True:
        input_file = input(
            "\n📁 Enter the path to your video/audio file: ").strip()
        if not input_file:
            print("❌ Input file path cannot be empty.")
            continue

        # Handle paths with quotes
        if (input_file.startswith("'") and input_file.endswith("'")) or \
           (input_file.startswith('"') and input_file.endswith('"')):
            input_file = input_file[1:-1]  # Remove surrounding quotes

        if not os.path.exists(input_file):
            print(f"❌ File not found: '{input_file}'")
            continue
        break

    # Get output name
    while True:
        output_name = input(
            "\n💾 Enter the output name (without extension): ").strip()
        if not output_name:
            print("❌ Output name cannot be empty.")
            continue
        break

    # Get source language with improved guidance
    print("\n🔤 Source language (language of the audio):")
    print("   This is the PRIMARY language spoken in your audio.")
    print("   ⚠️  IMPORTANT: For mixed language content, specify the dominant language.")
    print("   This ensures consistent transcription without language switching.")
    print("   Common codes: en (English), es (Spanish), fr (French), etc.")
    source_language = input(
        "   Enter source language code [default: en]: ").strip().lower() or "en"
    if source_language in COMMON_LANGUAGES:
        print(f"   Selected: {COMMON_LANGUAGES[source_language]}")

    # Get target language with improved guidance
    print("\n🌐 Target language (language for the subtitles):")
    print("   This is the language your subtitles will be translated to.")
    print("   If same as source language, no translation will be performed.")
    print("   Common codes: en (English), es (Spanish), fr (French), etc.")
    target_language = input(
        "   Enter target language code [default: en]: ").strip().lower() or "en"
    if target_language in COMMON_LANGUAGES:
        print(f"   Selected: {COMMON_LANGUAGES[target_language]}")

    # Add warning if source and target are different
    if source_language != target_language:
        print(
            f"\n📝 NOTE: Audio will be transcribed from {COMMON_LANGUAGES.get(source_language, source_language)} and translated to {COMMON_LANGUAGES.get(target_language, target_language)}.")
    else:
        print(
            f"\n📝 NOTE: Audio will be transcribed in {COMMON_LANGUAGES.get(source_language, source_language)} with no translation.")

    # Get output directory
    output_dir = input("\n📂 Enter output directory [default: output_srt_files]: ").strip(
    ) or "output_srt_files"

    # Handle paths with quotes for output directory
    if (output_dir.startswith("'") and output_dir.endswith("'")) or \
       (output_dir.startswith('"') and output_dir.endswith('"')):
        output_dir = output_dir[1:-1]  # Remove surrounding quotes

    # Confirm settings
    print("\n" + "-"*60)
    print("📋 Summary of settings:")
    print(f"   Input file: {input_file}")
    print(f"   Output name: {output_name}")
    print(f"   Source language: {source_language}")
    print(f"   Target language: {target_language}")
    print(f"   Output directory: {output_dir}")
    print("-"*60)

    confirm = input("\n✅ Proceed with these settings? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Operation cancelled.")
        return None

    return {
        'input': input_file,
        'output_name': output_name,
        'language': source_language,
        'target_language': target_language,
        'output_dir': output_dir
    }


def process_with_args(args):
    """Process video/audio with the given arguments."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: Set OPENAI_API_KEY in .env file.")
        exit(1)

    os.makedirs(args.output_dir, exist_ok=True)
    input_file = args.input
    srt_path = os.path.join(
        args.output_dir, f"{args.output_name}_{args.target_language}.srt")

    if not os.path.exists(input_file):
        print(f"Error: '{input_file}' not found.")
        exit(1)

    temp_files = []
    audio_file_path = None
    total_start_time = time.time()

    try:
        print(f"Processing file: {input_file}")
        client = OpenAI(api_key=api_key)
        duration = get_duration(input_file)
        if duration is None:
            print("Failed to get duration")
            exit(1)

        with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as tmp:
            audio_file_path = tmp.name
            if not convert_to_audio(input_file, audio_file_path, "64k"):
                exit(1)
            temp_files.append(audio_file_path)

        audio_size = os.path.getsize(audio_file_path) / (1024 * 1024)
        need_translation = not (
            args.language and args.language.lower() == args.target_language.lower())
        if need_translation:
            print(f"Translating to {args.target_language}.")
        else:
            print("Source and target languages match. Skipping translation.")

        if audio_size > 25:
            print(f"Audio size: {audio_size:.2f} MB. Chunking required.")
            with tempfile.TemporaryDirectory() as tmp_dir:
                chunk_info = chunk_audio(audio_file_path, tmp_dir)
                if not chunk_info:
                    print("Chunking failed.")
                    exit(1)
                chunk_files, time_offsets = zip(*chunk_info)

                print(
                    f"Starting parallel transcription of {len(chunk_files)} chunks...")
                transcription_start = time.time()
                with ThreadPoolExecutor(max_workers=5) as executor:
                    srt_contents = list(executor.map(
                        lambda cf: transcribe_chunk(client, cf, args.language), chunk_files))
                print(
                    f"Transcription completed in {time.time() - transcription_start:.1f} seconds")

                subtitles = merge_srt_files(srt_contents, time_offsets)
                if need_translation:
                    subtitles = translate_subtitles(
                        client, subtitles, args.target_language)
                subtitles = adjust_timestamps(subtitles)
                final_srt = subtitles_to_srt(subtitles)
                validate_srt(final_srt)
                with open(srt_path, "w") as srt_file:
                    srt_file.write(final_srt)
                print(f"SRT saved to '{srt_path}'.")
        else:
            print(f"Audio size: {audio_size:.2f} MB. Processing directly.")
            transcription_start = time.time()
            with open(audio_file_path, "rb") as f:
                print("Transcribing audio...")
                response = transcribe_audio(client, f, language=args.language)
                print(
                    f"Transcription completed in {time.time() - transcription_start:.1f} seconds")
                subtitles = parse_srt(response)
                print(f"Parsed {len(subtitles)} subtitles")
                if need_translation:
                    subtitles = translate_subtitles(
                        client, subtitles, args.target_language)
                subtitles = adjust_timestamps(subtitles)
                final_srt = subtitles_to_srt(subtitles)
                validate_srt(final_srt)
                with open(srt_path, "w") as srt_file:
                    srt_file.write(final_srt)
                print(f"SRT saved to '{srt_path}'.")

        total_time = time.time() - total_start_time
        print(
            f"Total processing time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")

    except Exception as e:
        print(f"Error: {e}")
    finally:
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                os.remove(temp_file)
                print(f"Cleaned up temporary file: {temp_file}")


def main():
    """Main function with support for both CLI arguments and interactive menu."""
    print("Starting video_to_srt.py...")

    # Check if any command line arguments were provided
    if len(sys.argv) > 1:
        # Use argparse for backward compatibility
        parser = argparse.ArgumentParser(
            description="Generate consistent SRT files with non-English translations.")
        parser.add_argument(
            "input", help="Input file path (e.g., video.mp4, audio.mp3)")
        parser.add_argument(
            "output_name", help="Output SRT file name (without extension)")
        parser.add_argument(
            "--language", default="en",
            help="Source language of the audio (e.g., 'en', 'es'). For mixed language content, specify the dominant language.")
        parser.add_argument("--target-language", default="en",
                            help="Target language for the subtitles (e.g., 'fr', 'es')")
        parser.add_argument(
            "--output-dir", default="output_srt_files", help="Output directory")
        parser.add_argument("--interactive", action="store_true",
                            help="Use interactive menu instead of command line arguments")

        args = parser.parse_args()

        # Handle quoted paths in command-line arguments
        if args.input and (args.input.startswith("'") and args.input.endswith("'")) or \
           (args.input.startswith('"') and args.input.endswith('"')):
            args.input = args.input[1:-1]  # Remove surrounding quotes

        if args.output_dir and (args.output_dir.startswith("'") and args.output_dir.endswith("'")) or \
           (args.output_dir.startswith('"') and args.output_dir.endswith('"')):
            # Remove surrounding quotes
            args.output_dir = args.output_dir[1:-1]

        # Display language information
        if args.language != args.target_language:
            if args.language in COMMON_LANGUAGES and args.target_language in COMMON_LANGUAGES:
                print(
                    f"\n📝 NOTE: Audio will be transcribed from {COMMON_LANGUAGES[args.language]} and translated to {COMMON_LANGUAGES[args.target_language]}.")
            else:
                print(
                    f"\n📝 NOTE: Audio will be transcribed from '{args.language}' and translated to '{args.target_language}'.")
        else:
            if args.language in COMMON_LANGUAGES:
                print(
                    f"\n📝 NOTE: Audio will be transcribed in {COMMON_LANGUAGES[args.language]} with no translation.")
            else:
                print(
                    f"\n📝 NOTE: Audio will be transcribed in '{args.language}' with no translation.")

        if args.interactive:
            menu_args = display_menu()
            if menu_args:
                # Convert dict to argparse.Namespace
                args = argparse.Namespace(**menu_args)
                process_with_args(args)
        else:
            process_with_args(args)
    else:
        # No command line args, use interactive menu
        menu_args = display_menu()
        if menu_args:
            # Convert dict to argparse.Namespace
            args = argparse.Namespace(**menu_args)
            process_with_args(args)


if __name__ == "__main__":
    main()
