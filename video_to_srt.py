import argparse
import os
import subprocess
import tempfile
import re
import time
import sys
import statistics
from openai import OpenAI
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential
from concurrent.futures import ThreadPoolExecutor, as_completed

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
    try:
        transcription_args = {"model": "whisper-1",
                              "file": audio_file, "response_format": "srt"}
        if language:
            transcription_args["language"] = language
        else:
            print("Using automatic language detection...")

        response = client.audio.transcriptions.create(**transcription_args)
        return response
    except Exception as e:
        print(f"Transcription error: {str(e)[:100]}. Retrying...")
        raise  # Re-raise for retry decorator to handle


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def translate_text(client, text, target_language):
    """Translate text to the target language using GPT-3.5-turbo with retry logic."""
    try:
        # Skip empty text
        if not text or len(text.strip()) == 0:
            return text

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system",
                    "content": f"Translate the following text to {target_language}. Preserve the original meaning and structure. Only translate meaningful speech, keep audio cues like [laughter] or [music] in brackets. If the text is gibberish, noise descriptions (like 'noises', 'sounds', etc.), or unintelligible, respond ONLY with [inaudible]. Only respond with the translation or audio cues, do not include any other text."},
                {"role": "user", "content": text}
            ],
            temperature=0.3,
            timeout=15  # Add timeout in seconds
        )

        translated_text = response.choices[0].message.content.strip()
        return translated_text
    except Exception as e:
        print(f"Translation API error: {str(e)[:100]}. Retrying...")
        raise  # Re-raise for retry decorator to handle


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
            'original_text': '\n'.join(lines[2:]).strip(),
            'detected_language': None  # Will be populated if available
        })
    return subtitles


def translate_subtitle(client, subtitle, target_language):
    """Translate a single subtitle, used for parallel processing."""
    try:
        # Skip translation if target language matches detected language
        # This is an optimization to avoid unnecessary API calls
        if 'detected_language' in subtitle and subtitle['detected_language'] == target_language:
            return subtitle

        # Add timeout handling for API calls
        start_time = time.time()
        max_time = 15  # Maximum seconds to wait for a single translation

        # Attempt translation
        translated_text = translate_text(
            client, subtitle['text'], target_language)

        # Check if we got a valid response
        if not translated_text or len(translated_text.strip()) == 0:
            raise ValueError("Empty translation received")

        subtitle['text'] = translated_text

    except Exception as e:
        print(
            f"Error translating subtitle {subtitle['number']}: {str(e)[:100]}. Using original text.")
        subtitle['text'] = subtitle['original_text']

    return subtitle


def translate_subtitles(client, subtitles, target_language):
    """Translate subtitles in parallel."""
    print(f"Translating {len(subtitles)} subtitles to {target_language}...")
    start_time = time.time()

    # Set up progress tracking
    total = len(subtitles)
    completed = 0
    print_progress = True

    # Reduce max_workers from 10 to 5 to avoid rate limiting
    max_workers = 5
    results = [None] * total

    # Track failed translations to retry them individually
    failed_indices = []

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all translation tasks
        futures = {executor.submit(translate_subtitle, client, subtitle, target_language): i
                   for i, subtitle in enumerate(subtitles)}

        # Process completed futures with timeout detection
        last_completion_time = time.time()
        stalled_threshold = 30  # seconds without progress before considering stalled

        for future in as_completed(futures):
            try:
                idx = futures[future]
                results[idx] = future.result()
                completed += 1
                last_completion_time = time.time()

                # Update progress every 5% or at least every 10 items
                if print_progress and (completed % max(1, min(10, total // 20)) == 0 or completed == total):
                    percent = (completed / total) * 100
                    elapsed = time.time() - start_time
                    est_total = elapsed / \
                        (completed / total) if completed > 0 else 0
                    remaining = est_total - elapsed
                    print(
                        f"  Progress: {completed}/{total} ({percent:.1f}%) - Est. remaining: {remaining:.1f}s")

                # Check for stalled progress
                if time.time() - last_completion_time > stalled_threshold:
                    print(
                        f"⚠️ Translation appears stalled. Continuing with available results...")
                    break

            except Exception as e:
                idx = futures[future]
                print(
                    f"Error translating subtitle {idx}: {e}. Using original text.")
                results[idx] = subtitles[idx]  # Use original subtitle
                failed_indices.append(idx)

    # Handle any missing results (should be rare)
    for i, result in enumerate(results):
        if result is None:
            print(
                f"Missing translation for subtitle {i}. Using original text.")
            results[i] = subtitles[i]

    # Retry failed translations individually with backoff
    if failed_indices:
        print(
            f"Retrying {len(failed_indices)} failed translations individually...")
        for idx in failed_indices:
            try:
                # Simple backoff - wait a moment before retrying
                time.sleep(1)
                results[idx] = translate_subtitle(
                    client, subtitles[idx], target_language)
                print(f"Successfully retried translation for subtitle {idx}")
            except Exception as e:
                print(
                    f"Retry failed for subtitle {idx}: {e}. Using original text.")
                results[idx] = subtitles[idx]

    elapsed = time.time() - start_time
    print(
        f"Translation completed in {elapsed:.1f} seconds ({completed}/{total} subtitles)")
    return results


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
    print("Converting subtitles to SRT format...")

    # Final check to ensure chronological ordering by timestamp
    subtitles_with_ms = []
    for subtitle in subtitles:
        try:
            start_str = subtitle['timestamp'].split(' --> ')[0]
            end_str = subtitle['timestamp'].split(' --> ')[1]
            start_ms = srt_time_to_ms(start_str)
            end_ms = srt_time_to_ms(end_str)
            subtitles_with_ms.append((subtitle, start_ms, end_ms))
        except Exception as e:
            print(
                f"⚠️ Warning: Error processing subtitle {subtitle.get('number', 'unknown')}: {e}")
            # Use a default value to avoid breaking the process
            subtitles_with_ms.append((subtitle, 0, 0))

    # Sort by start time first, then by end time
    print("Sorting subtitles by timestamp...")
    subtitles_with_ms.sort(key=lambda x: (x[1], x[2]))

    # Check for any out-of-order subtitles
    order_issues = 0
    for i in range(len(subtitles_with_ms) - 1):
        if subtitles_with_ms[i][1] > subtitles_with_ms[i+1][1]:
            order_issues += 1

    if order_issues > 0:
        print(
            f"⚠️ Found {order_issues} ordering issues after sorting. This should not happen.")
    else:
        print("✅ All subtitles are in strict chronological order")

    # Renumber subtitles sequentially
    print("Renumbering subtitles...")
    for i, (subtitle, _, _) in enumerate(subtitles_with_ms, 1):
        subtitle['number'] = str(i)

    # Generate SRT content
    srt_content = ""
    for subtitle, _, _ in subtitles_with_ms:
        srt_content += f"{subtitle['number']}\n{subtitle['timestamp']}\n{subtitle['text']}\n\n"

    return srt_content


def merge_srt_files(srt_contents, time_offsets):
    """Merge SRT files with timestamp adjustments and duplicate removal."""
    print("Merging subtitle chunks...")
    all_subtitles = []

    # Process each chunk and adjust timestamps
    for srt_content, offset in zip(srt_contents, time_offsets):
        subtitles = parse_srt(srt_content)
        for subtitle in subtitles:
            start_str, end_str = subtitle['timestamp'].split(' --> ')
            start_ms = srt_time_to_ms(start_str) + int(offset * 1000)
            end_ms = srt_time_to_ms(end_str) + int(offset * 1000)
            subtitle['timestamp'] = f"{ms_to_srt_time(start_ms)} --> {ms_to_srt_time(end_ms)}"
            # Store original start time for strict sorting
            subtitle['start_ms'] = start_ms
            subtitle['end_ms'] = end_ms
            all_subtitles.append(subtitle)

    print(f"Total subtitles before deduplication: {len(all_subtitles)}")

    # Sort strictly by start time first, then by end time
    all_subtitles.sort(key=lambda x: (x['start_ms'], x['end_ms']))

    # Improved deduplication that preserves chronological order
    unique_subtitles = []
    seen = set()

    for subtitle in all_subtitles:
        # Create a unique key that includes timing information
        key = (subtitle['timestamp'], subtitle['text'])

        # Only add if we haven't seen this exact subtitle before
        if key not in seen:
            seen.add(key)
            unique_subtitles.append(subtitle)

    # Renumber subtitles sequentially
    for i, subtitle in enumerate(unique_subtitles, 1):
        subtitle['number'] = str(i)

    # Final verification of chronological order - ensure entire list is sorted
    is_sorted = all(unique_subtitles[i]['start_ms'] <= unique_subtitles[i+1]['start_ms']
                    for i in range(len(unique_subtitles)-1))

    if not is_sorted:
        print("⚠️ Warning: Subtitles not in strict chronological order. Resorting...")
        # Resort by start time and end time
        unique_subtitles.sort(key=lambda x: (x['start_ms'], x['end_ms']))
        # Renumber after resorting
        for i, subtitle in enumerate(unique_subtitles, 1):
            subtitle['number'] = str(i)
    else:
        print("✅ Subtitles are in strict chronological order")

    # Clean up temporary fields
    for subtitle in unique_subtitles:
        if 'start_ms' in subtitle:
            del subtitle['start_ms']
        if 'end_ms' in subtitle:
            del subtitle['end_ms']

    print(f"Unique subtitles after deduplication: {len(unique_subtitles)}")
    return unique_subtitles


def validate_srt(srt_content):
    """Validate SRT for sequential timestamps and no overlaps."""
    print("Validating subtitles...")
    subtitles = parse_srt(srt_content)
    overlap_count = 0
    order_issues = 0

    if not subtitles:
        print("⚠️ Warning: No subtitles found to validate")
        return False

    # Check for chronological ordering
    prev_start = -1
    prev_start_str = "00:00:00,000"
    prev_number = "0"

    for i in range(len(subtitles)):
        try:
            current_start = srt_time_to_ms(
                subtitles[i]['timestamp'].split(' --> ')[0])
            current_end = srt_time_to_ms(
                subtitles[i]['timestamp'].split(' --> ')[1])
            current_start_str = subtitles[i]['timestamp'].split(' --> ')[0]

            # Check chronological order
            if prev_start > current_start:
                order_issues += 1
                print(
                    f"⚠️ Non-chronological subtitle: #{subtitles[i]['number']} ({current_start_str}) comes after #{prev_number} ({prev_start_str})")

            prev_start = current_start
            prev_start_str = current_start_str
            prev_number = subtitles[i]['number']

            # Check for overlaps with next subtitle
            if i < len(subtitles) - 1:
                next_start = srt_time_to_ms(
                    subtitles[i + 1]['timestamp'].split(' --> ')[0])
                if current_end > next_start:
                    overlap_count += 1
                    print(
                        f"⚠️ Overlapping subtitles: #{subtitles[i]['number']} ends at {subtitles[i]['timestamp'].split(' --> ')[1]}, " +
                        f"#{subtitles[i+1]['number']} starts at {subtitles[i+1]['timestamp'].split(' --> ')[0]}")
        except Exception as e:
            print(
                f"⚠️ Error validating subtitle #{subtitles[i]['number']}: {e}")

    # Print summary
    if order_issues == 0 and overlap_count == 0:
        print("✅ Validation complete: All subtitles are in chronological order with no overlaps")
    else:
        print(
            f"⚠️ Validation complete: {overlap_count} overlapping subtitles found, {order_issues} chronological issues found")

    return order_issues == 0 and overlap_count == 0


def transcribe_chunk(client, chunk_file, language, chunk_index=0):
    """Helper function for parallel transcription.

    Args:
        client: OpenAI client
        chunk_file: Path to audio chunk file
        language: Language code or None for auto-detection
        chunk_index: Index of the chunk in the original sequence

    Returns:
        Dictionary containing the transcription result and chunk index
    """
    chunk_name = os.path.basename(chunk_file)
    print(f"Transcribing {chunk_name} (chunk {chunk_index})...")
    try:
        with open(chunk_file, "rb") as f:
            result = transcribe_audio(client, f, language=language)
            print(
                f"Completed transcription of {chunk_name} (chunk {chunk_index})")
            return {"content": result, "chunk_index": chunk_index}
    except Exception as e:
        print(
            f"Warning: Failed to transcribe {chunk_name} (chunk {chunk_index}): {e}. Skipping.")
        return {"content": "", "chunk_index": chunk_index}


def fix_overlapping_subtitles(subtitles, gap_ms=20, min_duration_ms=500, chars_per_second=17):
    """
    Fix overlapping subtitles by considering word count and content length.
    Only modifies overlapping subtitles while preserving chronological order.

    Args:
        subtitles: List of subtitle dictionaries
        gap_ms: Desired gap between subtitles in milliseconds
        min_duration_ms: Absolute minimum duration for any subtitle
        chars_per_second: Reading speed in characters per second

    Returns:
        List of subtitles with overlaps fixed
    """
    # Ensure subtitles are sorted by start time
    subtitles = sorted(subtitles, key=lambda x: srt_time_to_ms(
        x['timestamp'].split(' --> ')[0]))

    adjusted_count = 0
    removed_count = 0

    # Create a log file for timestamp changes
    log_file_path = os.path.join("output_srt_files", "timestamp_fixes.log")
    with open(log_file_path, "w") as log_file:
        log_file.write("SUBTITLE TIMESTAMP FIXES\n")
        log_file.write("=======================\n\n")

        # Process each subtitle except the last one
        i = 0
        while i < len(subtitles) - 1:
            # Get previous, current, and next subtitles
            prev_sub = subtitles[i-1] if i > 0 else None
            current = subtitles[i]
            next_sub = subtitles[i + 1]

            # Extract timestamps
            current_start_str, current_end_str = current['timestamp'].split(
                ' --> ')
            next_start_str, next_end_str = next_sub['timestamp'].split(' --> ')

            # Get previous subtitle timestamps if available
            prev_end_ms = None
            if prev_sub:
                prev_end_str = prev_sub['timestamp'].split(' --> ')[1]
                prev_end_ms = srt_time_to_ms(prev_end_str)
                prev_is_empty = prev_sub['text'].strip() == ""

            # Convert to milliseconds
            current_start_ms = srt_time_to_ms(current_start_str)
            current_end_ms = srt_time_to_ms(current_end_str)
            next_start_ms = srt_time_to_ms(next_start_str)
            next_end_ms = srt_time_to_ms(next_end_str)

            # Check if next subtitle is empty (was [inaudible])
            next_is_empty = next_sub['text'].strip() == ""

            # Check for overlap
            if current_end_ms > next_start_ms:
                # Log the overlap
                overlap_ms = current_end_ms - next_start_ms
                log_file.write(
                    f"Subtitle #{current['number']} overlaps with #{next_sub['number']} by {overlap_ms}ms\n")
                log_file.write(f"BEFORE: {current['timestamp']}\n")

                # Store original timestamp for logging
                original_timestamp = current['timestamp']

                # Calculate word counts and character lengths
                current_text = current['text']
                next_text = next_sub['text']
                current_word_count = len(current_text.split())
                next_word_count = len(next_text.split())
                current_char_count = len(current_text)
                next_char_count = len(next_text)

                # Calculate minimum reading time needed for each subtitle (in ms)
                current_min_time = max(
                    min_duration_ms, (current_char_count * 1000) // chars_per_second)

                # Check if we can use space from previous subtitle
                prev_space_available = 0
                if prev_sub:
                    # Calculate gap between previous and current
                    prev_gap = current_start_ms - prev_end_ms

                    # If previous is empty, we can remove it and use its space
                    if prev_is_empty:
                        # Find the subtitle before previous
                        before_prev_idx = i - 2
                        if before_prev_idx >= 0:
                            before_prev = subtitles[before_prev_idx]
                            before_prev_end_str = before_prev['timestamp'].split(
                                ' --> ')[1]
                            before_prev_end_ms = srt_time_to_ms(
                                before_prev_end_str)

                            # Calculate total available space
                            prev_space_available = current_start_ms - before_prev_end_ms - gap_ms

                            # Log the potential space
                            log_file.write(
                                f"NOTE: Can use {prev_space_available}ms from empty previous subtitle #{prev_sub['number']}\n")

                            # Remove the empty previous subtitle
                            subtitles.pop(i-1)
                            removed_count += 1
                            i -= 1  # Adjust current index

                            # Update current after removal
                            current = subtitles[i]
                            current_start_str, current_end_str = current['timestamp'].split(
                                ' --> ')
                            current_start_ms = srt_time_to_ms(
                                current_start_str)
                    else:
                        # Just use the gap if previous is not empty
                        prev_space_available = prev_gap - gap_ms if prev_gap > gap_ms else 0
                        if prev_space_available > 0:
                            log_file.write(
                                f"NOTE: Can use {prev_space_available}ms gap from previous subtitle #{prev_sub['number']}\n")

                # If next subtitle is empty, we can use its space
                if next_is_empty:
                    # Find the subtitle after next
                    after_next_idx = i + 2
                    if after_next_idx < len(subtitles):
                        after_next = subtitles[after_next_idx]
                        after_next_start_str = after_next['timestamp'].split(
                            ' --> ')[0]
                        after_next_start_ms = srt_time_to_ms(
                            after_next_start_str)

                        # Use space up to the start of the subtitle after next
                        new_end_ms = after_next_start_ms - gap_ms

                        # If we have space from previous, we can also adjust start time
                        if prev_space_available > 0:
                            # Calculate how much we need to shift start time
                            needed_duration = current_min_time
                            current_duration = current_end_ms - current_start_ms

                            if current_duration < needed_duration:
                                # Calculate how much additional time we need
                                additional_needed = needed_duration - current_duration

                                # Use as much as possible from previous space
                                shift_start = min(
                                    prev_space_available, additional_needed)
                                new_start_ms = current_start_ms - shift_start

                                # Update start time
                                current_start_str = ms_to_srt_time(
                                    new_start_ms)
                                log_file.write(
                                    f"NOTE: Shifted start time earlier by {shift_start}ms\n")

                        # Update the timestamp
                        current['timestamp'] = f"{current_start_str} --> {ms_to_srt_time(new_end_ms)}"
                        adjusted_count += 1

                        # Log the change and removal
                        log_file.write(f"AFTER:  {current['timestamp']}\n")
                        log_file.write(
                            f"NOTE:   Removed empty subtitle #{next_sub['number']} to resolve overlap\n")
                        log_file.write(f"TEXT:   \"{current['text']}\"\n\n")

                        # Remove the empty subtitle
                        subtitles.pop(i + 1)
                        removed_count += 1

                        # Don't increment i since we removed a subtitle
                        continue
                    else:
                        # No subtitle after next, just extend to the end of next
                        new_end_ms = next_end_ms

                        # If we have space from previous, we can also adjust start time
                        if prev_space_available > 0:
                            # Calculate how much we need to shift start time
                            needed_duration = current_min_time
                            current_duration = current_end_ms - current_start_ms

                            if current_duration < needed_duration:
                                # Calculate how much additional time we need
                                additional_needed = needed_duration - current_duration

                                # Use as much as possible from previous space
                                shift_start = min(
                                    prev_space_available, additional_needed)
                                new_start_ms = current_start_ms - shift_start

                                # Update start time
                                current_start_str = ms_to_srt_time(
                                    new_start_ms)
                                log_file.write(
                                    f"NOTE: Shifted start time earlier by {shift_start}ms\n")

                        current['timestamp'] = f"{current_start_str} --> {ms_to_srt_time(new_end_ms)}"
                        adjusted_count += 1

                        # Log the change and removal
                        log_file.write(f"AFTER:  {current['timestamp']}\n")
                        log_file.write(
                            f"NOTE:   Removed empty subtitle #{next_sub['number']} to resolve overlap\n")
                        log_file.write(f"TEXT:   \"{current['text']}\"\n\n")

                        # Remove the empty subtitle
                        subtitles.pop(i + 1)
                        removed_count += 1

                        # Don't increment i since we removed a subtitle
                        continue

                # Original overlap resolution logic for non-empty subtitles
                if current_word_count >= next_word_count * 2:
                    # Current subtitle has significantly more content, preserve more of its time
                    # Adjust to minimize impact on the longer subtitle
                    new_end_ms = next_start_ms - gap_ms

                    # If we have space from previous, we can also adjust start time
                    if prev_space_available > 0:
                        # Calculate how much we need to shift start time
                        needed_duration = current_min_time
                        current_duration = new_end_ms - current_start_ms  # Use new end time

                        if current_duration < needed_duration:
                            # Calculate how much additional time we need
                            additional_needed = needed_duration - current_duration

                            # Use as much as possible from previous space
                            shift_start = min(
                                prev_space_available, additional_needed)
                            new_start_ms = current_start_ms - shift_start

                            # Update start time
                            current_start_str = ms_to_srt_time(new_start_ms)
                            log_file.write(
                                f"NOTE: Shifted start time earlier by {shift_start}ms\n")

                    # Ensure current subtitle still has its minimum needed time
                    current_duration = new_end_ms - \
                        srt_time_to_ms(current_start_str)
                    if current_duration < current_min_time:
                        # Can't shorten current subtitle enough, it needs more time
                        new_end_ms = srt_time_to_ms(
                            current_start_str) + current_min_time
                        log_file.write(
                            f"WARNING: Subtitle needs {current_min_time}ms for {current_word_count} words, overlap remains\n")
                    else:
                        # Update the timestamp
                        current['timestamp'] = f"{current_start_str} --> {ms_to_srt_time(new_end_ms)}"
                        adjusted_count += 1
                else:
                    # Next subtitle has similar or more content, or both are short
                    # Simply end current subtitle before next one starts
                    new_end_ms = next_start_ms - gap_ms

                    # If we have space from previous, we can also adjust start time
                    if prev_space_available > 0:
                        # Calculate how much we need to shift start time
                        needed_duration = current_min_time
                        current_duration = new_end_ms - current_start_ms  # Use new end time

                        if current_duration < needed_duration:
                            # Calculate how much additional time we need
                            additional_needed = needed_duration - current_duration

                            # Use as much as possible from previous space
                            shift_start = min(
                                prev_space_available, additional_needed)
                            new_start_ms = current_start_ms - shift_start

                            # Update start time
                            current_start_str = ms_to_srt_time(new_start_ms)
                            log_file.write(
                                f"NOTE: Shifted start time earlier by {shift_start}ms\n")

                    # Ensure minimum duration
                    current_duration = new_end_ms - \
                        srt_time_to_ms(current_start_str)
                    if current_duration < min_duration_ms:
                        # If fixing would make subtitle too short, use minimum duration
                        if srt_time_to_ms(current_start_str) + min_duration_ms < next_start_ms - gap_ms:
                            new_end_ms = srt_time_to_ms(
                                current_start_str) + min_duration_ms
                        else:
                            # We can't maintain minimum duration without overlap
                            new_end_ms = next_start_ms - gap_ms

                    # Update the timestamp
                    current['timestamp'] = f"{current_start_str} --> {ms_to_srt_time(new_end_ms)}"
                    adjusted_count += 1

                # Log the change
                log_file.write(f"AFTER:  {current['timestamp']}\n")
                log_file.write(f"TEXT:   \"{current['text']}\"\n\n")

                # Also print to console
                print(
                    f"Fixed overlap: #{current['number']} {original_timestamp} → {current['timestamp']}")

            # Check if current subtitle is empty and next is not empty
            elif current['text'].strip() == "" and next_sub['text'].strip() != "":
                # Remove empty subtitle
                log_file.write(
                    f"Removing empty subtitle #{current['number']}\n")
                log_file.write(f"TIMESTAMP: {current['timestamp']}\n\n")

                subtitles.pop(i)
                removed_count += 1

                # Don't increment i since we removed a subtitle
                continue

            # Move to next subtitle
            i += 1

    # Renumber subtitles sequentially after removals
    if removed_count > 0:
        for i, subtitle in enumerate(subtitles, 1):
            subtitle['number'] = str(i)

    if adjusted_count > 0:
        print(
            f"✅ Fixed {adjusted_count} overlapping subtitles (see output_srt_files/timestamp_fixes.log for details)")
    else:
        print("✅ No overlapping subtitles found")

    if removed_count > 0:
        print(f"✅ Removed {removed_count} empty subtitles to improve timing")

    return subtitles


def remove_inaudible_markers(subtitles):
    """Remove [inaudible] markers from subtitles while preserving timestamps."""
    inaudible_count = 0

    for subtitle in subtitles:
        if subtitle['text'].strip() == "[inaudible]":
            subtitle['text'] = ""
            inaudible_count += 1
        elif "[inaudible]" in subtitle['text']:
            subtitle['text'] = subtitle['text'].replace(
                "[inaudible]", "").strip()
            inaudible_count += 1

    if inaudible_count > 0:
        print(
            f"✅ Removed [inaudible] markers from {inaudible_count} subtitles")

    return subtitles


def split_subtitle(subtitle_num: int, start_time: str, end_time: str, text: str,
                   max_length: int = 78):
    """
    Split a long subtitle into multiple parts based on natural language breaks.
    Returns a list of (start_time, end_time, text) tuples.
    """
    if len(text) <= max_length:
        return [(start_time, end_time, text)]

    # Calculate total duration in milliseconds
    start_ms = srt_time_to_ms(start_time)
    end_ms = srt_time_to_ms(end_time)
    total_duration = end_ms - start_ms

    # Try to split at sentence boundaries first
    sentences = re.split(r'([.!?])\s+', text)
    if len(sentences) > 1:
        # Recombine the sentences with their punctuation
        reconstructed = []
        i = 0
        while i < len(sentences):
            if i + 1 < len(sentences) and sentences[i+1] in ['.', '!', '?']:
                reconstructed.append(sentences[i] + sentences[i+1])
                i += 2
            else:
                reconstructed.append(sentences[i])
                i += 1

        # Now try to group sentences into segments
        segments = []
        current_segment = ""

        for sentence in reconstructed:
            if not sentence.strip():
                continue

            if len(current_segment) + len(sentence) + 1 <= max_length or not current_segment:
                if current_segment:
                    current_segment += " " + sentence
                else:
                    current_segment = sentence
            else:
                segments.append(current_segment.strip())
                current_segment = sentence

        if current_segment:
            segments.append(current_segment.strip())

        if segments and all(len(segment) <= max_length for segment in segments):
            # Calculate time distribution based on character count
            total_chars = sum(len(s) for s in segments)
            result = []

            current_start_ms = start_ms
            for i, segment in enumerate(segments):
                segment_duration = int(
                    total_duration * (len(segment) / total_chars))

                # Ensure the last segment ends exactly at the original end time
                if i == len(segments) - 1:
                    segment_end_ms = end_ms
                else:
                    segment_end_ms = current_start_ms + segment_duration

                result.append((
                    ms_to_srt_time(current_start_ms),
                    ms_to_srt_time(segment_end_ms),
                    segment
                ))

                # Add a small gap between segments (100ms)
                current_start_ms = segment_end_ms + 100

            return result

    # If sentence splitting didn't work well, try clause splitting
    clauses = re.split(r',\s+', text)
    if len(clauses) > 1:
        segments = []
        current_segment = ""

        for clause in clauses:
            if not clause.strip():
                continue

            if len(current_segment) + len(clause) + 2 <= max_length or not current_segment:
                if current_segment:
                    current_segment += ", " + clause
                else:
                    current_segment = clause
            else:
                segments.append(current_segment.strip())
                current_segment = clause

        if current_segment:
            segments.append(current_segment.strip())

        if segments and all(len(segment) <= max_length for segment in segments):
            # Calculate time distribution based on character count
            total_chars = sum(len(s) for s in segments)
            result = []

            current_start_ms = start_ms
            for i, segment in enumerate(segments):
                segment_duration = int(
                    total_duration * (len(segment) / total_chars))

                # Ensure the last segment ends exactly at the original end time
                if i == len(segments) - 1:
                    segment_end_ms = end_ms
                else:
                    segment_end_ms = current_start_ms + segment_duration

                result.append((
                    ms_to_srt_time(current_start_ms),
                    ms_to_srt_time(segment_end_ms),
                    segment
                ))

                # Add a small gap between segments (100ms)
                current_start_ms = segment_end_ms + 100

            return result

    # If all else fails, split by words
    words = text.split()
    segments = []
    current_segment = ""

    for word in words:
        if len(current_segment) + len(word) + 1 <= max_length or not current_segment:
            if current_segment:
                current_segment += " " + word
            else:
                current_segment = word
        else:
            segments.append(current_segment.strip())
            current_segment = word

    if current_segment:
        segments.append(current_segment.strip())

    # Calculate time distribution based on character count
    total_chars = sum(len(s) for s in segments)
    result = []

    current_start_ms = start_ms
    for i, segment in enumerate(segments):
        segment_duration = int(total_duration * (len(segment) / total_chars))

        # Ensure the last segment ends exactly at the original end time
        if i == len(segments) - 1:
            segment_end_ms = end_ms
        else:
            segment_end_ms = current_start_ms + segment_duration

        result.append((
            ms_to_srt_time(current_start_ms),
            ms_to_srt_time(segment_end_ms),
            segment
        ))

        # Add a small gap between segments (100ms)
        current_start_ms = segment_end_ms + 100

    return result


def split_long_subtitles(subtitles, max_length=0):
    """
    Split long subtitles into multiple shorter subtitles.

    Args:
        subtitles: List of subtitle dictionaries
        max_length: Maximum length for subtitles (0 for auto-detection based on mean + std_dev)

    Returns:
        List of subtitle dictionaries with long subtitles split
    """
    print("Analyzing subtitle lengths for splitting...")

    # Calculate mean and standard deviation
    lengths = [len(s['text']) for s in subtitles]
    mean_length = statistics.mean(lengths) if lengths else 0
    std_dev = statistics.stdev(lengths) if len(lengths) > 1 else 0

    # Use mean + std_dev as the threshold if max_length is not specified
    if max_length <= 0:
        max_length = int(mean_length + std_dev)

    print(f"Mean subtitle length: {mean_length:.2f} characters")
    print(f"Standard deviation: {std_dev:.2f} characters")
    print(f"Maximum length threshold: {max_length} characters")

    # Find long subtitles
    long_subtitles = [s for s in subtitles if len(s['text']) > max_length]
    print(
        f"Found {len(long_subtitles)} subtitles longer than {max_length} characters")

    if not long_subtitles:
        return subtitles  # No long subtitles to split

    # Process subtitles
    new_subtitles = []
    split_count = 0

    for subtitle in subtitles:
        text = subtitle['text']
        timestamp = subtitle['timestamp']
        start_time, end_time = timestamp.split(' --> ')

        if len(text) > max_length:
            # Split long subtitle
            splits = split_subtitle(
                int(subtitle['number']), start_time, end_time, text, max_length)
            for split_start, split_end, split_text in splits:
                new_subtitle = subtitle.copy()
                new_subtitle['timestamp'] = f"{split_start} --> {split_end}"
                new_subtitle['text'] = split_text
                new_subtitles.append(new_subtitle)
            split_count += 1
        else:
            # Keep short subtitle as is
            new_subtitles.append(subtitle)

    print(
        f"✅ Split {split_count} long subtitles into {len(new_subtitles) - len(subtitles) + split_count} parts")
    return new_subtitles


def check_timestamps(srt_content):
    """
    Perform comprehensive validation of SRT timestamps and provide statistics.
    This function does not modify the SRT content, only analyzes it.
    """
    print("\n📊 TIMESTAMP VALIDATION AND STATISTICS 📊")
    print("═" * 60)

    # Extract timestamps
    timestamps = re.findall(
        r'(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})', srt_content)

    invalid = []
    overlaps = []
    durations = []
    short_subtitles = []
    min_duration_ms = 500  # 0.5 seconds

    # Check for empty subtitles
    empty_subtitles = re.findall(
        r'\d+\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n\n', srt_content)
    empty_count = len(empty_subtitles)

    # Extract subtitle text for analysis
    subtitle_blocks = re.split(r'\n\s*\n', srt_content.strip())
    subtitle_texts = []
    for block in subtitle_blocks:
        lines = block.strip().split('\n')
        if len(lines) >= 3:
            text = '\n'.join(lines[2:]).strip()
            subtitle_texts.append(text)

    for i, (start, end) in enumerate(timestamps, 1):
        start_ms = srt_time_to_ms(start)
        end_ms = srt_time_to_ms(end)
        duration_ms = end_ms - start_ms
        durations.append(duration_ms)

        if end_ms <= start_ms:
            invalid.append((i, start, end, end_ms - start_ms))

        if duration_ms < min_duration_ms:
            short_subtitles.append((i, start, end, duration_ms))

        if i < len(timestamps):
            next_start = timestamps[i][0]
            next_start_ms = srt_time_to_ms(next_start)

            if end_ms > next_start_ms:
                overlaps.append(
                    (i, i+1, end, next_start, end_ms - next_start_ms))

    # Calculate gaps between subtitles
    gaps = []
    for i in range(len(timestamps) - 1):
        current_end_ms = srt_time_to_ms(timestamps[i][1])
        next_start_ms = srt_time_to_ms(timestamps[i+1][0])
        gap_ms = next_start_ms - current_end_ms
        gaps.append(gap_ms)

    # Print validation results
    if invalid:
        print('❌ Invalid timestamps found:')
        for i, start, end, diff in invalid:
            print(f'  Subtitle #{i}: {start} --> {end} (diff: {diff}ms)')
    else:
        print('✅ No invalid timestamps found')

    if overlaps:
        print('\n❌ Overlapping timestamps found:')
        for i, j, end, next_start, diff in overlaps[:5]:  # Show first 5
            print(
                f'  Subtitle #{i} overlaps with #{j} by {diff}ms: {end} > {next_start}')
        if len(overlaps) > 5:
            print(f'  ... and {len(overlaps) - 5} more')
    else:
        print('\n✅ No overlapping timestamps found')

    if short_subtitles:
        print(
            f'\n⚠️ Found {len(short_subtitles)} subtitles shorter than {min_duration_ms}ms:')
        for i, start, end, duration in short_subtitles[:5]:  # Show first 5
            print(
                f'  Subtitle #{i}: {start} --> {end} (duration: {duration}ms)')
        if len(short_subtitles) > 5:
            print(f'  ... and {len(short_subtitles) - 5} more')
    else:
        print(f'\n✅ No subtitles shorter than {min_duration_ms}ms found')

    if empty_count > 0:
        print(f'\n⚠️ Found {empty_count} empty subtitles')
    else:
        print(f'\n✅ No empty subtitles found')

    # Duration statistics
    if durations:
        avg_duration = sum(durations) / len(durations)
        min_duration = min(durations)
        max_duration = max(durations)

        # Group durations into ranges
        duration_ranges = {
            '< 1s': 0,
            '1-2s': 0,
            '2-3s': 0,
            '3-5s': 0,
            '5-10s': 0,
            '> 10s': 0
        }

        for d in durations:
            if d < 1000:
                duration_ranges['< 1s'] += 1
            elif d < 2000:
                duration_ranges['1-2s'] += 1
            elif d < 3000:
                duration_ranges['2-3s'] += 1
            elif d < 5000:
                duration_ranges['3-5s'] += 1
            elif d < 10000:
                duration_ranges['5-10s'] += 1
            else:
                duration_ranges['> 10s'] += 1

        print('\n📈 Duration statistics:')
        print(
            f'  Average duration: {avg_duration:.2f}ms ({avg_duration/1000:.2f}s)')
        print(
            f'  Minimum duration: {min_duration}ms ({min_duration/1000:.2f}s)')
        print(
            f'  Maximum duration: {max_duration}ms ({max_duration/1000:.2f}s)')

        print('\n📊 Duration distribution:')
        for range_name, count in duration_ranges.items():
            percentage = (count / len(durations)) * 100
            print(f'  {range_name}: {count} subtitles ({percentage:.1f}%)')

    # Gap statistics
    if gaps:
        avg_gap = sum(gaps) / len(gaps)
        min_gap = min(gaps)
        max_gap = max(gaps)

        # Count negative gaps (overlaps)
        negative_gaps = sum(1 for g in gaps if g < 0)

        print('\n📏 Gap statistics:')
        print(f'  Average gap: {avg_gap:.2f}ms ({avg_gap/1000:.2f}s)')
        print(f'  Minimum gap: {min_gap}ms ({min_gap/1000:.2f}s)')
        print(f'  Maximum gap: {max_gap}ms ({max_gap/1000:.2f}s)')
        print(f'  Negative gaps (overlaps): {negative_gaps}')

    # Text statistics
    if subtitle_texts:
        text_lengths = [len(text) for text in subtitle_texts]
        avg_length = sum(text_lengths) / len(text_lengths)
        max_length = max(text_lengths)

        # Calculate standard deviation
        std_dev = statistics.stdev(text_lengths) if len(
            text_lengths) > 1 else 0

        print('\n📝 Text statistics:')
        print(f'  Average text length: {avg_length:.1f} characters')
        print(f'  Standard deviation: {std_dev:.1f} characters')
        print(f'  Maximum text length: {max_length} characters')

        # Count long subtitles (> mean + std_dev)
        threshold = avg_length + std_dev
        long_count = sum(1 for length in text_lengths if length > threshold)
        print(f'  Subtitles longer than {threshold:.1f} chars: {long_count}')

    print("═" * 60)

    # Return a summary of issues found
    return {
        'invalid': len(invalid),
        'overlaps': len(overlaps),
        'short': len(short_subtitles),
        'empty': empty_count
    }


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
    print("  ✅ Automatic language detection")
    print("  ✅ Long subtitle splitting")
    print("  ✅ Timestamp validation and statistics")

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

    # Get target language with improved guidance
    print("\n🌐 Target language (language for the subtitles):")
    print("   This is the language your subtitles will be translated to.")
    print("   The source language will be automatically detected.")
    print("   Common codes: en (English), es (Spanish), fr (French), etc.")
    target_language = input(
        "   Enter target language code [default: en]: ").strip().lower() or "en"
    if target_language in COMMON_LANGUAGES:
        print(f"   Selected: {COMMON_LANGUAGES[target_language]}")

    print(
        f"\n📝 NOTE: Audio language will be automatically detected and translated to {COMMON_LANGUAGES.get(target_language, target_language)}.")

    # Use default output directory without prompting
    output_dir = "output_srt_files"

    # Ask about splitting long subtitles
    split_long = input(
        "\n🔪 Split long subtitles? (y/n) [default: y]: ").strip().lower() != 'n'

    # Get max length if splitting is enabled
    max_length = 0
    if split_long:
        max_length_input = input(
            "   Maximum subtitle length (0 for auto-detection) [default: 0]: ").strip()
        if max_length_input and max_length_input.isdigit():
            max_length = int(max_length_input)

    # Ask about timestamp checking
    check_timestamps = input(
        "\n📊 Perform comprehensive timestamp checking? (y/n) [default: y]: ").strip().lower() != 'n'

    # Confirm settings
    print("\n" + "-"*60)
    print("📋 Summary of settings:")
    print(f"   Input file: {input_file}")
    print(f"   Output name: {output_name}")
    print(f"   Source language: Auto-detect")
    print(f"   Target language: {target_language}")
    print(f"   Output directory: {output_dir}")
    print(f"   Split long subtitles: {'Yes' if split_long else 'No'}")
    if split_long:
        print(
            f"   Maximum subtitle length: {max_length if max_length > 0 else 'Auto-detect'}")
    print(f"   Timestamp checking: {'Yes' if check_timestamps else 'No'}")
    print("-"*60)

    confirm = input("\n✅ Proceed with these settings? (y/n): ").strip().lower()
    if confirm != 'y':
        print("Operation cancelled.")
        return None

    return {
        'input': input_file,
        'output_name': output_name,
        'language': None,  # Set to None for auto-detection
        'target_language': target_language,
        'output_dir': output_dir,
        'no_split_long': not split_long,
        'max_length': max_length,
        'no_timestamp_check': not check_timestamps
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
        # Always translate to target language
        print(f"Will translate to {args.target_language} after transcription.")

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

                # Set up progress tracking for transcription
                total_chunks = len(chunk_files)
                completed_chunks = 0

                # Create a results array to maintain chunk order
                ordered_results = [None] * len(chunk_files)

                # Add timeout tracking
                transcription_start = time.time()
                last_progress_time = time.time()
                stall_timeout = 120  # 2 minutes without progress is considered stalled

                with ThreadPoolExecutor(max_workers=5) as executor:
                    futures = {executor.submit(transcribe_chunk, client, cf, args.language, i): i
                               for i, cf in enumerate(chunk_files)}

                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            chunk_index = result["chunk_index"]
                            ordered_results[chunk_index] = result["content"]

                            completed_chunks += 1
                            last_progress_time = time.time()
                            percent = (completed_chunks / total_chunks) * 100
                            print(
                                f"  Transcription progress: {completed_chunks}/{total_chunks} ({percent:.1f}%)")

                            # Check for stalled progress
                            if time.time() - last_progress_time > stall_timeout:
                                print(
                                    "⚠️ Transcription appears stalled. Continuing with available results...")
                                break
                        except Exception as e:
                            print(
                                f"Error processing chunk: {str(e)[:100]}. Continuing with available results.")

                # Handle any missing chunks
                if completed_chunks < total_chunks:
                    print(
                        f"⚠️ Only completed {completed_chunks}/{total_chunks} chunks. Proceeding with available data.")

                # Filter out None values and use results in the correct order
                srt_contents = [r for r in ordered_results if r is not None]

                print(
                    f"Transcription completed in {time.time() - transcription_start:.1f} seconds")

                # Merge all subtitles with proper timestamps
                subtitles = merge_srt_files(srt_contents, time_offsets)

                # Always translate
                subtitles = translate_subtitles(
                    client, subtitles, args.target_language)

                # Adjust timestamps
                subtitles = adjust_timestamps(subtitles)

                # Split long subtitles if enabled
                if not hasattr(args, 'no_split_long') or not args.no_split_long:
                    max_length = args.max_length if hasattr(
                        args, 'max_length') else 0
                    subtitles = split_long_subtitles(subtitles, max_length)

                # Final conversion to SRT with strict chronological ordering
                final_srt = subtitles_to_srt(subtitles)

                # Parse the final SRT to get subtitle objects for overlap fixing
                final_subtitles = parse_srt(final_srt)

                # Fix any overlapping subtitles
                fixed_subtitles = fix_overlapping_subtitles(final_subtitles)

                # Remove [inaudible] markers
                fixed_subtitles = remove_inaudible_markers(fixed_subtitles)

                # Convert back to SRT format
                fixed_srt = subtitles_to_srt(fixed_subtitles)

                # Validate the final SRT
                validate_srt(fixed_srt)

                # Perform comprehensive timestamp checking
                if not hasattr(args, 'no_timestamp_check') or not args.no_timestamp_check:
                    check_timestamps(fixed_srt)

                # Write to file
                with open(srt_path, "w") as srt_file:
                    srt_file.write(fixed_srt)
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
                # Always translate
                subtitles = translate_subtitles(
                    client, subtitles, args.target_language)
                subtitles = adjust_timestamps(subtitles)

                # Split long subtitles if enabled
                if not hasattr(args, 'no_split_long') or not args.no_split_long:
                    max_length = args.max_length if hasattr(
                        args, 'max_length') else 0
                    subtitles = split_long_subtitles(subtitles, max_length)

                final_srt = subtitles_to_srt(subtitles)

                # Parse the final SRT to get subtitle objects for overlap fixing
                final_subtitles = parse_srt(final_srt)

                # Fix any overlapping subtitles
                fixed_subtitles = fix_overlapping_subtitles(final_subtitles)

                # Remove [inaudible] markers
                fixed_subtitles = remove_inaudible_markers(fixed_subtitles)

                # Convert back to SRT format
                fixed_srt = subtitles_to_srt(fixed_subtitles)

                validate_srt(fixed_srt)

                # Perform comprehensive timestamp checking
                if not hasattr(args, 'no_timestamp_check') or not args.no_timestamp_check:
                    check_timestamps(fixed_srt)

                with open(srt_path, "w") as srt_file:
                    srt_file.write(fixed_srt)
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
            description="Generate consistent SRT files with automatic language detection and translation.")
        parser.add_argument(
            "input", help="Input file path (e.g., video.mp4, audio.mp3)")
        parser.add_argument(
            "output_name", help="Output SRT file name (without extension)")
        parser.add_argument("--target-language", default="en",
                            help="Target language for the subtitles (e.g., 'fr', 'es')")
        parser.add_argument(
            "--output-dir", default="output_srt_files", help="Output directory")
        parser.add_argument("--interactive", action="store_true",
                            help="Use interactive menu instead of command line arguments")
        parser.add_argument("--no-split-long", action="store_true",
                            help="Disable splitting of long subtitles")
        parser.add_argument("--max-length", type=int, default=0,
                            help="Maximum length for subtitles (0 for auto-detection)")
        parser.add_argument("--no-timestamp-check", action="store_true",
                            help="Disable comprehensive timestamp checking")

        args = parser.parse_args()

        # Handle quoted paths in command-line arguments
        if args.input and (args.input.startswith("'") and args.input.endswith("'")) or \
           (args.input.startswith('"') and args.input.endswith('"')):
            args.input = args.input[1:-1]  # Remove surrounding quotes

        if args.output_dir and (args.output_dir.startswith("'") and args.output_dir.endswith("'")) or \
           (args.output_dir.startswith('"') and args.output_dir.endswith('"')):
            # Remove surrounding quotes
            args.output_dir = args.output_dir[1:-1]

        # Set language to None for auto-detection
        args.language = None

        # Display language information
        if args.target_language in COMMON_LANGUAGES:
            print(
                f"\n📝 NOTE: Audio language will be automatically detected and translated to {COMMON_LANGUAGES[args.target_language]}.")
        else:
            print(
                f"\n📝 NOTE: Audio language will be automatically detected and translated to '{args.target_language}'.")

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
