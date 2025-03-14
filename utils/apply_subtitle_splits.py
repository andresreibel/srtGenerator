#!/usr/bin/env python3
import re
import sys
import os
import statistics
from typing import List, Tuple, Dict


def time_to_ms(time_str: str) -> int:
    """Convert SRT time format to milliseconds."""
    hours, minutes, seconds = time_str.split(':')
    seconds, milliseconds = seconds.split(',')
    return (int(hours) * 3600 + int(minutes) * 60 + int(seconds)) * 1000 + int(milliseconds)


def ms_to_srt_time(ms: int) -> str:
    """Convert milliseconds to SRT time format."""
    hours = ms // 3600000
    ms %= 3600000
    minutes = ms // 60000
    ms %= 60000
    seconds = ms // 1000
    ms %= 1000
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{ms:03d}"


def analyze_subtitle_lengths(srt_file: str) -> Tuple[float, float, float, List[Tuple[int, str, str, str, int]]]:
    """
    Analyze subtitle lengths from an SRT file.
    Returns mean, standard deviation, and a list of subtitles with their details.
    """
    with open(srt_file, 'r', encoding='utf-8') as f:
        content = f.read()

    # Extract subtitles using regex
    subtitle_pattern = r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n((?:.+\n)+)'
    subtitles = re.findall(subtitle_pattern, content, re.MULTILINE)

    # Calculate lengths
    subtitle_data = []
    lengths = []

    for subtitle_num, start_time, end_time, text in subtitles:
        text = text.strip()
        length = len(text)
        lengths.append(length)
        subtitle_data.append(
            (int(subtitle_num), start_time, end_time, text, length))

    mean_length = statistics.mean(lengths) if lengths else 0
    median_length = statistics.median(lengths) if lengths else 0
    std_dev = statistics.stdev(lengths) if len(lengths) > 1 else 0

    return mean_length, median_length, std_dev, subtitle_data


def split_subtitle(subtitle_num: int, start_time: str, end_time: str, text: str,
                   max_length: int = 78) -> List[Tuple[str, str, str]]:
    """
    Split a long subtitle into multiple parts based on natural language breaks.
    Returns a list of (start_time, end_time, text) tuples.
    """
    if len(text) <= max_length:
        return [(start_time, end_time, text)]

    # Calculate total duration in milliseconds
    start_ms = time_to_ms(start_time)
    end_ms = time_to_ms(end_time)
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


def extract_subtitles(srt_file: str) -> List[Tuple[int, str, str, str]]:
    """Extract subtitles from an SRT file."""
    with open(srt_file, 'r', encoding='utf-8') as f:
        content = f.read()

    subtitle_pattern = r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n((?:.+\n)+)'
    return re.findall(subtitle_pattern, content, re.MULTILINE)


def calculate_stats(subtitles: List[Tuple[int, str, str, str]]) -> Dict:
    """Calculate statistics for subtitle lengths."""
    lengths = []
    for _, _, _, text in subtitles:
        text = text.strip()
        lengths.append(len(text))

    return {
        'count': len(lengths),
        'min': min(lengths) if lengths else 0,
        'max': max(lengths) if lengths else 0,
        'mean': statistics.mean(lengths) if lengths else 0,
        'median': statistics.median(lengths) if lengths else 0,
        'std_dev': statistics.stdev(lengths) if len(lengths) > 1 else 0,
        # Using 78 as threshold (mean + std_dev)
        'long_count': sum(1 for l in lengths if l > 78)
    }


def print_comparison(original_stats: Dict, split_stats: Dict):
    """Print a comparison of statistics before and after splitting."""
    print("┌─────────────────────────────────────────────────────────┐")
    print("│             SUBTITLE STATISTICS COMPARISON              │")
    print("├───────────────────┬─────────────────┬─────────────────┤")
    print("│      Metric       │     Original    │      Split      │")
    print("├───────────────────┼─────────────────┼─────────────────┤")
    print(
        f"│ Total Subtitles   │ {original_stats['count']:^15} │ {split_stats['count']:^15} │")
    print(
        f"│ Min Length        │ {original_stats['min']:^15.2f} │ {split_stats['min']:^15.2f} │")
    print(
        f"│ Max Length        │ {original_stats['max']:^15.2f} │ {split_stats['max']:^15.2f} │")
    print(
        f"│ Mean Length       │ {original_stats['mean']:^15.2f} │ {split_stats['mean']:^15.2f} │")
    print(
        f"│ Median Length     │ {original_stats['median']:^15.2f} │ {split_stats['median']:^15.2f} │")
    print(
        f"│ Standard Dev      │ {original_stats['std_dev']:^15.2f} │ {split_stats['std_dev']:^15.2f} │")
    print(
        f"│ Long Subtitles    │ {original_stats['long_count']:^15} │ {split_stats['long_count']:^15} │")
    print("└───────────────────┴─────────────────┴─────────────────┘")

    # Calculate improvements
    length_reduction = (
        (original_stats['mean'] - split_stats['mean']) / original_stats['mean']) * 100
    long_reduction = ((original_stats['long_count'] - split_stats['long_count']) / max(
        1, original_stats['long_count'])) * 100

    print(f"\n✅ Mean subtitle length reduced by {length_reduction:.1f}%")
    print(f"✅ Long subtitles reduced by {long_reduction:.1f}%")
    print(
        f"✅ Added {split_stats['count'] - original_stats['count']} new subtitles for better readability")


def process_srt_file(input_file: str, output_file: str, max_length: int = 78) -> Tuple[int, int]:
    """
    Process an SRT file, splitting long subtitles and writing to a new file.
    Returns a tuple of (original_count, new_count).
    """
    # Get statistics and subtitle data
    mean_length, _, std_dev, subtitle_data = analyze_subtitle_lengths(
        input_file)

    # Use mean + std_dev as the threshold if max_length is not specified
    if max_length <= 0:
        max_length = int(mean_length + std_dev)

    # Process subtitles
    new_subtitles = []
    subtitle_count = 0

    for num, start_time, end_time, text, length in subtitle_data:
        subtitle_count += 1

        if length > max_length:
            # Split long subtitle
            splits = split_subtitle(
                num, start_time, end_time, text, max_length)
            for split_start, split_end, split_text in splits:
                new_subtitles.append((split_start, split_end, split_text))
        else:
            # Keep short subtitle as is
            new_subtitles.append((start_time, end_time, text))

    # Write new SRT file
    with open(output_file, 'w', encoding='utf-8') as f:
        for i, (start_time, end_time, text) in enumerate(new_subtitles, 1):
            f.write(f"{i}\n")
            f.write(f"{start_time} --> {end_time}\n")
            f.write(f"{text}\n\n")

    return subtitle_count, len(new_subtitles)


def compare_files(input_file: str, output_file: str):
    """Compare statistics between original and split subtitle files."""
    original_subtitles = extract_subtitles(input_file)
    split_subtitles = extract_subtitles(output_file)

    original_stats = calculate_stats(original_subtitles)
    split_stats = calculate_stats(split_subtitles)

    print_comparison(original_stats, split_stats)


def show_examples(input_file: str, output_file: str, num_examples: int = 5):
    """Show before/after examples of split subtitles."""
    # Get original subtitles
    with open(input_file, 'r', encoding='utf-8') as f:
        original_content = f.read()

    # Get improved subtitles
    with open(output_file, 'r', encoding='utf-8') as f:
        improved_content = f.read()

    # Extract original subtitles
    original_pattern = r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n((?:.+\n)+)'
    original_subtitles = re.findall(
        original_pattern, original_content, re.MULTILINE)

    # Find long subtitles
    mean_length, _, std_dev, _ = analyze_subtitle_lengths(input_file)
    max_length = int(mean_length + std_dev)

    long_subtitles = []
    for num, start, end, text in original_subtitles:
        if len(text.strip()) > max_length:
            long_subtitles.append((int(num), start, end, text.strip()))

    # Sort by length (descending)
    long_subtitles.sort(key=lambda x: len(x[3]), reverse=True)

    # Take top examples
    examples = long_subtitles[:num_examples]

    # Extract all improved subtitles for efficient lookup
    improved_pattern = r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n((?:.+\n)+)'
    all_improved_subtitles = re.findall(
        improved_pattern, improved_content, re.MULTILINE)

    # Convert to a list of (num, start_ms, end_ms, start, end, text) for easier processing
    improved_subtitles_data = []
    for num, start, end, text in all_improved_subtitles:
        start_ms = time_to_ms(start)
        end_ms = time_to_ms(end)
        improved_subtitles_data.append(
            (int(num), start_ms, end_ms, start, end, text.strip()))

    # Sort by start time
    improved_subtitles_data.sort(key=lambda x: x[1])

    # For each example, find the corresponding split subtitles
    print("\n📊 BEFORE/AFTER EXAMPLES OF SPLIT SUBTITLES 📊")
    print("═" * 80)

    for num, start, end, text in examples:
        print(f"\n🔹 ORIGINAL #{num}: {start} --> {end} ({len(text)} chars)")
        print(f"  {text}")

        # Get the time range of the original subtitle
        start_ms = time_to_ms(start)
        end_ms = time_to_ms(end)

        # Find corresponding split subtitles in improved file
        # We'll look for any subtitle that overlaps with the original time range
        split_subtitles = []

        for imp_num, imp_start_ms, imp_end_ms, imp_start, imp_end, imp_text in improved_subtitles_data:
            # Check if this subtitle overlaps with the original time range
            # A subtitle overlaps if:
            # 1. It starts during the original subtitle
            # 2. It ends during the original subtitle
            # 3. It completely contains the original subtitle
            if (imp_start_ms >= start_ms and imp_start_ms <= end_ms) or \
               (imp_end_ms >= start_ms and imp_end_ms <= end_ms) or \
               (imp_start_ms <= start_ms and imp_end_ms >= end_ms):
                split_subtitles.append((imp_start, imp_end, imp_text))

        # If we didn't find any overlapping subtitles, try a broader search
        # Look for subtitles that are close to the original time range
        if not split_subtitles:
            # Look for subtitles within a 5-second window before and after
            window_start_ms = max(0, start_ms - 5000)
            window_end_ms = end_ms + 5000

            for imp_num, imp_start_ms, imp_end_ms, imp_start, imp_end, imp_text in improved_subtitles_data:
                if (imp_start_ms >= window_start_ms and imp_start_ms <= window_end_ms) or \
                   (imp_end_ms >= window_start_ms and imp_end_ms <= window_end_ms):
                    split_subtitles.append((imp_start, imp_end, imp_text))

        # Sort by start time
        split_subtitles.sort(key=lambda x: time_to_ms(x[0]))

        # Print split subtitles
        print(f"\n🔸 SPLIT INTO {len(split_subtitles)} PARTS:")
        for i, (s_start, s_end, s_text) in enumerate(split_subtitles, 1):
            print(f"  {i}. {s_start} --> {s_end} ({len(s_text)} chars)")
            print(f"     {s_text}")

        print("─" * 80)


def main():
    if len(sys.argv) < 2:
        print(
            "Usage: python apply_subtitle_splits.py <input_srt> [output_srt] [max_length]")
        print("       python apply_subtitle_splits.py --compare <original_srt> <split_srt>")
        print(
            "       python apply_subtitle_splits.py --examples <original_srt> <split_srt> [num_examples]")
        sys.exit(1)

    # Check if we're in compare mode
    if sys.argv[1] == "--compare" and len(sys.argv) == 4:
        compare_files(sys.argv[2], sys.argv[3])
        return

    # Check if we're in examples mode
    if sys.argv[1] == "--examples" and len(sys.argv) >= 4:
        num_examples = int(sys.argv[4]) if len(sys.argv) > 4 else 5
        show_examples(sys.argv[2], sys.argv[3], num_examples)
        return

    input_file = sys.argv[1]

    # Default output file name if not provided
    if len(sys.argv) >= 3 and not sys.argv[2].startswith("--"):
        output_file = sys.argv[2]
    else:
        base, ext = os.path.splitext(input_file)
        output_file = f"{base}_improved{ext}"

    # Get max_length if provided
    max_length = 0
    for i in range(2, len(sys.argv)):
        if sys.argv[i].startswith("--max-length="):
            try:
                max_length = int(sys.argv[i].split("=")[1])
            except (IndexError, ValueError):
                print("Invalid max length format. Use --max-length=N")
                sys.exit(1)

    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        sys.exit(1)

    original_count, new_count = process_srt_file(
        input_file, output_file, max_length)

    print(f"✅ Processed {original_count} original subtitles")
    print(f"✅ Created {new_count} new subtitles")
    print(f"✅ Output saved to {output_file}")

    # Automatically compare the files
    print("\n--- Statistics Comparison ---")
    compare_files(input_file, output_file)

    # Show examples of split subtitles
    show_examples(input_file, output_file, 3)


if __name__ == "__main__":
    main()
