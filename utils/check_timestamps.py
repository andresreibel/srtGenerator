import re
from collections import Counter


def srt_time_to_ms(srt_time):
    h, m, s = srt_time.split(':')
    s, ms = s.split(',')
    return (int(h) * 3600 + int(m) * 60 + int(s)) * 1000 + int(ms)


def ms_to_srt_time(ms):
    h = ms // 3600000
    ms %= 3600000
    m = ms // 60000
    ms %= 60000
    s = ms // 1000
    ms %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


with open('/Users/master/code/srtGenerator/output_srt_files/mvp1_en.srt', 'r') as f:
    content = f.read()

timestamps = re.findall(
    r'(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})', content)
invalid = []
overlaps = []
durations = []
short_subtitles = []
min_duration_ms = 500  # 0.5 seconds

# Check for empty subtitles
empty_subtitles = re.findall(
    r'\d+\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n\n', content)
empty_count = len(empty_subtitles)

# Extract subtitle text for analysis
subtitle_blocks = re.split(r'\n\s*\n', content.strip())
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
            overlaps.append((i, i+1, end, next_start, end_ms - next_start_ms))

# Calculate gaps between subtitles
gaps = []
for i in range(len(timestamps) - 1):
    current_end_ms = srt_time_to_ms(timestamps[i][1])
    next_start_ms = srt_time_to_ms(timestamps[i+1][0])
    gap_ms = next_start_ms - current_end_ms
    gaps.append(gap_ms)

if invalid:
    print('Invalid timestamps found:')
    for i, start, end, diff in invalid:
        print(f'Subtitle #{i}: {start} --> {end} (diff: {diff}ms)')
else:
    print('No invalid timestamps found')

if overlaps:
    print('\nOverlapping timestamps found:')
    for i, j, end, next_start, diff in overlaps:
        print(
            f'Subtitle #{i} overlaps with #{j} by {diff}ms: {end} > {next_start}')
else:
    print('\nNo overlapping timestamps found')

if short_subtitles:
    print(
        f'\nFound {len(short_subtitles)} subtitles shorter than {min_duration_ms}ms:')
    for i, start, end, duration in short_subtitles[:10]:  # Show first 10
        print(f'Subtitle #{i}: {start} --> {end} (duration: {duration}ms)')
    if len(short_subtitles) > 10:
        print(f'... and {len(short_subtitles) - 10} more')
else:
    print(f'\nNo subtitles shorter than {min_duration_ms}ms found')

print(f'\nFound {empty_count} empty subtitles')

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

    print('\nDuration statistics:')
    print(f'Average duration: {avg_duration:.2f}ms ({avg_duration/1000:.2f}s)')
    print(f'Minimum duration: {min_duration}ms ({min_duration/1000:.2f}s)')
    print(f'Maximum duration: {max_duration}ms ({max_duration/1000:.2f}s)')

    print('\nDuration distribution:')
    for range_name, count in duration_ranges.items():
        percentage = (count / len(durations)) * 100
        print(f'{range_name}: {count} subtitles ({percentage:.1f}%)')

# Gap statistics
if gaps:
    avg_gap = sum(gaps) / len(gaps)
    min_gap = min(gaps)
    max_gap = max(gaps)

    # Count negative gaps (overlaps)
    negative_gaps = sum(1 for g in gaps if g < 0)

    print('\nGap statistics:')
    print(f'Average gap: {avg_gap:.2f}ms ({avg_gap/1000:.2f}s)')
    print(f'Minimum gap: {min_gap}ms ({min_gap/1000:.2f}s)')
    print(f'Maximum gap: {max_gap}ms ({max_gap/1000:.2f}s)')
    print(f'Negative gaps (overlaps): {negative_gaps}')

# Text statistics
if subtitle_texts:
    text_lengths = [len(text) for text in subtitle_texts]
    avg_length = sum(text_lengths) / len(text_lengths)
    max_length = max(text_lengths)

    print('\nText statistics:')
    print(f'Average text length: {avg_length:.1f} characters')
    print(f'Maximum text length: {max_length} characters')
