#!/usr/bin/env python3
"""
Merge separate performance logs from Server and Devices into a single sorted log
Usage: python3 merge_perf_logs.py <log_dir> [output_file]
"""

import os
import sys
from pathlib import Path
from collections import defaultdict
import re

def parse_vtime_event(line):
    """Parse a VTIME event line and return (timestamp_us, line_content)"""
    parts = line.strip().split(',')
    if len(parts) >= 2:
        try:
            timestamp_us = int(parts[1])
            return (timestamp_us, line)
        except ValueError:
            return None
    return None

def parse_throughput_event(line):
    """Parse a throughput event line and return (timestamp_us, line_content)"""
    parts = line.strip().split(',')
    if len(parts) >= 1 and parts[0].isdigit():
        try:
            timestamp_us = int(parts[0])
            return (timestamp_us, line)
        except ValueError:
            return None
    return None

def merge_logs(log_dir, output_file=None):
    """
    Merge all perf_*.log files in log_dir into a single sorted output
    
    Args:
        log_dir: Directory containing perf_*.log files
        output_file: Output file path (default: ./perf_merged.log)
    """
    
    if output_file is None:
        output_file = os.path.join(os.path.dirname(log_dir) or '.', 'perf_merged.log')
    
    log_dir = Path(log_dir)
    if not log_dir.is_dir():
        print(f"Error: Directory not found: {log_dir}")
        return False
    
    # Find all perf_*.log files
    log_files = sorted(log_dir.glob('perf_*.log'))
    if not log_files:
        print(f"Warning: No perf_*.log files found in {log_dir}")
        return False
    
    print(f"Found {len(log_files)} log files:")
    for f in log_files:
        print(f"  - {f.name}")
    
    # Read and parse all events
    all_events = []
    header_written = False
    
    for log_file in log_files:
        print(f"\nProcessing {log_file.name}...")
        event_count = 0
        
        with open(log_file, 'r') as f:
            for line in f:
                line = line.rstrip('\n')
                
                # Skip empty lines
                if not line:
                    continue
                
                # Skip header lines (only write once)
                if line.startswith('timestamp_us,') or line.startswith('#'):
                    if not header_written and line.startswith('timestamp_us,'):
                        all_events.append((0, line))  # Header with timestamp 0
                        header_written = True
                    continue
                
                # Parse VTIME events
                if line.startswith('VTIME,'):
                    parsed = parse_vtime_event(line)
                    if parsed:
                        all_events.append(parsed)
                        event_count += 1
                
                # Parse throughput events
                elif line and line[0].isdigit():
                    parsed = parse_throughput_event(line)
                    if parsed:
                        all_events.append(parsed)
                        event_count += 1
        
        print(f"  Loaded {event_count} events from {log_file.name}")
    
    if not all_events:
        print("Error: No events found in any log files")
        return False
    
    # Sort by timestamp
    print("\nSorting events by timestamp...")
    all_events.sort(key=lambda x: x[0])
    
    # Write merged log
    print(f"\nWriting merged log to {output_file}...")
    with open(output_file, 'w') as f:
        for timestamp, line in all_events:
            f.write(line + '\n')
    
    # Print statistics
    vtime_count = sum(1 for _, line in all_events if line.startswith('VTIME,'))
    throughput_count = sum(1 for _, line in all_events if line[0].isdigit())
    
    print(f"\n" + "="*60)
    print(f"Merge completed successfully!")
    print(f"="*60)
    print(f"Output file: {output_file}")
    print(f"Total events: {len(all_events) - 1}")  # -1 for header
    print(f"  VTIME events: {vtime_count}")
    print(f"  Throughput events: {throughput_count}")
    print(f"="*60)
    
    return True

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 merge_perf_logs.py <log_dir> [output_file]")
        print("\nExample:")
        print("  python3 merge_perf_logs.py ./logs")
        print("  python3 merge_perf_logs.py ./logs ./perf_merged.log")
        sys.exit(1)
    
    log_dir = sys.argv[1]
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    success = merge_logs(log_dir, output_file)
    sys.exit(0 if success else 1)

if __name__ == '__main__':
    main()
