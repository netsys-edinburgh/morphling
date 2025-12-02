#!/usr/bin/env python3
"""
Performance log analyzer - parses perf.log and provides statistics
"""

import sys
import csv
from collections import defaultdict
from statistics import mean, median, stdev

def analyze_perf_log(log_file):
    """Analyze performance log file"""
    
    if not log_file:
        print("Error: No log file specified")
        print("Usage: python3 analyze_perf.py <log_file>")
        return
    
    try:
        with open(log_file, 'r') as f:
            reader = csv.DictReader(f)
            
            # Collect statistics by device and direction
            stats = defaultdict(lambda: {
                'upload': [],
                'download': [],
                'packet_durations': [],
                'timestamps': []
            })
            
            total_rows = 0
            for row in reader:
                total_rows += 1
                device_id = row['device_id']
                direction = row['direction'].lower()
                throughput = float(row['throughput_b_s'])
                packet_duration = int(row['packet_duration_us'])
                
                stats[device_id][direction].append(throughput)
                stats[device_id]['packet_durations'].append(packet_duration)
                stats[device_id]['timestamps'].append(int(row['timestamp_us']))
            
            if total_rows == 0:
                print("No data found in log file")
                return
            
            # Print summary statistics
            print(f"\n{'='*80}")
            print(f"Performance Log Analysis: {log_file}")
            print(f"{'='*80}\n")
            print(f"Total entries: {total_rows}\n")
            
            for device_id in sorted(stats.keys()):
                device_stats = stats[device_id]
                print(f"\nDevice {device_id}:")
                print(f"  {'-'*60}")
                
                # Upload statistics
                if device_stats['upload']:
                    upload_tp = device_stats['upload']
                    print(f"  Upload Throughput (B/s):")
                    print(f"    Count: {len(upload_tp)}")
                    print(f"    Min: {min(upload_tp):.2f}")
                    print(f"    Max: {max(upload_tp):.2f}")
                    print(f"    Mean: {mean(upload_tp):.2f}")
                    print(f"    Median: {median(upload_tp):.2f}")
                    if len(upload_tp) > 1:
                        print(f"    StdDev: {stdev(upload_tp):.2f}")
                
                # Download statistics
                if device_stats['download']:
                    download_tp = device_stats['download']
                    print(f"  Download Throughput (B/s):")
                    print(f"    Count: {len(download_tp)}")
                    print(f"    Min: {min(download_tp):.2f}")
                    print(f"    Max: {max(download_tp):.2f}")
                    print(f"    Mean: {mean(download_tp):.2f}")
                    print(f"    Median: {median(download_tp):.2f}")
                    if len(download_tp) > 1:
                        print(f"    StdDev: {stdev(download_tp):.2f}")
                
                # Packet duration statistics
                if device_stats['packet_durations']:
                    durations = device_stats['packet_durations']
                    print(f"  Packet Duration (microseconds):")
                    print(f"    Count: {len(durations)}")
                    print(f"    Min: {min(durations)}")
                    print(f"    Max: {max(durations)}")
                    print(f"    Mean: {mean(durations):.2f}")
                    print(f"    Median: {median(durations):.2f}")
                    if len(durations) > 1:
                        print(f"    StdDev: {stdev(durations):.2f}")
            
            # Overall statistics
            print(f"\n{'='*80}")
            print("Overall Statistics:")
            print(f"{'='*80}")
            all_upload = [tp for device_stats in stats.values() for tp in device_stats['upload']]
            all_download = [tp for device_stats in stats.values() for tp in device_stats['download']]
            
            if all_upload:
                print(f"\nTotal Upload Throughput (B/s):")
                print(f"  Count: {len(all_upload)}")
                print(f"  Min: {min(all_upload):.2f}")
                print(f"  Max: {max(all_upload):.2f}")
                print(f"  Mean: {mean(all_upload):.2f}")
                print(f"  Median: {median(all_upload):.2f}")
                if len(all_upload) > 1:
                    print(f"  StdDev: {stdev(all_upload):.2f}")
            
            if all_download:
                print(f"\nTotal Download Throughput (B/s):")
                print(f"  Count: {len(all_download)}")
                print(f"  Min: {min(all_download):.2f}")
                print(f"  Max: {max(all_download):.2f}")
                print(f"  Mean: {mean(all_download):.2f}")
                print(f"  Median: {median(all_download):.2f}")
                if len(all_download) > 1:
                    print(f"  StdDev: {stdev(all_download):.2f}")
            
            print(f"\n{'='*80}\n")
    
    except FileNotFoundError:
        print(f"Error: Log file not found: {log_file}")
    except Exception as e:
        print(f"Error analyzing log file: {e}")

if __name__ == "__main__":
    log_file = sys.argv[1] if len(sys.argv) > 1 else "./perf.log"
    analyze_perf_log(log_file)
