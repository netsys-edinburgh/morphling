#!/usr/bin/env python3
"""
Performance log plotter - visualizes perf.log data with multiple charts
"""

import sys
import csv
import argparse
from collections import defaultdict
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

def parse_perf_log(log_file):
    """Parse performance log file and organize data"""
    
    # Data structure: {device_id: {direction: [(timestamp, bytes, tp, duration), ...]}}
    device_data = defaultdict(lambda: {'UPLOAD': [], 'DOWNLOAD': []})
    server_data = {'UPLOAD': [], 'DOWNLOAD': []}
    all_timestamps = []
    
    try:
        with open(log_file, 'r') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                timestamp_us = int(row['timestamp_us'])
                device_id = int(row['device_id'])
                direction = row['direction'].upper()
                bytes_val = int(row['bytes'])
                throughput = float(row['throughput_b_s'])
                epoch_start = int(row['epoch_start_us'])
                epoch_end = int(row['epoch_end_us'])
                duration = int(row['packet_duration_us'])
                
                device_data[device_id][direction].append({
                    'timestamp_us': timestamp_us,
                    'bytes': bytes_val,
                    'throughput': throughput,
                    'duration': duration
                })
                
                all_timestamps.append(timestamp_us)
        
        if not all_timestamps:
            print("Error: No data found in log file")
            return None
        
        # Calculate server total (sum of all device throughputs per timestamp)
        min_ts = min(all_timestamps)
        max_ts = max(all_timestamps)
        
        # Group by direction and timestamp
        upload_by_ts = defaultdict(float)
        download_by_ts = defaultdict(float)
        
        for device_id, directions in device_data.items():
            for ts_data in directions['UPLOAD']:
                upload_by_ts[ts_data['timestamp_us']] += ts_data['throughput']
            for ts_data in directions['DOWNLOAD']:
                download_by_ts[ts_data['timestamp_us']] += ts_data['throughput']
        
        server_data['UPLOAD'] = sorted(upload_by_ts.items())
        server_data['DOWNLOAD'] = sorted(download_by_ts.items())
        
        return {
            'device_data': device_data,
            'server_data': server_data,
            'min_ts': min_ts,
            'max_ts': max_ts,
            'num_devices': len(device_data)
        }
    
    except FileNotFoundError:
        print(f"Error: Log file not found: {log_file}")
        return None
    except Exception as e:
        print(f"Error parsing log file: {e}")
        return None

def plot_all_metrics(data, output_prefix="./perf"):
    """Generate all visualization plots"""
    
    if data is None:
        return
    
    device_data = data['device_data']
    server_data = data['server_data']
    min_ts = data['min_ts']
    max_ts = data['max_ts']
    num_devices = data['num_devices']
    
    # Normalize timestamps to seconds from start
    def normalize_ts(ts_us):
        return (ts_us - min_ts) / 1_000_000.0
    
    # 1. Server Total Throughput (Upload + Download)
    print("Generating plot 1: Server Total Throughput...")
    fig, ax = plt.subplots(figsize=(14, 6))
    
    if server_data['UPLOAD']:
        upload_ts, upload_tp = zip(*server_data['UPLOAD'])
        upload_ts = [normalize_ts(ts) for ts in upload_ts]
        ax.plot(upload_ts, upload_tp, 'o-', label='Upload', linewidth=2, markersize=6, alpha=0.7)
    
    if server_data['DOWNLOAD']:
        download_ts, download_tp = zip(*server_data['DOWNLOAD'])
        download_ts = [normalize_ts(ts) for ts in download_ts]
        ax.plot(download_ts, download_tp, 's-', label='Download', linewidth=2, markersize=6, alpha=0.7)
    
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Throughput (B/s)', fontsize=12)
    ax.set_title('Server Total Throughput Over Time', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_server_total.png", dpi=150)
    plt.close()
    print(f"✓ Saved: {output_prefix}_server_total.png")
    
    # 2. Per-Device Upload Throughput
    print("Generating plot 2: Per-Device Upload Throughput...")
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = plt.cm.tab20(np.linspace(0, 1, num_devices))
    
    for idx, device_id in enumerate(sorted(device_data.keys())):
        upload_data = device_data[device_id]['UPLOAD']
        if upload_data:
            timestamps = [normalize_ts(d['timestamp_us']) for d in upload_data]
            throughputs = [d['throughput'] for d in upload_data]
            ax.plot(timestamps, throughputs, 'o-', label=f'Device {device_id}',
                   linewidth=1.5, markersize=4, alpha=0.7, color=colors[idx])
    
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Upload Throughput (B/s)', fontsize=12)
    ax.set_title(f'Per-Device Upload Throughput ({num_devices} devices)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, ncol=2, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_upload_per_device.png", dpi=150)
    plt.close()
    print(f"✓ Saved: {output_prefix}_upload_per_device.png")
    
    # 3. Per-Device Download Throughput
    print("Generating plot 3: Per-Device Download Throughput...")
    fig, ax = plt.subplots(figsize=(14, 8))
    
    for idx, device_id in enumerate(sorted(device_data.keys())):
        download_data = device_data[device_id]['DOWNLOAD']
        if download_data:
            timestamps = [normalize_ts(d['timestamp_us']) for d in download_data]
            throughputs = [d['throughput'] for d in download_data]
            ax.plot(timestamps, throughputs, 's-', label=f'Device {device_id}',
                   linewidth=1.5, markersize=4, alpha=0.7, color=colors[idx])
    
    ax.set_xlabel('Time (seconds)', fontsize=12)
    ax.set_ylabel('Download Throughput (B/s)', fontsize=12)
    ax.set_title(f'Per-Device Download Throughput ({num_devices} devices)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9, ncol=2, loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_download_per_device.png", dpi=150)
    plt.close()
    print(f"✓ Saved: {output_prefix}_download_per_device.png")
    
    # 4. Box Plot - Upload Distribution by Device
    print("Generating plot 4: Upload Throughput Distribution...")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    upload_data_by_device = []
    device_labels = []
    
    for device_id in sorted(device_data.keys()):
        upload_list = [d['throughput'] for d in device_data[device_id]['UPLOAD']]
        if upload_list:
            upload_data_by_device.append(upload_list)
            device_labels.append(f"Dev {device_id}")
    
    if upload_data_by_device:
        bp = ax.boxplot(upload_data_by_device, labels=device_labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
            patch.set_facecolor(color)
        
        ax.set_ylabel('Upload Throughput (B/s)', fontsize=12)
        ax.set_xlabel('Device', fontsize=12)
        ax.set_title('Upload Throughput Distribution by Device', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_upload_distribution.png", dpi=150)
        plt.close()
        print(f"✓ Saved: {output_prefix}_upload_distribution.png")
    
    # 5. Box Plot - Download Distribution by Device (Black & White)
    print("Generating plot 5: Download Throughput Distribution...")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    download_data_by_device = []
    device_labels = []
    
    for device_id in sorted(device_data.keys()):
        download_list = [d['throughput'] for d in device_data[device_id]['DOWNLOAD']]
        if download_list:
            download_data_by_device.append(download_list)
            device_labels.append(f"Dev {device_id}")
    
    if download_data_by_device:
        bp = ax.boxplot(download_data_by_device, labels=device_labels, patch_artist=True)
        
        # Black and white color scheme
        gray_colors = plt.cm.Greys(np.linspace(0.3, 0.7, len(bp['boxes'])))
        
        for patch, color in zip(bp['boxes'], gray_colors):
            patch.set_facecolor(color)
            patch.set_edgecolor('black')
            patch.set_linewidth(1.5)
        
        # Style whiskers, caps, and medians
        for whisker in bp['whiskers']:
            whisker.set(color='black', linewidth=1.5)
        for cap in bp['caps']:
            cap.set(color='black', linewidth=1.5)
        for median in bp['medians']:
            median.set(color='darkred', linewidth=2)
        
        ax.set_ylabel('Download Throughput (B/s)', fontsize=12)
        ax.set_xlabel('Device', fontsize=12)
        ax.set_title('Download Throughput Distribution by Device', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_download_distribution.png", dpi=150)
        plt.close()
        print(f"✓ Saved: {output_prefix}_download_distribution.png")
    
    # 6. Stacked Area Chart - Server Total Breakdown
    print("Generating plot 6: Server Total Breakdown by Device...")
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Collect upload data for all devices at each timestamp
    all_upload_timestamps = set()
    for device_id, directions in device_data.items():
        for d in directions['UPLOAD']:
            all_upload_timestamps.add(d['timestamp_us'])
    
    if all_upload_timestamps:
        all_upload_timestamps = sorted(all_upload_timestamps)
        upload_by_device = {}
        
        for device_id in sorted(device_data.keys()):
            device_upload = {d['timestamp_us']: d['throughput'] 
                           for d in device_data[device_id]['UPLOAD']}
            upload_by_device[device_id] = [device_upload.get(ts, 0) for ts in all_upload_timestamps]
        
        normalized_ts = [normalize_ts(ts) for ts in all_upload_timestamps]
        
        ax.stackplot(normalized_ts, 
                    *[upload_by_device[did] for did in sorted(device_data.keys())],
                    labels=[f'Device {did}' for did in sorted(device_data.keys())],
                    alpha=0.8)
        
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('Throughput (B/s)', fontsize=12)
        ax.set_title('Server Total Upload - Breakdown by Device', fontsize=14, fontweight='bold')
        ax.legend(fontsize=9, ncol=2, loc='upper left')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{output_prefix}_server_upload_breakdown.png", dpi=150)
        plt.close()
        print(f"✓ Saved: {output_prefix}_server_upload_breakdown.png")
    
    # 7. Summary Statistics Table
    print("Generating plot 7: Summary Statistics...")
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.axis('tight')
    ax.axis('off')
    
    summary_data = []
    for device_id in sorted(device_data.keys()):
        upload_tps = [d['throughput'] for d in device_data[device_id]['UPLOAD']]
        download_tps = [d['throughput'] for d in device_data[device_id]['DOWNLOAD']]
        
        upload_avg = np.mean(upload_tps) if upload_tps else 0
        download_avg = np.mean(download_tps) if download_tps else 0
        total_avg = upload_avg + download_avg
        
        summary_data.append([
            f'Device {device_id}',
            f'{len(upload_tps)}',
            f'{upload_avg:.2f}',
            f'{np.max(upload_tps):.2f}' if upload_tps else '0.00',
            f'{len(download_tps)}',
            f'{download_avg:.2f}',
            f'{np.max(download_tps):.2f}' if download_tps else '0.00',
            f'{total_avg:.2f}'
        ])
    
    columns = ['Device', 'Upload\nCount', 'Upload\nAvg (B/s)', 'Upload\nMax (B/s)',
               'Download\nCount', 'Download\nAvg (B/s)', 'Download\nMax (B/s)', 'Total\nAvg (B/s)']
    
    table = ax.table(cellText=summary_data, colLabels=columns, cellLoc='center',
                    loc='center', bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Color header
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(summary_data) + 1):
        color = '#f0f0f0' if i % 2 == 0 else 'white'
        for j in range(len(columns)):
            table[(i, j)].set_facecolor(color)
    
    plt.title('Performance Summary by Device', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(f"{output_prefix}_summary_table.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved: {output_prefix}_summary_table.png")
    
    print(f"\n✅ All plots generated successfully!")
    print(f"   Output prefix: {output_prefix}")

def main():
    parser = argparse.ArgumentParser(description='Plot performance metrics from perf.log')
    parser.add_argument('--log', default='./perf.log', help='Path to perf.log file')
    parser.add_argument('--output', default='./perf', help='Output file prefix for plots')
    args = parser.parse_args()
    
    print(f"Parsing performance log: {args.log}")
    data = parse_perf_log(args.log)
    
    if data:
        print(f"✓ Found {data['num_devices']} devices")
        print(f"✓ Time range: {(data['max_ts'] - data['min_ts'])/1_000_000:.2f} seconds")
        print()
        plot_all_metrics(data, args.output)

if __name__ == "__main__":
    main()
