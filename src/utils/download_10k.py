#!/usr/bin/env python3
"""
Simple MIMIC-CXR Image Downloader for 10k images
"""

import os
import subprocess
import time
from pathlib import Path
from tqdm import tqdm
import json

def download_10k_images():
    """Download 10,000 images with progress tracking."""
    username = "rahul11s"
    password = "[REDACTED]"
    num_images = 10000
    
    print("MIMIC-CXR Image Downloader - 10k Images")
    print("=" * 40)
    
    # Read image paths
    with open('selected_images.txt', 'r') as f:
        image_paths = [line.strip() for line in f.readlines() if line.strip()]
    
    # Take first 10k images
    image_paths = image_paths[:num_images]
    
    print(f"Downloading {len(image_paths)} images...")
    
    # Create base directory
    os.makedirs('files', exist_ok=True)
    
    # Set up progress bar
    pbar = tqdm(total=len(image_paths), desc="Downloading", unit="img")
    
    downloaded_count = 0
    failed_count = 0
    start_time = time.time()
    
    try:
        for i, image_path in enumerate(image_paths):
            # Create directory
            dir_path = os.path.dirname(image_path)
            os.makedirs(dir_path, exist_ok=True)
            
            # Skip if file already exists
            if os.path.exists(image_path) and os.path.getsize(image_path) > 0:
                downloaded_count += 1
                pbar.update(1)
                continue
            
            # Construct URL
            url = f"https://physionet.org/files/mimic-cxr-jpg/2.1.0/{image_path}"
            
            # Use wget command
            cmd = [
                'wget',
                '-q',  # Quiet mode
                '--timeout=30',
                '--tries=3',
                f'--user={username}',
                f'--password={password}',
                '-O', image_path,
                url
            ]
            
            try:
                result = subprocess.run(cmd, capture_output=True, timeout=60)
                
                if result.returncode == 0 and os.path.exists(image_path) and os.path.getsize(image_path) > 0:
                    downloaded_count += 1
                else:
                    failed_count += 1
                    
            except Exception as e:
                failed_count += 1
            
            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({
                'Downloaded': downloaded_count,
                'Failed': failed_count,
                'Success': f"{(downloaded_count/(downloaded_count+failed_count)*100):.1f}%" if (downloaded_count+failed_count) > 0 else "0%"
            })
            
            # Save checkpoint every 100 images
            if (i + 1) % 100 == 0:
                elapsed = time.time() - start_time
                print(f"\n📊 Checkpoint: {downloaded_count}/{len(image_paths)} downloaded ({elapsed/60:.1f} min elapsed)")
                
                # Save progress to file
                progress_data = {
                    'downloaded_count': downloaded_count,
                    'failed_count': failed_count,
                    'total_images': len(image_paths),
                    'elapsed_time': elapsed,
                    'timestamp': time.time()
                }
                
                with open('download_progress.json', 'w') as f:
                    json.dump(progress_data, f, indent=2)
    
    except KeyboardInterrupt:
        print(f"\n\nDownload interrupted!")
    
    finally:
        pbar.close()
        
        # Final save
        progress_data = {
            'downloaded_count': downloaded_count,
            'failed_count': failed_count,
            'total_images': len(image_paths),
            'elapsed_time': time.time() - start_time,
            'timestamp': time.time(),
            'completed': True
        }
        
        with open('download_progress.json', 'w') as f:
            json.dump(progress_data, f, indent=2)
    
    # Final statistics
    elapsed_time = time.time() - start_time
    success_rate = (downloaded_count / len(image_paths)) * 100 if len(image_paths) > 0 else 0
    
    print(f"\n🎉 Download completed!")
    print(f"📊 Final Statistics:")
    print(f"   • Downloaded: {downloaded_count:,} images")
    print(f"   • Failed: {failed_count:,} images")
    print(f"   • Success Rate: {success_rate:.1f}%")
    print(f"   • Total Time: {elapsed_time/60:.1f} minutes")
    print(f"   • Avg Time per Image: {elapsed_time/len(image_paths):.1f} seconds")
    
    # Update joined dataset
    print(f"\n🔄 Updating joined dataset...")
    try:
        subprocess.run(['python3', 'download_images.py'], check=True)
        print("✅ Dataset updated successfully!")
    except Exception as e:
        print(f"❌ Error updating dataset: {e}")
    
    print(f"\n✅ All done! Images saved in 'files/' directory")

if __name__ == "__main__":
    download_10k_images()
