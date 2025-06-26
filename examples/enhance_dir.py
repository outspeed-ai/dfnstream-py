"""
Batch enhance all WAV files in a directory using DeepFilterNet.
"""

import numpy as np
import time
import sys
import wave
import os
import argparse
import glob
from pathlib import Path
from dfnstream_py import DeepFilterNetStreamingONNX


def load_wav_file(filepath):
    """Load audio from WAV file and return as float32 array."""
    try:
        with wave.open(filepath, 'rb') as wav_file:
            # Get audio parameters
            sample_rate = wav_file.getframerate()
            n_channels = wav_file.getnchannels()
            n_frames = wav_file.getnframes()
            sample_width = wav_file.getsampwidth()
            
            # Read all frames
            frames = wav_file.readframes(n_frames)
            
            # Convert to numpy array based on sample width
            if sample_width == 1:  # 8-bit
                audio_data = np.frombuffer(frames, dtype=np.uint8)
                audio_data = (audio_data.astype(np.float32) - 128) / 128.0
            elif sample_width == 2:  # 16-bit
                audio_data = np.frombuffer(frames, dtype=np.int16)
                audio_data = audio_data.astype(np.float32) / 32768.0
            elif sample_width == 4:  # 32-bit
                audio_data = np.frombuffer(frames, dtype=np.int32)
                audio_data = audio_data.astype(np.float32) / 2147483648.0
            else:
                raise ValueError(f"Unsupported sample width: {sample_width}")
            
            # Handle multi-channel audio (take first channel only)
            if n_channels > 1:
                audio_data = audio_data.reshape(-1, n_channels)
                audio_data = audio_data[:, 0]  # Take first channel
            
            return audio_data, sample_rate
            
    except Exception as e:
        raise RuntimeError(f"Error loading WAV file: {e}")


def save_wav_file(filepath, audio_data, sample_rate):
    """Save audio data to WAV file."""
    try:
        # Ensure audio is in valid range
        audio_data = np.clip(audio_data, -1.0, 1.0)
        
        # Convert to 16-bit PCM
        audio_int16 = (audio_data * 32767).astype(np.int16)
        
        with wave.open(filepath, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
            
    except Exception as e:
        raise RuntimeError(f"Error saving WAV file: {e}")


def resample_audio(audio, original_rate, target_rate):
    """Simple resampling using linear interpolation."""
    if original_rate == target_rate:
        return audio
    
    # Calculate resampling ratio
    ratio = target_rate / original_rate
    
    # Create new time indices
    original_length = len(audio)
    new_length = int(original_length * ratio)
    
    # Linear interpolation
    old_indices = np.arange(original_length)
    new_indices = np.linspace(0, original_length - 1, new_length)
    
    resampled_audio = np.interp(new_indices, old_indices, audio)
    
    return resampled_audio.astype(np.float32)


def process_wav_file(input_file, output_file, processor):
    """Process a single WAV file using streaming audio denoising."""
    try:
        # Load audio file
        audio_data, sample_rate = load_wav_file(input_file)
        
        # Resample if necessary
        if sample_rate != processor.sample_rate:
            audio_data = resample_audio(audio_data, sample_rate, processor.sample_rate)
            sample_rate = processor.sample_rate
        
        # Process in streaming fashion
        chunk_size = processor.frame_length * 4  # Process 4 frames at a time
        processed_chunks = []
        
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            processed_chunk = processor.process_chunk(chunk)
            processed_chunks.append(processed_chunk)
        
        # Combine processed audio
        processed_audio = np.concatenate(processed_chunks) if processed_chunks else np.array([])
        
        # Finalize processing to get remaining delayed samples
        final_samples = processor.finalize()
        if len(final_samples) > 0:
            processed_audio = np.concatenate([processed_audio, final_samples])
        
        # Save processed audio
        save_wav_file(output_file, processed_audio, processor.sample_rate)
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing {input_file}: {e}")
        return False


def enhance_directory(input_dir):
    """Enhance all WAV files in a directory."""
    print("🎵 DeepFilterNet Batch Directory Processing")
    print("=" * 50)
    
    # Create outputs directory in current working directory
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)
    print(f"📁 Output directory: {output_dir.absolute()}")
    
    # Find all .wav files in input directory
    input_path = Path(input_dir)
    wav_files = list(input_path.glob("*.wav")) + list(input_path.glob("*.WAV"))
    
    if not wav_files:
        print(f"❌ No WAV files found in {input_dir}")
        return False
    
    print(f"🔍 Found {len(wav_files)} WAV files to process")
    
    try:
        # Initialize processor once for all files
        print("\n🤖 Initializing DeepFilterNet...")
        processor = DeepFilterNetStreamingONNX(
            model_path=None,  # Use default model path
            atten_lim=None,  # No attenuation limit (full noise reduction)
            log_level="warn", 
            compensate_delay=True,
            post_filter_beta=0.02
        )
        
        print("✓ Model loaded successfully")
        print(f"✓ Frame length: {processor.frame_length} samples")
        print(f"✓ Expected sample rate: {processor.sample_rate} Hz")
        
        # Process each file
        successful_files = 0
        total_processing_time = 0
        
        for i, input_file in enumerate(wav_files, 1):
            # Generate output filename
            output_filename = f"{input_file.stem}_enhanced.wav"
            output_file = output_dir / output_filename
            
            print(f"\n[{i}/{len(wav_files)}] Processing: {input_file.name}")
            print(f"                    → {output_filename}")
            
            start_time = time.time()
            success = process_wav_file(str(input_file), str(output_file), processor)
            processing_time = time.time() - start_time
            total_processing_time += processing_time
            
            if success:
                successful_files += 1
                print(f"✓ Completed in {processing_time:.2f}s")
            else:
                print(f"❌ Failed after {processing_time:.2f}s")
        
        # Cleanup
        processor.close()
        
        # Results summary
        print(f"\n📊 Processing Summary:")
        print(f"✓ Successfully processed: {successful_files}/{len(wav_files)} files")
        print(f"✓ Total processing time: {total_processing_time:.2f}s")
        print(f"✓ Average time per file: {total_processing_time/len(wav_files):.2f}s")
        print(f"✓ Output directory: {output_dir.absolute()}")
        
        return successful_files == len(wav_files)
        
    except Exception as e:
        print(f"❌ Error during batch processing: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Batch enhance all WAV files in a directory")
    parser.add_argument("input_dir", help="Path to directory containing WAV files")
    
    args = parser.parse_args()
    input_dir = args.input_dir
    
    print("DeepFilterNet Batch Directory Processing")
    print("This tool enhances all WAV files in a directory.\n")
    
    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"❌ Input directory not found: {input_dir}")
        return False
    
    if not os.path.isdir(input_dir):
        print(f"❌ Path is not a directory: {input_dir}")
        return False
    
    success = enhance_directory(input_dir)
    
    if success:
        print("\n🎉 Batch processing completed successfully!")
    else:
        print("\n⚠️  Batch processing completed with some errors.")
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)