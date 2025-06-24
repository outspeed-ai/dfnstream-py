"""
Example of enhancing an audio file incrementally using DeepFilterNet.
"""

import numpy as np
import time
import sys
import wave
import os
import argparse
from dfnstream_py import DeepFilterNetStreaming


def load_wav_file(filepath):
    """Load audio from WAV file and return as float32 array."""
    try:
        with wave.open(filepath, 'rb') as wav_file:
            # Get audio parameters
            sample_rate = wav_file.getframerate()
            n_channels = wav_file.getnchannels()
            n_frames = wav_file.getnframes()
            sample_width = wav_file.getsampwidth()
            
            print(f"Input file: {filepath}")
            print(f"Sample rate: {sample_rate} Hz")
            print(f"Channels: {n_channels}")
            print(f"Duration: {n_frames/sample_rate:.2f} seconds")
            print(f"Sample width: {sample_width} bytes")
            
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
                print(f"Multi-channel audio detected, using first channel only")
            
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
            
        print(f"Saved denoised audio to: {filepath}")
        
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
    
    print(f"Resampled from {original_rate} Hz to {target_rate} Hz")
    print(f"Length changed from {original_length} to {new_length} samples")
    
    return resampled_audio.astype(np.float32)


def process_wav_file(input_file):
    """Process a WAV file using streaming audio denoising."""
    print("🎵 DeepFilterNet Streaming WAV File Processing")
    print("=" * 50)
    
    input_basename = os.path.basename(input_file)
    base_name = os.path.splitext(input_basename)[0]
    output_file = f"{base_name}_denoised.wav"
    
    try:
        # Check if input file exists
        if not os.path.exists(input_file):
            print(f"❌ Input file not found: {input_file}")
            return False
        
        # Load audio file
        print("\n📁 Loading input audio file...")
        audio_data, sample_rate = load_wav_file(input_file)
        
        print("\n🤖 Initializing DeepFilterNet...")
        processor = DeepFilterNetStreaming(
            model_path=None,
            atten_lim=None,
            log_level="warn", 
            compensate_delay=False,
            post_filter_beta=0.025
        )
        
        print("✓ Model loaded successfully")
        print(f"✓ Frame length: {processor.frame_length} samples")
        print(f"✓ Expected sample rate: {processor.sample_rate} Hz")
        print(f"✓ Frame duration: {processor.frame_length / processor.sample_rate * 1000:.1f} ms")
        
        # Resample if necessary
        if sample_rate != processor.sample_rate:
            print(f"\n🔄 Resampling audio...")
            audio_data = resample_audio(audio_data, sample_rate, processor.sample_rate)
            sample_rate = processor.sample_rate
        
        print(f"\n🔄 Processing audio in streaming chunks...")
        chunk_size = processor.frame_length * 4  # Process 4 frames at a time
        processed_chunks = []
        total_processing_time = 0
        
        num_chunks = (len(audio_data) + chunk_size - 1) // chunk_size
        print(f"✓ Processing {num_chunks} chunks of {chunk_size} samples each")
        
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i + chunk_size]
            
            # Measure processing time for this chunk
            start_time = time.time()
            processed_chunk = processor.process_chunk(chunk)
            processing_time = time.time() - start_time
            
            total_processing_time += processing_time
            processed_chunks.append(processed_chunk)
            
            # Show progress
            chunk_num = i // chunk_size + 1
            audio_duration = len(chunk) / processor.sample_rate
            rt_factor = audio_duration / processing_time if processing_time > 0 else float('inf')
            
            if chunk_num % 10 == 0 or chunk_num == num_chunks:  # Show every 10th chunk
                print(f"  Chunk {chunk_num:3d}/{num_chunks}: {processing_time*1000:5.1f}ms "
                      f"(RT factor: {rt_factor:5.1f}x)")
        
        # Combine processed audio
        processed_audio = np.concatenate(processed_chunks) if processed_chunks else np.array([])
        
        # Finalize processing to get remaining delayed samples
        final_samples = processor.finalize()
        if len(final_samples) > 0:
            processed_audio = np.concatenate([processed_audio, final_samples])
        
        # Calculate results
        audio_duration = len(audio_data) / processor.sample_rate
        overall_rt_factor = audio_duration / total_processing_time
        
        # With delay compensation, the output length should match the input
        print(f"Input length: {len(audio_data)} samples, Output length: {len(processed_audio)} samples")
        
        # Results
        print(f"\n📊 Processing Results:")
        print(f"✓ Audio duration: {audio_duration:.2f}s")
        print(f"✓ Total processing time: {total_processing_time:.3f}s")
        print(f"✓ Overall real-time factor: {overall_rt_factor:.2f}x")
        print(f"✓ {'Real-time capable!' if overall_rt_factor >= 1.0 else 'Not real-time on this system'}")
        
        # Save processed audio
        print(f"\n💾 Saving denoised audio...")
        save_wav_file(output_file, processed_audio, processor.sample_rate)
        
        # Test individual frame processing
        print(f"\n🔬 Testing individual frame processing...")
        test_frame = audio_data[:processor.frame_length]
        _, local_snr = processor.process_frame(test_frame)
        
        print(f"✓ Single frame processed successfully")
        print(f"✓ Local SNR estimate: {local_snr:.2f} dB")
        
        # Cleanup
        processor.close()
        print(f"\n✅ Processing completed successfully!")
        print(f"Input file: {input_file}")
        print(f"Output file: {output_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during processing: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepFilterNet Streaming Audio Denoising")
    parser.add_argument("audio_file", help="Path to the input audio file (.wav)")
    
    args = parser.parse_args()
    input_file = args.audio_file
    
    print("DeepFilterNet Streaming")
    
    success = process_wav_file(input_file)
    
    if success:
        print("\n🎉 WAV file processing completed successfully!")
    else:
        print("\n❌ WAV file processing failed!")