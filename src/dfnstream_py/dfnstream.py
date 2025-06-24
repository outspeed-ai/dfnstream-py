"""
DeepFilterNet Streaming implementation for Python (using C API bindings)
"""

import ctypes
import numpy as np
import os
import platform
from typing import Optional, Tuple


class DeepFilterNetStreaming:
    def __init__(self, model_path: str | None = None, atten_lim: Optional[float] = None, 
                 log_level: Optional[str] = None, compensate_delay: bool = True,
                 post_filter_beta: float = 0.0):
        """
        Initialize DeepFilterNet streaming processor.
        
        Args:
            model_path: Optional path to custom DeepFilterNet model (.tar.gz file)
            atten_lim: Attenuation limit in dB (default: None = no limit, full noise reduction)
            log_level: Optional logging level ('error', 'warn', 'info', 'debug', 'trace')
            compensate_delay: Whether to compensate for STFT and model delay (default: True)
            post_filter_beta: Post-filter beta parameter (default: 0.0 = disabled)
        """
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "models", "DeepFilterNet3_ll_onnx.tar.gz")
        
        self.model_path = model_path
        self.atten_lim = atten_lim
        self.log_level = log_level
        self.compensate_delay = compensate_delay
        self.post_filter_beta = post_filter_beta
        self._state_ptr = None
        self._frame_length = None
        self._delay_samples = 0
        self._partial_frame_buffer = np.array([], dtype=np.float32)
        self._output_delay_buffer = np.array([], dtype=np.float32)
        self._is_warmed_up = False
        self._pre_analysis_buffer = []
        self._pre_analysis_done = False
        
        # Load the shared library
        self._load_library()
        self._setup_function_signatures()
        
        # Initialize the model
        self._initialize_model()
    
    def _should_log(self) -> bool:
        """Check if logging should be enabled based on log level."""
        return self.log_level in ["info", "debug", "trace"]
    
    def _load_library(self):
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            lib_name = "libdf_arm64_darwin.dylib"
        elif platform.system() == "Linux" and platform.machine() == "x86_64":
            lib_name = "libdf_x86_64_linux.so"
        elif platform.system() == "Linux" and platform.machine() == "aarch64":
            lib_name = "libdf_aarch64_linux.so"
        else:
            raise RuntimeError(f"Unsupported platform: {platform.system()} {platform.machine()}")
        
        path = os.path.join(os.path.dirname(__file__), "lib", lib_name)
        
        self._lib = None
        try:
            self._lib = ctypes.CDLL(path)
            if self._should_log():
                print(f"Loaded DeepFilterNet library from: {path}")
        except OSError:
            raise RuntimeError(f"Could not load DeepFilterNet library from: {path}")
    
    def _setup_function_signatures(self):
        # df_create
        self._lib.df_create.argtypes = [
            ctypes.c_char_p,  # path
            ctypes.c_float,   # atten_lim
            ctypes.c_char_p   # log_level
        ]
        self._lib.df_create.restype = ctypes.c_void_p
        
        # df_get_frame_length
        self._lib.df_get_frame_length.argtypes = [ctypes.c_void_p]
        self._lib.df_get_frame_length.restype = ctypes.c_size_t
        
        # df_process_frame
        self._lib.df_process_frame.argtypes = [
            ctypes.c_void_p,          # state
            ctypes.POINTER(ctypes.c_float),  # input
            ctypes.POINTER(ctypes.c_float)   # output
        ]
        self._lib.df_process_frame.restype = ctypes.c_float
        
        # df_set_atten_lim
        self._lib.df_set_atten_lim.argtypes = [
            ctypes.c_void_p,  # state
            ctypes.c_float    # lim_db
        ]
        self._lib.df_set_atten_lim.restype = None
        
        # df_set_post_filter_beta
        self._lib.df_set_post_filter_beta.argtypes = [
            ctypes.c_void_p,  # state
            ctypes.c_float    # beta
        ]
        self._lib.df_set_post_filter_beta.restype = None
        
        # df_next_log_msg
        self._lib.df_next_log_msg.argtypes = [ctypes.c_void_p]
        self._lib.df_next_log_msg.restype = ctypes.c_char_p
        
        # df_free_log_msg
        self._lib.df_free_log_msg.argtypes = [ctypes.c_char_p]
        self._lib.df_free_log_msg.restype = None
        
        # df_free
        self._lib.df_free.argtypes = [ctypes.c_void_p]
        self._lib.df_free.restype = None
    
    def _initialize_model(self):
        model_path_bytes = self.model_path.encode('utf-8')
        log_level_bytes = self.log_level.encode('utf-8') if self.log_level else None
        
        # Use 100.0 dB (effectively no limit) if atten_lim is None
        effective_atten_lim = 100.0 if self.atten_lim is None else self.atten_lim
        
        self._state_ptr = self._lib.df_create(
            model_path_bytes,
            ctypes.c_float(effective_atten_lim),
            log_level_bytes
        )
        
        if not self._state_ptr:
            raise RuntimeError("Failed to initialize DeepFilterNet model")
        
        # Get frame length
        self._frame_length = self._lib.df_get_frame_length(self._state_ptr)
        
        # Calculate delay compensation
        # DeepFilterNet3 typically has: STFT delay (frame_length/2) + model lookahead
        # This matches the delay compensation in enhance_wav.rs
        if self.compensate_delay:
            # Conservative estimate: STFT window delay + some model lookahead
            # This may need adjustment based on specific model architecture
            self._delay_samples = self._frame_length // 2 + self._frame_length
        
        # Set post-filter if enabled
        if self.post_filter_beta > 0.0:
            self._lib.df_set_post_filter_beta(self._state_ptr, ctypes.c_float(self.post_filter_beta))
        
        if self._should_log():
            print(f"DeepFilterNet initialized with frame length: {self._frame_length}")
            if self.compensate_delay:
                print(f"Delay compensation: {self._delay_samples} samples ({self._delay_samples/self.sample_rate*1000:.1f}ms)")
            if self.atten_lim is not None:
                print(f"Attenuation limit: {self.atten_lim:.1f} dB")
            else:
                print("Attenuation limit: None (full noise reduction)")
            if self.post_filter_beta > 0.0:
                print(f"Post-filter beta: {self.post_filter_beta:.3f}")
    
    @property
    def frame_length(self) -> int:
        """Get the required frame length for processing."""
        return self._frame_length
    
    @property
    def sample_rate(self) -> int:
        """Get the expected sample rate."""
        return 48000
    
    def process_frame(self, audio_frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Process a single frame of audio with proper delay compensation.
        
        Args:
            audio_frame: Input audio frame as numpy array of shape (frame_length,)
                        with dtype float32, normalized to [-1, 1]
        
        Returns:
            Tuple of (denoised_audio, local_snr) where:
            - denoised_audio: Processed audio frame as numpy array (delay compensated)
            - local_snr: Local signal-to-noise ratio estimate
        
        Raises:
            ValueError: If input frame has incorrect shape or dtype
            RuntimeError: If processing fails
        """
        if self._state_ptr is None:
            raise RuntimeError("DeepFilterNet not initialized")
        
        if audio_frame.dtype != np.float32:
            audio_frame = audio_frame.astype(np.float32)
        
        if audio_frame.shape != (self._frame_length,):
            raise ValueError(
                f"Input frame must have shape ({self._frame_length},), "
                f"got {audio_frame.shape}"
            )
        
        # Prepare output buffer
        output_frame = np.zeros(self._frame_length, dtype=np.float32)
        
        input_ptr = audio_frame.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        output_ptr = output_frame.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        
        # Process the frame
        try:
            local_snr = self._lib.df_process_frame(self._state_ptr, input_ptr, output_ptr)
            
            # Apply delay compensation if enabled
            if self.compensate_delay:
                # Add output frame to delay buffer
                self._output_delay_buffer = np.concatenate([self._output_delay_buffer, output_frame])
                
                # Extract delayed output if enough samples available
                if len(self._output_delay_buffer) >= self._delay_samples + self._frame_length:
                    # Extract frame from delay position
                    delayed_output = self._output_delay_buffer[
                        self._delay_samples:self._delay_samples + self._frame_length
                    ]
                    # Remove consumed samples from buffer
                    self._output_delay_buffer = self._output_delay_buffer[self._frame_length:]
                    return delayed_output, float(local_snr)
                else:
                    # Not enough delayed samples yet, return zeros during initial delay period
                    return np.zeros(self._frame_length, dtype=np.float32), float(local_snr)
            else:
                return output_frame, float(local_snr)
                
        except Exception as e:
            raise RuntimeError(f"Failed to process audio frame: {e}")
    
    def process_chunk(self, audio_chunk: np.ndarray) -> np.ndarray:
        """
        Process a chunk of audio that may contain multiple frames, with proper
        boundary handling to avoid zero-padding artifacts.
        
        Args:
            audio_chunk: Input audio chunk as numpy array with dtype float32
        
        Returns:
            Processed audio chunk as numpy array
        """
        if len(audio_chunk) == 0:
            return np.array([], dtype=np.float32)
        
        # Ensure float32
        if audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)
        
        # Add new chunk to partial frame buffer
        combined_audio = np.concatenate([self._partial_frame_buffer, audio_chunk])
        num_complete_frames = len(combined_audio) // self._frame_length
        
        if num_complete_frames == 0:
            # Not enough data for a complete frame, store in buffer
            self._partial_frame_buffer = combined_audio
            return np.array([], dtype=np.float32)
        
        # Process complete frames
        output_frames = []
        for i in range(num_complete_frames):
            start_idx = i * self._frame_length
            end_idx = start_idx + self._frame_length
            frame = combined_audio[start_idx:end_idx]
            
            processed_frame, _ = self.process_frame(frame)
            output_frames.append(processed_frame)
        
        # Store remainder for next chunk
        remainder_start = num_complete_frames * self._frame_length
        self._partial_frame_buffer = combined_audio[remainder_start:]
        
        if output_frames:
            return np.concatenate(output_frames)
        else:
            return np.array([], dtype=np.float32)
    
    def flush(self) -> np.ndarray:
        """
        Flush any remaining buffered audio by processing the final partial frame.
        Call this at the end of a stream to ensure all audio is processed.
        
        Returns:
            Final processed audio samples
        """
        if len(self._partial_frame_buffer) == 0:
            return np.array([], dtype=np.float32)
        
        # Zero-pad only the final partial frame to complete processing
        padded_frame = np.zeros(self._frame_length, dtype=np.float32)
        original_length = len(self._partial_frame_buffer)
        padded_frame[:original_length] = self._partial_frame_buffer
        
        processed_frame, _ = self.process_frame(padded_frame)
        
        # Clear the buffer
        self._partial_frame_buffer = np.array([], dtype=np.float32)
        
        # Return only the portion corresponding to the original audio
        return processed_frame[:original_length]
    
    def set_attenuation_limit(self, lim_db: float):
        """
        Set the attenuation limit in dB.
        
        Args:
            lim_db: New attenuation limit in dB
        """
        if self._state_ptr is None:
            raise RuntimeError("DeepFilterNet not initialized")
        
        self._lib.df_set_atten_lim(self._state_ptr, ctypes.c_float(lim_db))
        self.atten_lim = lim_db
    
    def set_post_filter_beta(self, beta: float):
        """
        Set the post filter beta parameter.
        
        Args:
            beta: Post filter attenuation (0.0 disables post filter)
        """
        if self._state_ptr is None:
            raise RuntimeError("DeepFilterNet not initialized")
        
        self._lib.df_set_post_filter_beta(self._state_ptr, ctypes.c_float(beta))
    
    def get_log_messages(self) -> list:
        """
        Retrieve any pending log messages from the DeepFilterNet library.
        
        Returns:
            List of log message strings
        """
        if self._state_ptr is None:
            return []
        
        messages = []
        while True:
            msg_ptr = self._lib.df_next_log_msg(self._state_ptr)
            if not msg_ptr:
                break
            
            try:
                message = msg_ptr.decode('utf-8')
                messages.append(message)
            except UnicodeDecodeError:
                messages.append("<Invalid UTF-8 log message>")
            
            # Free the message
            self._lib.df_free_log_msg(msg_ptr)
        
        return messages
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        self.close()
        return False
    
    def close(self):
        """Clean up resources."""
        if self._state_ptr is not None:
            self._lib.df_free(self._state_ptr)
            self._state_ptr = None
        
        # Clear buffers
        self._partial_frame_buffer = np.array([], dtype=np.float32)
        self._output_delay_buffer = np.array([], dtype=np.float32)
        self._is_warmed_up = False
    
    def finalize(self) -> np.ndarray:
        """
        Finalize processing and return any remaining delayed samples.
        Call this at the end of a stream when using delay compensation.
        
        Returns:
            Final delayed samples from the delay buffer
        """
        result = np.array([], dtype=np.float32)
        
        # Flush partial frame buffer first
        if len(self._partial_frame_buffer) > 0:
            final_chunk = self.flush()
            result = np.concatenate([result, final_chunk]) if len(result) > 0 else final_chunk
        
        # Return remaining delay buffer samples if delay compensation is enabled
        if self.compensate_delay and len(self._output_delay_buffer) > 0:
            # Extract remaining samples from delay buffer
            remaining_samples = self._output_delay_buffer[self._delay_samples:]
            self._output_delay_buffer = np.array([], dtype=np.float32)
            result = np.concatenate([result, remaining_samples]) if len(result) > 0 else remaining_samples
        
        return result
    
    def __del__(self):
        """Destructor - ensure cleanup."""
        self.close()