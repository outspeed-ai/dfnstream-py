"""
DeepFilterNet Streaming implementation for Python (using Torch ONNX)
"""

import os
from typing import Optional, Tuple
import numpy as np

from .onnx.dfn import ONNXDeepFilterNetStreaming


class DeepFilterNetStreaming:
    def __init__(self, model_path: str | None = None, atten_lim: Optional[float] = None, 
                 log_level: Optional[str] = None, compensate_delay: bool = True,
                 post_filter_beta: float = 0.0):
        """
        Initialize DeepFilterNet streaming processor using ONNX backend.
        
        Args:
            model_path: Optional path to ONNX model file (.onnx file)
            atten_lim: Attenuation limit in dB (default: None = no limit, full noise reduction)
            log_level: Ignored in ONNX version
            compensate_delay: Ignored in ONNX version
            post_filter_beta: Ignored in ONNX version
        """
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "onnx", "denoiser_model.onnx")
        
        self._onnx_processor = ONNXDeepFilterNetStreaming(
            onnx_model_path=model_path,
            atten_lim_db=atten_lim,
        )
    
    @property
    def frame_length(self) -> int:
        """Get the required frame length for processing."""
        return self._onnx_processor.frame_length
    
    @property
    def sample_rate(self) -> int:
        """Get the expected sample rate."""
        return self._onnx_processor.sample_rate
    
    def process_frame(self, audio_frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Process a single frame of audio
        
        Args:
            audio_frame: Input audio frame as numpy array of shape (frame_length,)
                        with dtype float32, normalized to [-1, 1]
        
        Returns:
            Tuple of (denoised_audio, local_snr) where local_snr is always 0.0 for ONNX version
        
        Raises:
            ValueError: If input frame has incorrect shape or dtype
            RuntimeError: If processing fails
        """
        return self._onnx_processor.process_frame(audio_frame), 0.0
    
    def process_chunk(self, audio_chunk: np.ndarray) -> np.ndarray:
        """
        Process a chunk of audio that may contain multiple frames, with proper
        boundary handling to avoid zero-padding artifacts.
        
        Args:
            audio_chunk: Input audio chunk as numpy array with dtype float32
        
        Returns:
            Processed audio chunk as numpy array
        """
        return self._onnx_processor.process_chunk(audio_chunk)
    
    def flush(self) -> np.ndarray:
        """
        Flush any remaining buffered audio by processing the final partial frame.
        Call this at the end of a stream to ensure all audio is processed.
        
        Returns:
            Final processed audio samples
        """
        return self._onnx_processor.flush()
    
    def set_attenuation_limit(self, lim_db: float):
        """
        Set the attenuation limit in dB.
        
        Args:
            lim_db: New attenuation limit in dB
        """
        self._onnx_processor.set_attenuation_limit(lim_db)
    
    def set_post_filter_beta(self, beta: float):
        """
        Set the post filter beta parameter.
        Note: Not implemented in ONNX version.
        
        Args:
            beta: Post filter attenuation (0.0 disables post filter)
        """
        pass
    
    def get_log_messages(self) -> list:
        """
        Retrieve any pending log messages.
        Note: Not implemented in ONNX version.
        
        Returns:
            Empty list (no logging in ONNX version)
        """
        return []
    
    def finalize(self) -> np.ndarray:
        """
        Finalize processing and return any remaining delayed samples.
        Call this at the end of a stream when using delay compensation.
        
        Returns:
            Final delayed samples from the delay buffer
        """
        return self._onnx_processor.finalize()
    
    def close(self):
        """Clean up resources."""
        self._onnx_processor.close()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup resources."""
        self.close()
        return False
    
    def __del__(self):
        """Destructor - ensure cleanup."""
        self.close()