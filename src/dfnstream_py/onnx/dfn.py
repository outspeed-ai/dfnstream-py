import numpy as np
import onnxruntime as ort
from typing import Optional

class ONNXDeepFilterNetStreaming:
    def __init__(self, onnx_model_path: str, atten_lim_db: Optional[float] = None):
        self.onnx_model_path = onnx_model_path

        if atten_lim_db is None:
            self.atten_lim = 0.0
        else:
            self.atten_lim = atten_lim_db
        
        self.ort_session = ort.InferenceSession(onnx_model_path)
        
        self.input_names = [input.name for input in self.ort_session.get_inputs()]
        self.output_names = [output.name for output in self.ort_session.get_outputs()]
        
        # Audio processing parameters
        self._frame_length = 512
        self._sample_rate = 48000
        
        # Initialize flattened states
        self.states = np.zeros(46136, dtype=np.float32)
        
        self._partial_frame_buffer = np.array([], dtype=np.float32)
    
    
    @property
    def frame_length(self) -> int:
        return self._frame_length
    
    @property
    def sample_rate(self) -> int:
        return self._sample_rate
    
    def set_attenuation_limit(self, lim_db: float):
        """Set attenuation limit in dB."""
        self.atten_lim = lim_db
    
    def process_frame(self, audio_frame: np.ndarray) -> np.ndarray:
        if audio_frame.dtype != np.float32:
            audio_frame = audio_frame.astype(np.float32)
        
        if audio_frame.shape != (self._frame_length,):
            raise ValueError(
                f"Input frame must have shape ({self._frame_length},), "
                f"got {audio_frame.shape}"
            )
        
        try:
            # Prepare input features
            input_features = {
                self.input_names[0]: audio_frame,  # input_frame
                self.input_names[1]: self.states,  # states (flattened)
                self.input_names[2]: np.array(self.atten_lim, dtype=np.float32)  # atten_lim_db
            }
            
            # Run inference
            outputs = self.ort_session.run(self.output_names, input_features)
            
            # Extract enhanced audio, updated states, and lsnr
            enhanced_chunk = outputs[0]  # enhanced_audio_frame
            self.states = outputs[1]     # new_states (flattened)
            # lsnr = outputs[2]         # lsnr (optional, can be used for monitoring)
            
            return enhanced_chunk
                
        except Exception as e:
            raise RuntimeError(f"Failed to process audio frame: {e}")
    
    def process_chunk(self, audio_chunk: np.ndarray) -> np.ndarray:
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
            
            processed_frame = self.process_frame(frame)
            output_frames.append(processed_frame)
        
        # Store remainder for next chunk
        remainder_start = num_complete_frames * self._frame_length
        self._partial_frame_buffer = combined_audio[remainder_start:]
        
        if output_frames:
            return np.concatenate(output_frames)
        else:
            return np.array([], dtype=np.float32)
    
    def flush(self) -> np.ndarray:
        if len(self._partial_frame_buffer) == 0:
            return np.array([], dtype=np.float32)
        
        # Zero-pad only the final partial frame to complete processing
        padded_frame = np.zeros(self._frame_length, dtype=np.float32)
        original_length = len(self._partial_frame_buffer)
        padded_frame[:original_length] = self._partial_frame_buffer
        
        processed_frame = self.process_frame(padded_frame)
        
        # Clear the buffer
        self._partial_frame_buffer = np.array([], dtype=np.float32)
        
        return processed_frame[:original_length]
    
    def finalize(self) -> np.ndarray:
        return self.flush()
    
    def close(self):
        """Clean up resources."""
        # Clear buffers
        self._partial_frame_buffer = np.array([], dtype=np.float32)
        self.states = np.zeros(46136, dtype=np.float32)
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, _exc_type, _exc_val, _exc_tb):
        """Context manager exit - cleanup resources."""
        self.close()
        return False
    
    def __del__(self):
        """Destructor - ensure cleanup."""
        self.close()