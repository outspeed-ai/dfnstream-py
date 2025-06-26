import numpy as np
import onnxruntime as ort
from typing import List


class ONNXDeepFilterNetStreaming:
    def __init__(self, onnx_model_path: str):
        self.onnx_model_path = onnx_model_path
        
        self.ort_session = ort.InferenceSession(onnx_model_path)
        
        self.input_names = [input.name for input in self.ort_session.get_inputs()]
        self.output_names = [output.name for output in self.ort_session.get_outputs()]
        
        # Audio processing parameters
        self._frame_length = 512
        self._sample_rate = 48000
        
        self.states = self._initialize_states()
        
        self._partial_frame_buffer = np.array([], dtype=np.float32)
    
    def _initialize_states(self) -> List[np.ndarray]:
        """Initialize all model states with zeros using exact ONNX shapes"""
        states = []
        
        for i, input_info in enumerate(self.ort_session.get_inputs()):
            if i == 0:
                continue
            
            shape = input_info.shape
            
            if "erb_norm_state" in input_info.name:
                # Shape: [32]
                state = np.linspace(-60.0, -90.0, 32, dtype=np.float32)
            elif "band_unit_norm_state" in input_info.name:
                # Shape: [1, 96, 1]
                state = np.linspace(0.001, 0.0001, 96, dtype=np.float32).reshape(1, 96, 1)
            elif "analysis_mem" in input_info.name:
                # Shape: [512]
                state = np.zeros(512, dtype=np.float32)
            elif "synthesis_mem" in input_info.name:
                # Shape: [512]
                state = np.zeros(512, dtype=np.float32)
            elif "rolling_erb_buf" in input_info.name:
                # Shape: [1, 1, 3, 32]
                state = np.zeros((1, 1, 3, 32), dtype=np.float32)
            elif "rolling_feat_spec_buf" in input_info.name:
                # Shape: [1, 2, 3, 96]
                state = np.zeros((1, 2, 3, 96), dtype=np.float32)
            elif "rolling_c0_buf" in input_info.name:
                # Shape: [1, 64, 5, 96]
                state = np.zeros((1, 64, 5, 96), dtype=np.float32)
            elif "rolling_spec_buf_x" in input_info.name:
                # Shape: [5, 513, 2]
                state = np.zeros((5, 513, 2), dtype=np.float32)
            elif "rolling_spec_buf_y" in input_info.name:
                # Shape: [7, 513, 2]
                state = np.zeros((7, 513, 2), dtype=np.float32)
            elif "enc_hidden" in input_info.name:
                # Shape: [1, 1, 256]
                state = np.zeros((1, 1, 256), dtype=np.float32)
            elif "erb_dec_hidden" in input_info.name:
                # Shape: [2, 1, 256]
                state = np.zeros((2, 1, 256), dtype=np.float32)
            elif "df_dec_hidden" in input_info.name:
                # Shape: [2, 1, 256]
                state = np.zeros((2, 1, 256), dtype=np.float32)
            else:
                # Fallback to zeros
                state = np.zeros(shape, dtype=np.float32)
            
            states.append(state)
            
        return states
    
    @property
    def frame_length(self) -> int:
        return self._frame_length
    
    @property
    def sample_rate(self) -> int:
        return self._sample_rate
    
    def process_frame(self, audio_frame: np.ndarray) -> np.ndarray:
        if audio_frame.dtype != np.float32:
            audio_frame = audio_frame.astype(np.float32)
        
        if audio_frame.shape != (self._frame_length,):
            raise ValueError(
                f"Input frame must have shape ({self._frame_length},), "
                f"got {audio_frame.shape}"
            )
        
        try:
            input_features = {
                self.input_names[0]: audio_frame
            }
            
            for i, state in enumerate(self.states):
                input_features[self.input_names[i + 1]] = state
            
            # Run inference
            outputs = self.ort_session.run(self.output_names, input_features)
            
            # Extract enhanced audio and updated states
            enhanced_chunk = outputs[0]  # First output is enhanced audio
            self.states = outputs[1:]    # Remaining outputs are updated states
            
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
        self.states = self._initialize_states()
    
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