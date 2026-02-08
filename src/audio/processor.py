import os
import requests
import numpy as np
import librosa
import torch
import tempfile

class AudioProcessor:
    def __init__(self, sample_rate=22050):
        self.sr = sample_rate
        
    def download_preview(self, url):
        """Downloads a 30s preview MP3 to a temp buffer."""
        try:
            r = requests.get(url)
            if r.status_code == 200:
                # Create a temp file
                fd, path = tempfile.mkstemp(suffix='.mp3')
                with os.fdopen(fd, 'wb') as f:
                    f.write(r.content)
                return path
            return None
        except Exception as e:
            print(f"Error downloading preview: {e}")
            return None

    def process_track(self, file_path):
        """
        Computes audio features for a track.
        Returns a dict of features:
        - 'mfcc': texture/timbre (mean)
        - 'chroma': harmony/key (mean)
        - 'contrast': spectral contrast (mean)
        - 'tempo': BPM
        """
        try:
            # Load audio
            y, sr = librosa.load(file_path, sr=self.sr)
            
            # 1. MFCC (Timbre)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfcc, axis=1)
            
            # 2. Chroma (Harmony)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            chroma_mean = np.mean(chroma, axis=1)
            
            # 3. Spectral Contrast (Bright vs Dark)
            contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
            contrast_mean = np.mean(contrast, axis=1)
            
            # 4. Tempo (Rhythm)
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            tempo = librosa.feature.tempo(onset_envelope=onset_env, sr=sr)
            
            # Concatenate into a single Audio Vector
            # 13 MFCC + 12 Chroma + 7 Contrast + 1 Tempo = 33 dimensions
            # (In Prod we use a CNN embedding, but this is a solid handcrafted baseline)
            vector = np.concatenate([
                mfcc_mean, 
                chroma_mean, 
                contrast_mean, 
                [tempo[0]] 
            ])
            
            return {
                'vector': vector,
                'details': {
                    'tempo': tempo[0],
                    'duration': librosa.get_duration(y=y, sr=sr)
                }
            }
            
        except Exception as e:
            print(f"Error processing audio: {e}")
            return None
        finally:
            # Cleanup temp file if it exists and we created it
            # (Caller should handle cleanup if they own the file, but here we assume we do)
            pass 

    def analyze_url(self, preview_url):
        """Pipeline: URL -> Temp File -> Vector -> Cleanup"""
        if not preview_url:
            return None
            
        path = self.download_preview(preview_url)
        if not path:
            return None
            
        result = self.process_track(path)
        
        # Cleanup
        try:
            os.remove(path)
        except:
            pass
            
        return result
