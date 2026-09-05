"""
audio_capture.py — System audio loopback capture with Voice Activity Detection (VAD)

Captures system audio output (what you hear through speakers/headphones),
detects speech segments using RMS energy + silence threshold,
and yields audio chunks when someone stops speaking.

Backends:
1. soundcard (WASAPI Loopback) — Primary on Windows
2. sounddevice (PortAudio) — Primary on Linux/macOS or fallback for microphone/stereo mix
3. arecord subprocess (ALSA/PulseAudio) — Fallback on Linux only
"""

import numpy as np
import queue
import threading
from collections import deque
import time
import logging
import subprocess
import sys
import os
import warnings

logger = logging.getLogger(__name__)

# Try to import soundcard (WASAPI loopback support on Windows)
try:
    import soundcard as sc
    SOUNDCARD_AVAILABLE = True
    # Filter by category, not module path: `module=` in filterwarnings matches
    # against the warning's source *file path* (see warnings.warn_explicit),
    # not soundcard's dotted module name — so `module='soundcard'` never
    # actually matched a file under .../site-packages/soundcard/... and this
    # warning was never suppressed despite the previous attempt below.
    warnings.filterwarnings('ignore', category=sc.SoundcardRuntimeWarning)
except Exception:
    SOUNDCARD_AVAILABLE = False

# Try to import sounddevice
try:
    import sounddevice as sd
    SD_AVAILABLE = True
except (ImportError, OSError):
    SD_AVAILABLE = False
    logger.info("sounddevice not available")


class AudioCapture:
    def __init__(self, sample_rate=16000, chunk_duration=3,
                 silence_threshold=0.01, silence_duration=1.5,
                 min_utterance_sec=1.2, ring_seconds=30,
                 max_utterance_sec=30):
        self.sample_rate = sample_rate
        # Utterances shorter than this never reach the transcription API:
        # "mm-hm" and "right" are not questions, and each one would cost a
        # request out of the daily quota.
        self.min_utterance_sec = min_utterance_sec
        # And a ceiling, because nothing else ends an utterance that never
        # goes quiet.
        self.max_utterance_sec = max_utterance_sec
        self.chunk_duration = chunk_duration
        self.silence_threshold = silence_threshold
        self.silence_duration = silence_duration
        
        # "Auto" lets start() pick; otherwise "sc:<id>" (WASAPI loopback via
        # soundcard) or "sd:<index>" (PortAudio input via sounddevice).
        self.device_id = "Auto"

        # Watchdog bookkeeping: when the last frame arrived, and which
        # speaker the WASAPI loopback was opened against.
        self.last_frame_time = None
        self._opened_speaker_id = None

        # Everything heard in the last `ring_seconds`, whether or not VAD
        # thought it was speech. The grab hotkey slices its window out of
        # here, which is why it can answer a question that was already over
        # by the time you decided you wanted help with it.
        self._ring = deque()
        self._ring_samples = 0
        self._ring_capacity = int(sample_rate * ring_seconds)
        self._ring_lock = threading.Lock()

        self.audio_queue = queue.Queue()
        self.is_running = False
        self._thread = None
        self._device_index = None
        self._arecord_proc = None
        
        # Buffer for accumulating audio
        self._buffer = []
        self._buffer_samples = 0
        self._silence_frames = 0
        self._is_speaking = False
        self._frames_per_chunk = int(sample_rate * 0.1)  # 100ms frames

        # Why the capture thread stopped, if it did. See capture_alive().
        self._capture_error = None

        # The few frames before the gate opened. Speech starts quieter than it
        # continues, so the frame that finally crosses the threshold is never
        # the first frame of the word — buffering only from there clips the
        # attack and Whisper loses or mangles the opening word. "How would you
        # design..." arriving as "would you design..." changes the answer, not
        # just the transcript.
        self._preroll = deque(maxlen=3)          # ~300ms

        # Set by the grab hotkey. The speech it just transcribed is still
        # sitting in _buffer, so without this the VAD emits it a moment later,
        # it gets transcribed a second time, and the duplicate answer
        # supersedes the grabbed one — the user watches the answer they asked
        # for get wiped. A float write from another thread is atomic enough
        # for a deadline nobody reads twice.
        self._drop_until = 0.0
        
    def set_device(self, device_id):
        """Point capture at a device id from list_input_devices(). Takes
        effect on the next start()."""
        self.device_id = device_id or "Auto"
        logger.info(f"Audio device preference: {self.device_id}")

    def list_input_devices(self):
        """
        [(device_id, label)] for the setup panel — "Auto" first, then WASAPI
        loopback devices (what you want: the interviewer's voice out of the
        speakers), then ordinary inputs.
        """
        devices = [("Auto", "Auto — detect")]

        if sys.platform == 'win32' and SOUNDCARD_AVAILABLE:
            try:
                for mic in sc.all_microphones(include_loopback=True):
                    tag = "loopback" if getattr(mic, "isloopback", False) else "input"
                    devices.append((f"sc:{mic.id}", f"{mic.name}  ({tag})"))
            except Exception as e:
                logger.warning(f"Could not list soundcard devices: {e}")

        if SD_AVAILABLE:
            try:
                for i, dev in enumerate(sd.query_devices()):
                    if dev.get('max_input_channels', 0) > 0:
                        devices.append((f"sd:{i}", f"{dev['name']}  (input)"))
            except Exception as e:
                logger.warning(f"Could not list sounddevice devices: {e}")

        return devices

    def find_loopback_device(self):
        """Find an input device that supports audio capture (sounddevice)."""
        if not SD_AVAILABLE:
            return None
            
        devices = sd.query_devices()
        candidates = []
        
        for i, dev in enumerate(devices):
            name = dev['name'].lower()
            max_input = dev.get('max_input_channels', 0)
            
            if max_input > 0:
                if 'monitor' in name:
                    candidates.append((i, dev['name'], 'linux-monitor'))
                elif 'stereo mix' in name:
                    candidates.append((i, dev['name'], 'windows-stereo-mix'))
                else:
                    candidates.append((i, dev['name'], 'input-device'))
        
        if not candidates:
            try:
                default = sd.default.device[0]
                if default is not None and default >= 0:
                    return default
            except:
                pass
            raise RuntimeError("No valid audio input device found")
        
        # Prioritize monitor / stereo mix if available
        for idx, name, dtype in candidates:
            if 'monitor' in name.lower() or 'stereo mix' in name.lower():
                logger.info(f"Found loopback input device: [{idx}] {name} ({dtype})")
                return idx
        
        idx, name, dtype = candidates[0]
        logger.info(f"Using audio input device: [{idx}] {name} ({dtype})")
        return idx
    
    def _find_arecord_monitor(self):
        """Find PulseAudio monitor source for arecord (Linux only)."""
        if not sys.platform.startswith('linux'):
            return 'default'
        try:
            result = subprocess.run(
                ['pactl', 'list', 'short', 'sources'],
                capture_output=True, text=True, timeout=5
            )
            for line in result.stdout.strip().split('\n'):
                if '.monitor' in line:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        source_name = parts[1]
                        logger.info(f"Found monitor source: {source_name}")
                        return source_name
            logger.info("No monitor source found, using default")
            return 'default'
        except Exception as e:
            logger.warning(f"pactl failed: {e}, using default")
            return 'default'
    
    def _rms(self, audio_chunk):
        """Calculate RMS energy of audio chunk."""
        return np.sqrt(np.mean(np.square(audio_chunk)))

    def _ring_push(self, audio):
        """Add a frame to the rolling window, dropping the oldest to fit."""
        with self._ring_lock:
            self._ring.append(audio.copy())
            self._ring_samples += len(audio)
            while self._ring_samples > self._ring_capacity and self._ring:
                self._ring_samples -= len(self._ring.popleft())

    def _ring_clear(self):
        """Drop the rolling window — called when capture moves device."""
        with self._ring_lock:
            self._ring.clear()
            self._ring_samples = 0

    def grab_recent(self, seconds: float):
        """
        The last `seconds` of audio, or None if barely anything is buffered.

        Unlike the VAD path this ignores speech boundaries entirely: it
        returns the window as heard, silence included, because the press
        that asked for it is the only intent signal needed.
        """
        wanted = int(self.sample_rate * seconds)
        with self._ring_lock:
            if self._ring_samples < self.sample_rate * self.min_utterance_sec:
                logger.info(
                    f"Grab found only {self._ring_samples / self.sample_rate:.1f}s "
                    f"buffered (floor {self.min_utterance_sec}s)")
                return None
            frames = list(self._ring)
        combined = np.concatenate(frames)
        return combined[-wanted:] if len(combined) > wanted else combined
    
    def _audio_callback(self, indata, frames, time_info, status):
        """PortAudio callback — called for each audio frame."""
        if status:
            logger.debug(f"Audio status: {status}")
        
        if indata.shape[1] > 1:
            audio = np.mean(indata, axis=1)
        else:
            audio = indata.flatten()
        
        self._process_audio(audio)
    
    def _process_audio(self, audio):
        """Process audio buffer with VAD — shared between backends."""
        self.last_frame_time = time.time()
        self._ring_push(audio)
        rms = self._rms(audio)
        
        # Debug: log every 10th frame (every 1s)
        if not hasattr(self, '_frame_count'):
            self._frame_count = 0
        self._frame_count += 1
        if self._frame_count % 10 == 0:
            logger.debug(f"VAD frame {self._frame_count}: RMS={rms:.4f} threshold={self.silence_threshold} speaking={self._is_speaking}")
        
        if rms > self.silence_threshold:
            if not self._is_speaking:
                # Opening the gate: take the sub-threshold frames with it, so
                # the utterance starts before the first loud syllable.
                for frame in self._preroll:
                    self._buffer.append(frame)
                    self._buffer_samples += len(frame)
                self._preroll.clear()
            self._is_speaking = True
            self._silence_frames = 0
            self._buffer.append(audio.copy())
            self._buffer_samples += len(audio)

            # Nothing else stops an utterance growing. Room tone, music or a
            # fan holds the gate open indefinitely, so three questions arrive
            # as one chunk — or nothing is emitted until the room falls quiet
            # and a multi-minute WAV goes to the STT API in one request.
            if self._buffer_samples >= self.sample_rate * self.max_utterance_sec:
                logger.info(f"Utterance hit the {self.max_utterance_sec:.0f}s "
                            f"ceiling — emitting early")
                self._emit_buffer()
        else:
            # Silence
            if self._is_speaking:
                self._silence_frames += 1
                self._buffer.append(audio.copy())  # keep trailing silence
                self._buffer_samples += len(audio)

                silence_threshold_frames = int(
                    (self.silence_duration * self.sample_rate) / self._frames_per_chunk
                )
                if self._silence_frames >= silence_threshold_frames:
                    self._emit_buffer()
            else:
                # Silence before any speech. Kept only as pre-roll for the
                # utterance that may be about to start.
                self._preroll.append(audio.copy())

    def _emit_buffer(self):
        """Hand the buffered utterance to the queue and reset the gate."""
        if self._buffer:
            combined = np.concatenate(self._buffer)
            secs = len(combined) / self.sample_rate
            if time.time() < self._drop_until:
                # The grab hotkey already transcribed this window.
                logger.info(f"Dropped {secs:.1f}s the grab hotkey just answered")
            elif len(combined) >= self.sample_rate * self.min_utterance_sec:
                self.audio_queue.put(combined)
                logger.info(f"Audio chunk emitted: {secs:.1f}s")
            else:
                logger.debug(f"Skipped {secs:.2f}s utterance "
                             f"(floor {self.min_utterance_sec}s)")
        self._buffer = []
        self._buffer_samples = 0
        self._is_speaking = False
        self._silence_frames = 0

    def suppress_pending(self, seconds: float):
        """
        Ignore whatever the VAD is holding, and anything it emits for the next
        `seconds`. Called by the grab hotkey, which has just transcribed that
        same audio out of the ring buffer.
        """
        self._drop_until = time.time() + seconds
        drained = 0
        try:
            while True:
                self.audio_queue.get_nowait()
                drained += 1
        except queue.Empty:
            pass
        if drained:
            logger.info(f"Discarded {drained} queued chunk(s) after a grab")
    
    def capture_alive(self):
        """
        (alive, reason) for the capture thread.

        The stall check next to this cannot answer the question: WASAPI
        loopback delivers frames of zeros when nothing is playing, so "silent"
        and "dead" look identical from the frame timestamps. Whether the
        thread is still running is a fact, not a heuristic.
        """
        if not self.is_running or self._thread is None:
            return True, None                 # not started, or not thread-based
        if self._thread.is_alive():
            return True, None
        return False, self._capture_error or "the capture thread stopped"

    def _soundcard_thread(self):
        """Thread that captures system audio loopback using soundcard (Windows WASAPI)."""
        logger.info("Starting soundcard WASAPI loopback capture...")
        try:
            if str(self.device_id).startswith("sc:"):
                mic = sc.get_microphone(id=self.device_id[3:], include_loopback=True)
                self._opened_speaker_id = None
            else:
                spk = sc.default_speaker()
                # Remembered so the watchdog can tell when Windows switches
                # the default out from under us.
                self._opened_speaker_id = str(spk.id)
                mic = sc.get_microphone(id=self._opened_speaker_id,
                                        include_loopback=True)
            logger.info(f"Using soundcard loopback device: {mic.name}")
            
            with mic.recorder(samplerate=self.sample_rate, channels=1) as recorder:
                while self.is_running:
                    data = recorder.record(numframes=self._frames_per_chunk)
                    if data.size == 0:
                        time.sleep(0.01)
                        continue
                    audio = data[:, 0].astype(np.float32)
                    self._process_audio(audio)
        except Exception as e:
            # Recorded, not just logged. This thread dying is the app's real
            # silent death: is_running stays True, the queue simply never
            # fills again, and every status the user can see keeps saying
            # "listening". The watchdog reads this back to say what happened.
            self._capture_error = str(e)
            logger.error(f"soundcard thread error: {e}")

    def _arecord_thread(self):
        """Thread that reads audio from arecord subprocess (Linux only)."""
        if not sys.platform.startswith('linux'):
            logger.error("arecord is only supported on Linux.")
            return

        import select
        
        monitor = self._find_arecord_monitor()
        
        cmd = [
            'arecord',
            '-D', f'pulse:{monitor}',
            '-f', 'S16_LE',
            '-r', str(self.sample_rate),
            '-c', '1',
            '-t', 'raw',
            '--buffer-size=1024',
        ]
        
        logger.info(f"Starting arecord: {' '.join(cmd)}")
        
        try:
            self._arecord_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            
            fd = self._arecord_proc.stdout.fileno()
            bytes_per_frame = 2  # S16_LE = 2 bytes
            bytes_per_chunk = self._frames_per_chunk * bytes_per_frame
            
            byte_buffer = bytearray()
            
            while self.is_running and self._arecord_proc.poll() is None:
                ready, _, _ = select.select([fd], [], [], 0.1)
                if not ready:
                    continue
                
                chunk_data = os.read(fd, bytes_per_chunk * 4)
                if not chunk_data:
                    continue
                
                byte_buffer.extend(chunk_data)
                
                while len(byte_buffer) >= bytes_per_chunk:
                    raw = bytes(byte_buffer[:bytes_per_chunk])
                    del byte_buffer[:bytes_per_chunk]
                    
                    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                    self._process_audio(audio)
                
        except Exception as e:
            logger.error(f"arecord thread error: {e}")
        finally:
            if self._arecord_proc:
                self._arecord_proc.terminate()
    
    def start(self):
        """Start capturing audio."""
        self.is_running = True
        # Whatever the previous device left behind is not what the grab
        # hotkey means by "the last twenty seconds" — and a half-captured
        # utterance from the old device must not get concatenated onto the
        # first frames of the new one.
        self._ring_clear()
        self._buffer = []
        self._buffer_samples = 0
        self._silence_frames = 0
        self._is_speaking = False
        self._preroll.clear()
        self._drop_until = 0.0
        self.last_frame_time = None
        self._capture_error = None

        # A device picked explicitly decides the backend; "sd:" means the
        # user chose a PortAudio input, so skip the WASAPI path entirely.
        wants_sounddevice = str(self.device_id).startswith("sd:")

        # Primary backend on Windows: soundcard WASAPI Loopback
        if sys.platform == 'win32' and SOUNDCARD_AVAILABLE and not wants_sounddevice:
            try:
                self._thread = threading.Thread(target=self._soundcard_thread, daemon=True)
                self._thread.start()
                logger.info("Audio capture started (soundcard WASAPI loopback)")
                return
            except Exception as e:
                logger.warning(f"soundcard loopback initialization failed: {e}, falling back to sounddevice")

        # sounddevice backend (Linux monitor / Stereo Mix / default microphone)
        if SD_AVAILABLE:
            try:
                self._device_index = (int(self.device_id[3:]) if wants_sounddevice
                                      else self.find_loopback_device())
                self._stream = sd.InputStream(
                    device=self._device_index,
                    channels=1,
                    samplerate=self.sample_rate,
                    blocksize=self._frames_per_chunk,
                    dtype='float32',
                    callback=self._audio_callback
                )
                self._stream.start()
                logger.info(f"Audio capture started (sounddevice) on device {self._device_index}")
                return
            except Exception as e:
                logger.warning(f"sounddevice failed: {e}")

        # Linux fallback: arecord
        if sys.platform.startswith('linux'):
            self._thread = threading.Thread(target=self._arecord_thread, daemon=True)
            self._thread.start()
            logger.info("Audio capture started (arecord fallback)")
        else:
            logger.error("No valid audio capture backend available on this system.")
    
    def stop(self):
        """Stop capturing audio."""
        self.is_running = False
        if hasattr(self, '_stream'):
            try:
                self._stream.stop()
                self._stream.close()
            except:
                pass
        if self._arecord_proc:
            self._arecord_proc.terminate()
            self._arecord_proc = None
        if self._thread:
            self._thread.join(timeout=2)
        logger.info("Audio capture stopped")
    
    def default_device_changed(self) -> bool:
        """
        True when we are following the default speaker and Windows has since
        made a different one default — headphones plugged in, a headset
        connecting, a call app grabbing a device.
        """
        if self._opened_speaker_id is None or not SOUNDCARD_AVAILABLE:
            return False
        try:
            return str(sc.default_speaker().id) != self._opened_speaker_id
        except Exception as e:
            logger.debug(f"Could not read the default speaker: {e}")
            return False

    def get_audio_chunk(self, timeout=30):
        """Get the next audio chunk from the queue. Blocks until available."""
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def list_devices(self):
        """List all available audio devices for debugging."""
        if SOUNDCARD_AVAILABLE and sys.platform == 'win32':
            try:
                spk = sc.default_speaker()
                mics = sc.all_microphones(include_loopback=True)
                print("\n=== Available Audio Devices (soundcard) ===")
                print(f"Default Speaker: {spk.name}")
                for m in mics:
                    print(f"  Microphone/Loopback: {m.name}")
            except Exception as e:
                print(f"soundcard query error: {e}")

        if SD_AVAILABLE:
            devices = sd.query_devices()
            print("\n=== Available Audio Devices (sounddevice) ===")
            for i, dev in enumerate(devices):
                print(f"[{i}] {dev['name']} | in:{dev['max_input_channels']} out:{dev['max_output_channels']}")
            return devices
        elif sys.platform.startswith('linux'):
            print("\n=== Audio Devices (arecord/pulse) ===")
            try:
                result = subprocess.run(['pactl', 'list', 'short', 'sources'], 
                                      capture_output=True, text=True, timeout=5)
                print(result.stdout)
            except:
                print("Could not list devices")
            return []
        return []


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
    
    cap = AudioCapture()
    cap.list_devices()
    
    print("\n--- Starting 15-second capture test ---")
    print("Play some audio (YouTube, music, etc) to test...")
    cap.start()
    
    chunks = []
    start = time.time()
    while time.time() - start < 15:
        chunk = cap.get_audio_chunk(timeout=1)
        if chunk is not None:
            chunks.append(chunk)
            rms = np.sqrt(np.mean(np.square(chunk)))
            print(f"  Got chunk: {len(chunk)/cap.sample_rate:.1f}s, RMS: {rms:.4f}")
    
    cap.stop()
    print(f"\nCaptured {len(chunks)} chunks in 15 seconds")
    if chunks:
        total_audio = sum(len(c) for c in chunks) / cap.sample_rate
        print(f"Total audio: {total_audio:.1f}s")