import threading
import signal
import os
import cv2
import sys
import concurrent.futures
import time
from typing import Optional, Callable, Any
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = 'xcb'

repo_root = None
for p in Path(__file__).resolve().parents:
    if (p / "hailo_apps" / "config" / "config_manager.py").exists():
        repo_root = p
        break
if repo_root is not None:
    sys.path.insert(0, str(repo_root))

from hailo_apps.python.gen_ai_apps.vlm_chat.backend import Backend
from hailo_apps.python.core.common.core import get_standalone_parser, get_logger, handle_list_models_flag, resolve_hef_path
from hailo_apps.python.core.common.defines import (
    VLM_CHAT_APP,
    HAILO10H_ARCH,
    SHARED_VDEVICE_GROUP_ID,
)

# Configuration Constants
MAX_TOKENS = 200
TEMPERATURE = 0.1
SEED = 42
SYSTEM_PROMPT = "You are a helpful assistant that analyzes images and answers questions about them."
INFERENCE_TIMEOUT = 60
SAVE_FRAMES = False

# App States
STATE_STREAMING = "STREAMING"
STATE_CAPTURED = "CAPTURED"
STATE_PROCESSING = "PROCESSING"
STATE_RESULT = "RESULT"
# Event-trigger / monitor mode states.
STATE_TRIGGER_SETUP = "TRIGGER_SETUP"
STATE_MONITORING = "MONITORING"
STATE_EVENT = "EVENT"

# Monitor mode defaults.
MONITOR_INTERVAL_DEFAULT = 1.0   # min seconds between successive captures
MONITOR_COOLDOWN_DEFAULT = 30.0  # min seconds between fired events

# Initialize logger
logger = get_logger(__name__)

class VLMChatApp:
    """
    Main application class for VLM Chat.
    Handles video display, user input, and interaction with the VLM backend.
    """
    def __init__(self,
                 speech_enabled: bool = True, tts_enabled: bool = True,
                 monitor_interval: float = MONITOR_INTERVAL_DEFAULT,
                 monitor_cooldown: float = MONITOR_COOLDOWN_DEFAULT):
        """
        Initialize the VLM Chat Application.

        Args:
            speech_enabled (bool): If True, enable Whisper speech-to-text input.
            tts_enabled (bool): If True, enable Piper text-to-speech output.
        """
        self.running = True
        self.executor = concurrent.futures.ThreadPoolExecutor()
        signal.signal(signal.SIGINT, self.signal_handler)
        self.frozen_frame = None
        self.frozen_inference_frame = None
        self.current_state = STATE_STREAMING
        self.user_question = ''
        self.streamed_response = ''
        self.pip_anim_start: Optional[float] = None  # set when PiP appears, drives slide-in
        self.pip_anim_out_start: Optional[float] = None  # set when dismissing, drives slide-out
        self.backend: Optional[Backend] = None
        self.video_thread: Optional[threading.Thread] = None

        # Speech / TTS feature state — voice components are initialised lazily in show_video.
        self.speech_enabled = speech_enabled
        self.tts_enabled = tts_enabled
        self.is_recording = False
        self.is_transcribing = False
        self.has_typed = False  # Once True, Space is a normal char, not a record toggle
        self.transcribe_future: Optional[concurrent.futures.Future] = None
        self.audio_recorder = None
        self.s2t = None
        self.tts = None
        self.stt_vdevice = None
        self.tts_buffer = ''
        self.tts_gen_id: Optional[int] = None
        self.tts_first_chunk = True
        # Recording helpers can fill either user_question (default) or trigger_text.
        self._active_text_target = 'user_question'

        # Event-trigger / monitor state.
        self.trigger_text = ''
        self.last_event_text = ''
        self.last_capture_time = 0.0
        self.last_event_time = 0.0
        self.monitor_future: Optional[concurrent.futures.Future] = None
        self._last_monitor_view = None
        self.monitor_interval = monitor_interval
        self.monitor_cooldown = monitor_cooldown

    def signal_handler(self, sig, frame):
        """Handle interrupt signals."""
        print('')
        logger.info("Signal received, shutting down...")
        self.stop()

    def stop(self):
        """Stop the application and clean up resources."""
        self.running = False
        # Tear down audio first so nothing is still streaming to the speakers / mic.
        if self.is_recording and self.audio_recorder is not None:
            try:
                self.audio_recorder.stop()
            except Exception as e:
                logger.debug(f"Error stopping recorder: {e}")
            self.is_recording = False
        # Cancel any in-flight monitor inference best-effort.
        if self.monitor_future is not None:
            self.monitor_future.cancel()
            self.monitor_future = None
        if self.tts is not None:
            try:
                self.tts.stop()
            except Exception as e:
                logger.debug(f"Error stopping TTS: {e}")
        if self.audio_recorder is not None:
            close_fn = getattr(self.audio_recorder, 'close', None)
            if callable(close_fn):
                try:
                    close_fn()
                except Exception as e:
                    logger.debug(f"Error closing recorder: {e}")
        if self.stt_vdevice is not None:
            try:
                self.stt_vdevice.release()
            except Exception as e:
                logger.debug(f"Error releasing STT VDevice: {e}")
        if self.backend:
            self.backend.close()
        self.executor.shutdown(wait=True)

    def _init_voice_stack(self) -> None:
        """Best-effort init of STT/TTS components. Failure disables the feature, not the app."""
        if self.speech_enabled:
            try:
                from hailo_platform import VDevice
                from hailo_apps.python.gen_ai_apps.gen_ai_utils.voice_processing.speech_to_text import SpeechToTextProcessor
                from hailo_apps.python.gen_ai_apps.gen_ai_utils.voice_processing.audio_recorder import AudioRecorder
                params = VDevice.create_params()
                params.group_id = SHARED_VDEVICE_GROUP_ID
                self.stt_vdevice = VDevice(params)
                self.s2t = SpeechToTextProcessor(self.stt_vdevice)
                self.audio_recorder = AudioRecorder(device_id=None, debug=False)
                logger.info("Speech-to-text initialised (Whisper).")
            except Exception as e:
                logger.warning(f"Speech input disabled: {e}")
                self.speech_enabled = False
                self.s2t = None
                self.audio_recorder = None
                if self.stt_vdevice is not None:
                    try:
                        self.stt_vdevice.release()
                    except Exception:
                        pass
                    self.stt_vdevice = None

        if self.tts_enabled:
            try:
                from hailo_apps.python.gen_ai_apps.gen_ai_utils.voice_processing.text_to_speech import (
                    TextToSpeechProcessor, check_piper_model_installed,
                )
                check_piper_model_installed()
                self.tts = TextToSpeechProcessor(device_id=None)
                logger.info("Text-to-speech initialised (Piper).")
            except Exception as e:
                logger.warning(f"TTS disabled: {e}")
                self.tts_enabled = False
                self.tts = None

    def _start_recording(self) -> None:
        if self.audio_recorder is None:
            return
        # Clear whichever text field is being filled (user_question or trigger_text).
        setattr(self, self._active_text_target, '')
        self.has_typed = False
        try:
            self.audio_recorder.start()
            self.is_recording = True
        except Exception as e:
            logger.error(f"Failed to start recording: {e}")
            self.is_recording = False

    def _stop_recording_and_transcribe(self) -> None:
        if self.audio_recorder is None or not self.is_recording:
            return
        try:
            audio = self.audio_recorder.stop()
        except Exception as e:
            logger.error(f"Failed to stop recording: {e}")
            self.is_recording = False
            return
        self.is_recording = False
        if audio is None or len(audio) == 0 or self.s2t is None:
            return
        self.is_transcribing = True
        self.transcribe_future = self.executor.submit(self.s2t.transcribe, audio, "en", 15000)

    def _poll_transcription(self) -> None:
        if self.transcribe_future is None or not self.transcribe_future.done():
            return
        try:
            text = self.transcribe_future.result()
            if text:
                setattr(self, self._active_text_target, text.strip())
                # Transcribed text shouldn't suppress Space-to-record — only manual typing does.
                self.has_typed = False
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
        finally:
            self.transcribe_future = None
            self.is_transcribing = False

    def _abort_recording(self) -> None:
        """Discard any in-progress recording (used when leaving CAPTURED state)."""
        if self.is_recording and self.audio_recorder is not None:
            try:
                self.audio_recorder.stop()
            except Exception as e:
                logger.debug(f"Error aborting recorder: {e}")
            self.is_recording = False

    def _tts_begin(self) -> None:
        if self.tts is None:
            return
        try:
            self.tts.interrupt()
            if hasattr(self.tts, 'clear_interruption'):
                self.tts.clear_interruption()
            self.tts_gen_id = self.tts.get_current_gen_id()
            self.tts_buffer = ''
            self.tts_first_chunk = True
        except Exception as e:
            logger.debug(f"Error starting TTS generation: {e}")

    def _tts_feed(self, new_text: str) -> None:
        if self.tts is None or self.tts_gen_id is None or not new_text:
            return
        self.tts_buffer += new_text
        try:
            self.tts_buffer = self.tts.chunk_and_queue(
                self.tts_buffer, self.tts_gen_id, self.tts_first_chunk
            )
            # Once anything has been queued, switch to sentence-only chunking.
            if self.tts_first_chunk and not self.tts.speech_queue.empty():
                self.tts_first_chunk = False
        except Exception as e:
            logger.debug(f"Error queuing TTS chunk: {e}")

    def _tts_flush(self) -> None:
        if self.tts is None or self.tts_gen_id is None:
            return
        tail = self.tts_buffer.strip()
        if tail:
            try:
                self.tts.queue_text(tail, self.tts_gen_id)
            except Exception as e:
                logger.debug(f"Error flushing TTS tail: {e}")
        self.tts_buffer = ''

    def _tts_interrupt(self) -> None:
        if self.tts is None:
            return
        try:
            self.tts.interrupt()
        except Exception as e:
            logger.debug(f"Error interrupting TTS: {e}")
        self.tts_buffer = ''
        self.tts_gen_id = None
        self.tts_first_chunk = True

    @staticmethod
    def _build_monitor_prompt(trigger: str) -> str:
        """Construct the user prompt for monitor-mode VLM calls.

        Single-instruction prompt: 'Only say YES if you see X' generalises
        better than the descriptive-then-match pattern. Combined with the
        keyword + negation parser, this catches both well-formed YES/NO
        replies and chatty ones.
        """
        return f"Only say YES if you see {trigger} in the image. Otherwise say NO."

    def _classify_monitor_response(self, answer: str) -> tuple:
        """Decide whether `answer` indicates the trigger has fired.

        Single rule: fire iff the answer begins with 'yes' (case-
        insensitive, word boundary). The monitor prompt explicitly asks
        for YES or NO, so anything else is treated as a non-fire.

        Returns (fired: bool, description: str).
        """
        if not answer:
            return (False, '')
        stripped = answer.lstrip()
        import re
        if re.match(r"^yes\b", stripped, re.IGNORECASE):
            tail = stripped[3:].lstrip(" \t\n,;:.-—–")
            return (True, tail or self.trigger_text)
        return (False, '')

    def _fire_event(self, description: str, frame) -> None:
        """Latch the alert: capture the current viewfinder for the PiP and switch to STATE_EVENT.

        The PiP shows the *current* live frame (at fire time), not the
        frame that was inferred — the latter can be 1–3 s stale by the
        time inference completes, which feels wrong in a security-alert
        UX.
        """
        if frame is None:
            return
        self.last_event_time = time.time()
        self.last_event_text = description.strip() or "(no description)"
        # Reuse the existing PiP zoom-in animation by populating frozen_frame.
        self.frozen_frame = frame.copy()
        self.pip_anim_start = time.time()
        self.pip_anim_out_start = None
        self.current_state = STATE_EVENT
        # Speak the description if TTS is enabled.
        if self.tts is not None:
            try:
                self._tts_interrupt()
                gen_id = self.tts.get_current_gen_id()
                self.tts.queue_text("Event detected. " + self.last_event_text, gen_id)
            except Exception as e:
                logger.debug(f"TTS event announcement failed: {e}")

    def _cancel_monitor_future(self) -> None:
        """Best-effort cancel of an in-flight monitor inference."""
        if self.monitor_future is not None:
            self.monitor_future.cancel()
            self.monitor_future = None

    @staticmethod
    def _draw_translucent_box(frame, x: int, y: int, w: int, h: int,
                              alpha: float = 0.55, tint=None) -> None:
        """Tint a region of `frame` in-place.

        With `tint=None`, darkens the ROI (translucent black). With a BGR
        `tint` tuple, blends the ROI toward that colour (translucent
        coloured box) — used for the red EVENT alert overlay.
        """
        h_f, w_f = frame.shape[:2]
        x0, y0 = max(0, x), max(0, y)
        x1, y1 = min(w_f, x + w), min(h_f, y + h)
        if x1 <= x0 or y1 <= y0:
            return
        roi = frame[y0:y1, x0:x1]
        if tint is None:
            cv2.addWeighted(roi, 1.0 - alpha, roi, 0.0, 0.0, dst=roi)
        else:
            import numpy as np
            tint_arr = np.full_like(roi, tint, dtype=roi.dtype)
            cv2.addWeighted(tint_arr, alpha, roi, 1.0 - alpha, 0.0, dst=roi)

    @staticmethod
    def _wrap_text(text: str, max_width_px: int, font: int, scale: float, thickness: int) -> list:
        """Greedy word-wrap; falls back to per-character break for over-long tokens."""
        if not text:
            return ['']
        lines = []
        for paragraph in text.split('\n'):
            current = ''
            for word in paragraph.split(' '):
                candidate = word if not current else current + ' ' + word
                (cw, _), _ = cv2.getTextSize(candidate, font, scale, thickness)
                if cw <= max_width_px:
                    current = candidate
                    continue
                if current:
                    lines.append(current)
                    current = ''
                # Word alone doesn't fit — break by character.
                buf = ''
                for ch in word:
                    nxt = buf + ch
                    (bw, _), _ = cv2.getTextSize(nxt, font, scale, thickness)
                    if bw <= max_width_px:
                        buf = nxt
                    else:
                        if buf:
                            lines.append(buf)
                        buf = ch
                current = buf
            lines.append(current)
        return lines

    def _draw_overlay(self, frame):
        """Compose the state-specific overlay on a copy of `frame` and return it."""
        out = frame.copy()
        h, w = out.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.5, w / 1400.0)  # ≈0.7 at 960 wide; scales sensibly elsewhere.
        thickness = 1
        text_color = (255, 255, 255)
        margin = max(8, w // 80)
        line_h = int(28 * scale) + 6

        # Red tint when an event is firing; black darken otherwise.
        box_tint = (0, 0, 180) if self.current_state == STATE_EVENT else None

        # Top banner: state header.
        if self.current_state == STATE_STREAMING:
            header = "LIVE  |  Enter: capture   T: trigger   Q/Esc: quit"
        elif self.current_state == STATE_CAPTURED:
            if self.is_recording:
                header = "Recording...  |  Space: stop"
            elif self.is_transcribing:
                header = "Transcribing..."
            elif self.speech_enabled:
                header = "Type or Space to record  |  Enter: send   Esc: cancel"
            else:
                header = "Type question  |  Enter: send   Esc: cancel"
        elif self.current_state == STATE_PROCESSING:
            header = "Processing..."
        elif self.current_state == STATE_RESULT:
            header = "Enter: continue   Q/Esc: quit"
        elif self.current_state == STATE_TRIGGER_SETUP:
            if self.is_recording:
                header = "Recording trigger...  |  Space: stop"
            elif self.is_transcribing:
                header = "Transcribing..."
            elif self.speech_enabled:
                header = "Type or Space to record  |  Enter: arm   Esc: cancel"
            else:
                header = "Type sentence  |  Enter: arm   Esc: cancel"
        elif self.current_state == STATE_MONITORING:
            header = f"Monitoring: {self.trigger_text}  |  t: change   Esc: stop"
        elif self.current_state == STATE_EVENT:
            header = "EVENT  |  Enter / t: resume monitoring   Esc: stop"
        else:
            header = ""

        banner_h = line_h + 2 * margin
        self._draw_translucent_box(out, 0, 0, w, banner_h, alpha=0.55, tint=box_tint)
        cv2.putText(out, header, (margin, margin + line_h - 6),
                    font, scale, text_color, thickness, cv2.LINE_AA)

        # Picture-in-picture inset of the captured frame (top-right), with a
        # zoom-in / zoom-out animation. `s` is a 0..1 shrink factor:
        # 0 = filling the whole frame, 1 = settled at the small PiP rect.
        anim_duration = 0.35
        s = None
        is_exiting = False
        if self.pip_anim_out_start is not None and self.frozen_frame is not None:
            elapsed = time.time() - self.pip_anim_out_start
            t = max(0.0, min(1.0, 1.0 - elapsed / anim_duration))
            s = t * t * t  # ease-in cubic for zoom-out
            is_exiting = True
        elif (self.current_state in (STATE_CAPTURED, STATE_PROCESSING, STATE_RESULT, STATE_EVENT)
                and self.frozen_frame is not None):
            elapsed = (time.time() - self.pip_anim_start) if self.pip_anim_start is not None else anim_duration
            t = max(0.0, min(1.0, elapsed / anim_duration))
            s = 1.0 - (1.0 - t) ** 3  # ease-out cubic for zoom-in

        if s is not None:
            final_w = max(160, w // 4)
            ih, iw = self.frozen_frame.shape[:2]
            final_h = max(1, int(round(final_w * ih / iw)))
            final_x = w - final_w - margin
            final_y = banner_h + margin

            # Lerp the rect between full-frame (0, 0, w, h) and the final PiP rect.
            cur_w = max(1, int(round(w + (final_w - w) * s)))
            cur_h = max(1, int(round(h + (final_h - h) * s)))
            cur_x = int(round(final_x * s))
            cur_y = int(round(final_y * s))

            # Alpha: zoom-in stays opaque; zoom-out fades as it grows.
            alpha = s if is_exiting else 1.0

            inset = cv2.resize(self.frozen_frame, (cur_w, cur_h), interpolation=cv2.INTER_AREA)
            x_clip0, x_clip1 = max(0, cur_x), min(w, cur_x + cur_w)
            y_clip0, y_clip1 = max(0, cur_y), min(h, cur_y + cur_h)
            if x_clip1 > x_clip0 and y_clip1 > y_clip0:
                src_x0 = x_clip0 - cur_x
                src_x1 = src_x0 + (x_clip1 - x_clip0)
                src_y0 = y_clip0 - cur_y
                src_y1 = src_y0 + (y_clip1 - y_clip0)
                inset_clip = inset[src_y0:src_y1, src_x0:src_x1]
                roi = out[y_clip0:y_clip1, x_clip0:x_clip1]
                if alpha >= 1.0:
                    roi[...] = inset_clip
                else:
                    cv2.addWeighted(inset_clip, alpha, roi, 1.0 - alpha, 0.0, dst=roi)
                # Border only once the inset is small enough to read as a thumbnail.
                if s > 0.6:
                    border_intensity = int(255 * alpha)
                    cv2.rectangle(out,
                                  (cur_x - 1, cur_y - 1),
                                  (cur_x + cur_w, cur_y + cur_h),
                                  (border_intensity, border_intensity, border_intensity), 1)

        # Bottom panel: question / response / trigger / event.
        panel_lines = []
        max_text_w = w - 2 * margin
        if self.current_state == STATE_CAPTURED:
            # Suppress the typing caret while recording / transcribing.
            caret = "" if (self.is_recording or self.is_transcribing) else "_"
            panel_lines = self._wrap_text("Q: " + self.user_question + caret,
                                          max_text_w, font, scale, thickness)
        elif self.current_state in (STATE_PROCESSING, STATE_RESULT):
            q_lines = self._wrap_text("Q: " + self.user_question,
                                      max_text_w, font, scale, thickness)
            response_text = self.streamed_response.strip()
            r_lines = self._wrap_text(response_text, max_text_w, font, scale, thickness) if response_text else []
            panel_lines = q_lines + ([''] + r_lines if r_lines else [])
        elif self.current_state == STATE_TRIGGER_SETUP:
            caret = "" if (self.is_recording or self.is_transcribing) else "_"
            panel_lines = self._wrap_text("What to trigger on: " + self.trigger_text + caret,
                                          max_text_w, font, scale, thickness)
        elif self.current_state == STATE_EVENT:
            panel_lines = self._wrap_text("Event detected: " + self.last_event_text,
                                          max_text_w, font, scale, thickness)
        # STATE_MONITORING: leave the bottom panel empty so the live feed is uncluttered.

        if panel_lines:
            max_panel_lines = max(1, int(h * 0.45) // line_h)
            if len(panel_lines) > max_panel_lines:
                panel_lines = panel_lines[-max_panel_lines:]
            panel_h = len(panel_lines) * line_h + 2 * margin
            panel_y = h - panel_h
            self._draw_translucent_box(out, 0, panel_y, w, panel_h, alpha=0.55, tint=box_tint)
            for i, line in enumerate(panel_lines):
                y = panel_y + margin + (i + 1) * line_h - 6
                cv2.putText(out, line, (margin, y),
                            font, scale, text_color, thickness, cv2.LINE_AA)

        return out

    def _dismiss_pip(self) -> None:
        """Start the PiP slide-out animation; cleanup happens once it finishes."""
        if self.frozen_frame is not None and self.pip_anim_out_start is None:
            self.pip_anim_out_start = time.time()
        # Stop any in-flight TTS playback so it doesn't keep talking after we move on.
        self._tts_interrupt()

    def _clear_pip(self) -> None:
        """Hard-reset all PiP / per-question state. Called after slide-out completes."""
        self.frozen_frame = None
        self.frozen_inference_frame = None
        self.user_question = ''
        self.streamed_response = ''
        self.pip_anim_start = None
        self.pip_anim_out_start = None

    def _init_camera(self) -> tuple[Callable[[], Any], Callable[[], None], str]:
        """
        Initialize the picamera2 camera with dual streams.

        Returns:
            tuple: (get_frame_callback, cleanup_callback, camera_name)
        """
        try:
            from picamera2 import Picamera2
            from libcamera import controls
            picam2 = Picamera2()
            # Dual stream: large 'main' for viewfinder, smaller 'lores' for inference.
            # Raw stream is half the sensor's pixel array (binned mode) — sensor-agnostic.
            raw_size = tuple([v // 2 for v in picam2.camera_properties['PixelArraySize']])
            config = picam2.create_preview_configuration(
                main={"size": (1920, 1080), "format": "RGB888"},
                lores={"size": (448, 448), "format": "RGB888"},
                raw={"size": raw_size},
            )
            picam2.configure(config)
            picam2.start()
            # Enable continuous autofocus only if the sensor supports it.
            if 'AfMode' in picam2.camera_controls:
                try:
                    picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
                except Exception as e:
                    logger.debug(f"Failed to set continuous autofocus: {e}")
            else:
                logger.debug("Sensor does not expose AfMode; skipping autofocus setup.")

            def get_frame():
                arrays = picam2.capture_arrays(["main", "lores"])[0]
                return arrays[0], arrays[1]

            cleanup = lambda: picam2.stop()
            return get_frame, cleanup, "RPI"
        except (ImportError, Exception) as e:
            logger.error(f"Error initializing RPI camera: {e}")
            raise

    def show_video(self):
        """Main loop: render video + overlay and handle keystrokes from the cv2 window."""
        try:
            get_frame, cleanup, _ = self._init_camera()
        except Exception:
            logger.error("Failed to initialize camera. Exiting.")
            self.running = False
            return

        # Full-screen window without the Qt toolbar/status bar.
        cv2.namedWindow('Video', cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        cv2.setWindowProperty('Video', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # Initialize Backend
        try:
            self.backend = Backend(
                hef_path=str(hef_path),
                max_tokens=MAX_TOKENS,
                temperature=TEMPERATURE,
                seed=SEED,
                system_prompt=SYSTEM_PROMPT
            )
        except Exception as e:
            logger.error(f"Failed to initialize backend: {e}")
            cleanup()
            self.running = False
            return

        # Best-effort init of speech I/O. Failures degrade gracefully.
        self._init_voice_stack()

        vlm_future = None
        viewfinder_frame = None
        inference_frame = None

        try:
            while self.running:
                # Always grab a live frame so the viewfinder stays alive in every state
                # (PiP shows the captured frame as an inset on top of the live feed).
                raw_view, raw_inf = get_frame()
                if raw_view is None:
                    logger.error("Failed to read frame from camera")
                    break
                # picamera2 'RGB888' yields BGR-ordered bytes in numpy (matching cv2). USB is BGR. No conversion needed.
                viewfinder_frame = raw_view
                inference_frame = raw_inf
                base_frame = viewfinder_frame

                # If a transcription was running, see if it's ready.
                self._poll_transcription()

                # Drain any pending streamed tokens before drawing, and feed them to TTS.
                if self.current_state == STATE_PROCESSING and self.backend is not None:
                    new_tokens = self.backend.poll_stream()
                    if new_tokens:
                        self.streamed_response += new_tokens
                        self._tts_feed(new_tokens)

                if base_frame is not None:
                    cv2.imshow('Video', self._draw_overlay(base_frame))

                # Single source of input: the cv2 window.
                key = cv2.waitKey(25)
                key = -1 if key == -1 else key & 0xFFFF
                self._handle_key(key, viewfinder_frame, inference_frame)

                # Once the slide-out animation has finished, drop the PiP state.
                if (self.pip_anim_out_start is not None
                        and time.time() - self.pip_anim_out_start >= 0.35):
                    self._clear_pip()

                if not self.running:
                    break

                # Inference completion check.
                if self.current_state == STATE_PROCESSING and vlm_future is not None and vlm_future.done():
                    if self.backend is not None:
                        tail_tokens = self.backend.poll_stream()
                        if tail_tokens:
                            self.streamed_response += tail_tokens
                            self._tts_feed(tail_tokens)
                    try:
                        result = vlm_future.result()
                        # Prefer the canonical answer (handles error/timeout strings too).
                        answer = result.get('answer') if isinstance(result, dict) else None
                        if answer:
                            self.streamed_response = answer
                    except Exception as e:
                        logger.error(f"Error getting future result: {e}")
                        self.streamed_response = f"Error processing request: {e}"
                    # Flush whatever's left in the TTS buffer so the tail sentence isn't dropped.
                    self._tts_flush()
                    vlm_future = None
                    self.current_state = STATE_RESULT

                # Schedule the inference future once we've fully entered PROCESSING.
                if (self.current_state == STATE_PROCESSING
                        and vlm_future is None
                        and self.frozen_inference_frame is not None
                        and self.user_question
                        and self.backend is not None):
                    vlm_future = self.executor.submit(
                        self.backend.vlm_inference,
                        self.frozen_inference_frame.copy(),
                        self.user_question,
                        INFERENCE_TIMEOUT
                    )

                # Monitor mode: kick off a periodic VLM call when no future is in flight.
                if (self.current_state == STATE_MONITORING
                        and self.monitor_future is None
                        and self.backend is not None
                        and inference_frame is not None
                        and time.time() - self.last_capture_time >= self.monitor_interval):
                    self.last_capture_time = time.time()
                    self._last_monitor_view = (
                        viewfinder_frame.copy() if viewfinder_frame is not None else None
                    )
                    self.monitor_future = self.executor.submit(
                        self.backend.vlm_inference,
                        inference_frame.copy(),
                        self._build_monitor_prompt(self.trigger_text),
                        INFERENCE_TIMEOUT,
                    )

                # Monitor result handling.
                if (self.current_state == STATE_MONITORING
                        and self.monitor_future is not None
                        and self.monitor_future.done()):
                    try:
                        m_result = self.monitor_future.result()
                        m_answer = (m_result.get('answer') if isinstance(m_result, dict) else '') or ''
                    except Exception as e:
                        logger.error(f"Monitor inference failed: {e}")
                        m_answer = ''
                    self.monitor_future = None
                    fired, description = self._classify_monitor_response(m_answer)
                    cooldown_ok = time.time() - self.last_event_time >= self.monitor_cooldown
                    # Honour the alert cooldown (skill recipe) so a continuously-
                    # satisfied trigger doesn't flap.
                    if fired and cooldown_ok:
                        # Pass the *current* viewfinder so the PiP is fresh,
                        # not stale by the inference latency.
                        self._fire_event(description, viewfinder_frame)

        finally:
            cleanup()
            cv2.destroyAllWindows()
            self.stop()

    def _handle_key(self, key: int, viewfinder_frame, inference_frame) -> None:
        """Dispatch a key press according to the current state."""
        if key == -1:
            return

        ENTER = (10, 13)
        BACKSPACE = (8, 127)
        ESC = 27

        if self.current_state == STATE_STREAMING:
            if key == ESC or key in (ord('q'), ord('Q')):
                self.stop()
                return
            if key in (ord('t'), ord('T')):
                # Enter trigger-setup mode for the event monitor.
                self.trigger_text = ''
                self.has_typed = False
                self._active_text_target = 'trigger_text'
                self.current_state = STATE_TRIGGER_SETUP
                return
            if key in ENTER:
                if viewfinder_frame is not None and inference_frame is not None:
                    self.frozen_frame = viewfinder_frame.copy()
                    self.frozen_inference_frame = inference_frame.copy()
                    self.user_question = ''
                    self.streamed_response = ''
                    self.has_typed = False
                    self._active_text_target = 'user_question'
                    self.pip_anim_start = time.time()
                    self.pip_anim_out_start = None  # cancel any in-progress slide-out
                    self.current_state = STATE_CAPTURED
            return

        if self.current_state == STATE_CAPTURED:
            SPACE = 32

            # Space toggles recording, but only when the user hasn't started typing —
            # once they've typed anything, Space is treated as a regular character.
            if (key == SPACE and self.speech_enabled
                    and self.audio_recorder is not None
                    and not self.is_transcribing
                    and not self.has_typed):
                if self.is_recording:
                    self._stop_recording_and_transcribe()
                else:
                    self._start_recording()
                return

            # While recording or transcribing, only Esc / Enter affect the app — block edits.
            if self.is_recording or self.is_transcribing:
                if key == ESC:
                    self._abort_recording()
                    self.is_transcribing = False
                    self.transcribe_future = None
                    self._dismiss_pip()
                    self.current_state = STATE_STREAMING
                return

            if key == ESC:
                self._abort_recording()
                self._dismiss_pip()
                self.current_state = STATE_STREAMING
                return
            if key in ENTER:
                self._abort_recording()
                if not self.user_question:
                    self.user_question = "Describe the image"
                if SAVE_FRAMES and self.frozen_inference_frame is not None:
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    cv2.imwrite(f"frame_{timestamp}.jpg", self.frozen_inference_frame)
                if self.backend is not None:
                    self.backend.clear_stream()
                self.streamed_response = ''
                self._tts_begin()
                self.current_state = STATE_PROCESSING
                return
            if key in BACKSPACE:
                self.user_question = self.user_question[:-1]
                self.has_typed = True
                return
            if 32 <= key <= 126:
                self.user_question += chr(key)
                self.has_typed = True
            return

        if self.current_state == STATE_PROCESSING:
            return  # Inputs ignored while inference is running.

        if self.current_state == STATE_RESULT:
            if key == ESC or key in (ord('q'), ord('Q')):
                self.stop()
                return
            if key in ENTER:
                self._dismiss_pip()
                self.current_state = STATE_STREAMING
            return

        if self.current_state == STATE_TRIGGER_SETUP:
            SPACE = 32

            # Space toggles recording until the user starts typing — same rule as CAPTURED.
            if (key == SPACE and self.speech_enabled
                    and self.audio_recorder is not None
                    and not self.is_transcribing
                    and not self.has_typed):
                if self.is_recording:
                    self._stop_recording_and_transcribe()
                else:
                    self._start_recording()
                return

            if self.is_recording or self.is_transcribing:
                if key == ESC:
                    self._abort_recording()
                    self.is_transcribing = False
                    self.transcribe_future = None
                    self._active_text_target = 'user_question'
                    self.current_state = STATE_STREAMING
                return

            if key == ESC:
                self._abort_recording()
                self._active_text_target = 'user_question'
                self.current_state = STATE_STREAMING
                return
            if key in ENTER:
                self._abort_recording()
                if not self.trigger_text.strip():
                    self.trigger_text = "Describe activity in the scene."
                self._active_text_target = 'user_question'
                self.last_capture_time = 0.0    # fire first monitor capture immediately
                self.last_event_time = 0.0      # cooldown not yet active
                self.current_state = STATE_MONITORING
                return
            if key in BACKSPACE:
                self.trigger_text = self.trigger_text[:-1]
                self.has_typed = True
                return
            if 32 <= key <= 126:
                self.trigger_text += chr(key)
                self.has_typed = True
            return

        if self.current_state == STATE_MONITORING:
            if key == ESC or key in (ord('q'), ord('Q')):
                self._cancel_monitor_future()
                self.trigger_text = ''
                self.current_state = STATE_STREAMING
                return
            if key in (ord('t'), ord('T')):
                self._cancel_monitor_future()
                self.trigger_text = ''
                self.has_typed = False
                self._active_text_target = 'trigger_text'
                self.current_state = STATE_TRIGGER_SETUP
            return

        if self.current_state == STATE_EVENT:
            if key == ESC:
                self._tts_interrupt()
                self._cancel_monitor_future()
                self.trigger_text = ''
                self.last_event_time = 0.0
                self._dismiss_pip()
                self.current_state = STATE_STREAMING
                return
            if key in ENTER:
                self._tts_interrupt()
                self._dismiss_pip()
                # Manual dismiss → reset the cooldown so the next detection fires
                # immediately. The cooldown's only purpose was to suppress
                # repeat-fires while already alerted, which the state machine
                # enforces anyway by not submitting in STATE_EVENT.
                self.last_event_time = 0.0
                self.last_capture_time = time.time()  # respect cadence after dismiss
                self.current_state = STATE_MONITORING
                return
            if key in (ord('t'), ord('T')):
                self._tts_interrupt()
                self._cancel_monitor_future()
                self._dismiss_pip()
                self.trigger_text = ''
                self.last_event_time = 0.0
                self.has_typed = False
                self._active_text_target = 'trigger_text'
                self.current_state = STATE_TRIGGER_SETUP

    def run(self):
        """Start the application thread."""
        self.video_thread = threading.Thread(target=self.show_video)
        self.video_thread.start()
        try:
            self.video_thread.join()
        except KeyboardInterrupt:
            self.stop()
            self.video_thread.join()

if __name__ == "__main__":
    parser = get_standalone_parser()
    parser.add_argument('--no-stt', action='store_true',
                        help='Disable speech-to-text (Whisper) input. Type the question instead.')
    parser.add_argument('--no-tts', action='store_true',
                        help='Disable text-to-speech (Piper) output of the response.')
    parser.add_argument('--monitor-interval', type=float, default=MONITOR_INTERVAL_DEFAULT,
                        help='Min seconds between successive captures in monitor mode (default 1.0).')
    parser.add_argument('--monitor-cooldown', type=float, default=MONITOR_COOLDOWN_DEFAULT,
                        help='Min seconds between fired events in monitor mode (default 30.0).')

    # Handle --list-models flag before full initialization
    handle_list_models_flag(parser, VLM_CHAT_APP)

    options_menu = parser.parse_args()

    # Resolve HEF path with auto-download (VLM is Hailo-10H only)
    hef_path = resolve_hef_path(
        options_menu.hef_path if hasattr(options_menu, 'hef_path') else None,
        app_name=VLM_CHAT_APP,
        arch=HAILO10H_ARCH
    )
    if hef_path is None:
        logger.error("Failed to resolve HEF path for VLM model. Exiting.")
        sys.exit(1)
    app = VLMChatApp(
        speech_enabled=not options_menu.no_stt,
        tts_enabled=not options_menu.no_tts,
        monitor_interval=options_menu.monitor_interval,
        monitor_cooldown=options_menu.monitor_cooldown,
    )
    app.run()
    sys.exit(0)
