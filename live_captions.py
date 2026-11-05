#!/usr/bin/env python3
"""Live Captions — offline system-audio translation to English, Windows 10/11.

One source file, native Qt interface, no web server or API key.
Run with an existing Python installation: py live_captions.py
Automatic setup downloads a private 64-bit Python 3.12.10, pip, all app libraries,
and both speech models. It installs Microsoft's C++ runtime if missing (Windows
may request administrator approval). No compiler, FFmpeg program, or API key is
needed. The caption window opens automatically when setup completes.
Translation then runs offline. Your existing Python installation is unchanged.
To repair: py live_captions.py --repair-setup

Captures the selected Windows output device, including other apps using it.
This is delayed live translation, not subtitles synchronized to the video clock.
No audio is saved; caption history exists in memory unless explicitly exported.
Settings, models and technical logs: %LOCALAPPDATA%/OfflineLiveCaptions
"""
from __future__ import annotations

import collections
import ctypes
import dataclasses
import datetime as dt
import gc
import importlib.util
import json
import logging
from logging.handlers import RotatingFileHandler
import math
import os
from pathlib import Path
import queue
import struct
import subprocess
import sys
import threading
import time
import traceback

APP_NAME = "Live Captions"
VERSION = "1.1.0"
DEPENDENCIES = [
    ("PySide6", "PySide6-Essentials==6.8.3"),
    ("faster_whisper", "faster-whisper==1.2.1"),
    ("pyaudiowpatch", "PyAudioWPatch==0.2.12.7"),
    ("numpy", "numpy==1.26.4"),
    ("scipy", "scipy==1.13.1"),
    ("huggingface_hub", "huggingface-hub>=0.24,<1"),
    ("ctranslate2", "ctranslate2==4.6.0"),
    ("onnxruntime", "onnxruntime==1.22.1"),
    ("av", "av==14.2.0"),
]
LANGUAGES = {
    "Automatic detection": None, "Japanese": "ja", "Russian": "ru",
    "Hindi": "hi", "English": "en", "Spanish": "es", "French": "fr",
    "German": "de", "Chinese": "zh", "Korean": "ko", "Arabic": "ar",
    "Portuguese": "pt", "Italian": "it", "Turkish": "tr", "Bengali": "bn",
    "Tamil": "ta", "Telugu": "te", "Marathi": "mr", "Urdu": "ur",
    "Indonesian": "id", "Vietnamese": "vi", "Thai": "th", "Dutch": "nl",
    "Polish": "pl", "Ukrainian": "uk", "Persian": "fa", "Swedish": "sv",
    "Romanian": "ro", "Greek": "el", "Hebrew": "he", "Malay": "ms",
    "Finnish": "fi", "Czech": "cs", "Danish": "da", "Hungarian": "hu",
    "Norwegian": "no", "Filipino / Tagalog": "tl",
}
MODEL_LABELS = {"Quality — Whisper small": "small", "Fast — Whisper base": "base"}
MODEL_FILES = ("model.bin", "config.json", "tokenizer.json")
LOG = logging.getLogger("live_captions")


def console_python():
    executable = Path(sys.executable)
    return str(executable.with_name("python.exe") if executable.name.lower() == "pythonw.exe" else executable)


def app_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA")
    return (Path(base) if base else Path.home() / ".local" / "share") / "OfflineLiveCaptions"


def init_logging(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    handler = RotatingFileHandler(root / "app.log", maxBytes=1_000_000, backupCount=2, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    LOG.addHandler(handler)
    LOG.setLevel(logging.INFO)


@dataclasses.dataclass
class Settings:
    model: str = "small"
    language: str = "Automatic detection"
    device_name: str = ""
    threads: int = 2
    chunk_seconds: float = 5.0
    silence_seconds: float = 0.6
    gate_db: int = -48
    font_size: int = 28
    opacity: int = 82
    width_percent: int = 74
    bottom_percent: int = 10
    display_seconds: int = 8
    screen_name: str = ""
    locked: bool = False
    overlay_x: int = -1
    overlay_y: int = -1

    @classmethod
    def load(cls, path: Path):
        default = cls()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                return default
            for field in dataclasses.fields(cls):
                value = raw.get(field.name, getattr(default, field.name))
                old = getattr(default, field.name)
                if type(value) is type(old) or (type(old) is float and type(value) in (float, int)):
                    setattr(default, field.name, value)
            if default.model not in ("base", "small"):
                default.model = "small"
            if default.language not in LANGUAGES:
                default.language = "Automatic detection"
            bounds = {"threads": (1, min(4, os.cpu_count() or 2)), "chunk_seconds": (2, 10),
                      "silence_seconds": (0.3, 1.5), "gate_db": (-70, -20),
                      "font_size": (16, 56), "opacity": (15, 100), "width_percent": (35, 95),
                      "bottom_percent": (0, 70), "display_seconds": (3, 30)}
            for key, (low, high) in bounds.items():
                value = getattr(default, key)
                if not math.isfinite(value):
                    value = getattr(cls(), key)
                setattr(default, key, min(high, max(low, value)))
        except (OSError, ValueError, TypeError):
            LOG.warning("Settings unavailable or invalid; using defaults")
        return default

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(dataclasses.asdict(self), indent=2, allow_nan=False), encoding="utf-8")
        os.replace(tmp, path)


def model_ready(root: Path, model: str) -> bool:
    folder = root / "models" / model
    try:
        marker = json.loads((folder / "ready.json").read_text(encoding="utf-8"))
        return marker.get("model") == model and all(
            (folder / name).is_file() and (folder / name).stat().st_size > 0 for name in MODEL_FILES
        )
    except (OSError, ValueError, TypeError, AttributeError):
        return False


class LatestQueue:
    """One pending utterance. Replace old pending audio instead of growing RAM."""
    def __init__(self):
        self.queue = queue.Queue(maxsize=1)
        self.dropped = 0

    def put(self, item):
        try:
            self.queue.put_nowait(item)
        except queue.Full:
            try:
                self.queue.get_nowait()
                self.dropped += 1
            except queue.Empty:
                pass
            self.queue.put_nowait(item)  # one producer, one consumer

    def get(self, timeout=0.2):
        return self.queue.get(timeout=timeout)


class Segmenter:
    """Energy endpointing with pre-roll. Silero VAD verifies speech at inference.

    push accepts float32, 16 kHz mono arrays, normally 100 ms per call.
    This stage avoids running the recognizer repeatedly during silence.
    """
    def __init__(self, maximum=5.0, silence=0.6, gate_db=-48):
        self.maximum = maximum
        self.silence_limit = silence
        self.threshold = 10 ** (gate_db / 20)
        self.pre = collections.deque(maxlen=3)
        self.parts = []
        self.duration = 0.0
        self.quiet = 0.0
        self.voiced = 0.0

    def push(self, samples):
        import numpy as np
        duration = len(samples) / 16000.0
        rms = float(np.sqrt(np.mean(samples * samples) + 1e-12))
        active = rms >= self.threshold
        if not self.parts:
            if not active:
                self.pre.append(samples)
                return None
            self.parts = list(self.pre)
            self.pre.clear()
            self.duration = sum(len(x) for x in self.parts) / 16000.0
        self.parts.append(samples)
        self.duration += duration
        if active:
            self.voiced += duration
            self.quiet = 0.0
        else:
            self.quiet += duration
        if self.duration + 1e-6 < self.maximum and self.quiet + 1e-6 < self.silence_limit:
            return None
        result = np.concatenate(self.parts) if self.voiced >= 0.25 else None
        # Keep 200 ms of context when splitting continuous speech. Translation
        # overlap is removed later only when words match exactly.
        tail = self.parts[-2:]
        self.parts = []
        self.duration = self.quiet = self.voiced = 0.0
        self.pre.extend(tail)
        return result


def remove_overlap(previous: str, current: str) -> str:
    """Only remove a substantial exact boundary match, not repeated short phrases."""
    before, after = previous.split(), current.split()
    def normalized(words):
        return [word.casefold().strip(".,!?;:\"'()[]") for word in words]
    a, b = normalized(before), normalized(after)
    for count in range(min(16, len(a), len(b)), 3, -1):
        if a[-count:] == b[:count]:
            return " ".join(after[count:])
    return current.strip()


def audio_devices():
    import pyaudiowpatch as pa
    audio = pa.PyAudio()
    try:
        result = []
        default_index = None
        try:
            default_index = int(audio.get_default_wasapi_loopback()["index"])
        except (OSError, ValueError, LookupError):
            pass
        for device in audio.get_loopback_device_info_generator():
            if int(device["maxInputChannels"]) > 0:
                result.append({"index": int(device["index"]), "name": device["name"],
                               "default": int(device["index"]) == default_index})
        return result
    finally:
        audio.terminate()


class CaptionEngine:
    """All expensive work is outside the GUI thread. Only emits plain data."""
    def __init__(self, root, settings, device_index, emit):
        self.root = root
        self.settings = dataclasses.replace(settings)
        self.device_index = device_index
        self.emit = emit
        self.stop_event = threading.Event()
        self.pending = LatestQueue()
        self.thread = None
        self.capture_thread = None
        self.capture_error = None

    def start(self):
        self.thread = threading.Thread(target=self._run, name="recognizer", daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_event.set()

    def _capture(self):
        audio = None
        stream = None
        try:
            import numpy as np
            import pyaudiowpatch as pa
            from scipy.signal import resample_poly
            audio = pa.PyAudio()
            # Recheck the identity: Windows can reorder device indices.
            info = audio.get_device_info_by_index(self.device_index)
            if not info.get("isLoopbackDevice") or info["name"] != self.settings.device_name:
                raise RuntimeError("Audio device changed. Stop, refresh devices, and select it again.")
            channels = int(info["maxInputChannels"])
            rate = int(round(info["defaultSampleRate"]))
            frames = max(1, round(rate / 10))
            divisor = math.gcd(rate, 16000)
            segmenter = Segmenter(self.settings.chunk_seconds, self.settings.silence_seconds,
                                  self.settings.gate_db)
            stream = audio.open(format=pa.paInt16, channels=channels, rate=rate, input=True,
                                input_device_index=self.device_index, frames_per_buffer=frames)
            self.emit("status", "Listening — play a video on the selected output device")
            meter_time = 0.0
            last_read = last_idle_feed = time.monotonic()
            overflows = 0
            while not self.stop_event.is_set():
                try:
                    # Poll instead of blocking on an idle output, so Stop works
                    # even when there is no audio playing.
                    if stream.get_read_available() < frames:
                        now = time.monotonic()
                        # Some loopback drivers stop delivering packets when a
                        # player pauses. Finish the last phrase even in that case.
                        if now - last_read > 0.25 and now - last_idle_feed >= 0.1:
                            utterance = segmenter.push(np.zeros(1600, dtype=np.float32))
                            if utterance is not None:
                                self.pending.put((utterance, now))
                                self.emit("dropped", self.pending.dropped)
                            self.emit("level", -80.0)
                            last_idle_feed = now
                        self.stop_event.wait(0.02)
                        continue
                    data = stream.read(frames, exception_on_overflow=True)
                    last_read = last_idle_feed = time.monotonic()
                except OSError as exc:
                    if getattr(exc, "errno", None) == pa.paInputOverflowed or (exc.args and exc.args[0] == pa.paInputOverflowed):
                        overflows += 1
                        self.emit("warning", f"Audio buffer overflow ({overflows}). Try Fast mode or fewer CPU threads.")
                        continue
                    raise
                samples = np.frombuffer(data, dtype=np.int16).astype(np.float32)
                samples = samples.reshape(-1, channels).mean(axis=1) / 32768.0
                if rate != 16000:
                    samples = resample_poly(samples, 16000 // divisor, rate // divisor).astype(np.float32)
                now = time.monotonic()
                if now - meter_time > 0.2:
                    rms = float(np.sqrt(np.mean(samples * samples) + 1e-12))
                    self.emit("level", max(-80.0, 20 * math.log10(max(rms, 1e-8))))
                    meter_time = now
                utterance = segmenter.push(samples)
                if utterance is not None:
                    self.pending.put((utterance, now))
                    self.emit("dropped", self.pending.dropped)
        except Exception as exc:
            LOG.exception("Audio capture failed")
            self.capture_error = str(exc)
            self.stop_event.set()
        finally:
            if stream is not None:
                try:
                    stream.close()
                except Exception:
                    LOG.exception("Closing audio stream")
            if audio is not None:
                audio.terminate()

    def _run(self):
        model = None
        try:
            from faster_whisper import WhisperModel
            if not model_ready(self.root, self.settings.model):
                raise RuntimeError("Download the selected model before starting.")
            self.emit("status", "Loading local model — please wait…")
            model = WhisperModel(str(self.root / "models" / self.settings.model), device="cpu",
                                 compute_type="int8", cpu_threads=self.settings.threads,
                                 num_workers=1, local_files_only=True)
            if self.stop_event.is_set():
                return
            self.capture_thread = threading.Thread(target=self._capture, name="wasapi", daemon=True)
            self.capture_thread.start()
            previous = ""
            previous_end = 0.0
            while not self.stop_event.is_set():
                try:
                    samples, end_time = self.pending.get()
                except queue.Empty:
                    continue
                # Drop even the latest pending item if it became stale during a
                # very slow inference. Queue size alone cannot bound its age.
                if time.monotonic() - end_time > 12:
                    self.pending.dropped += 1
                    self.emit("dropped", self.pending.dropped)
                    continue
                started = time.monotonic()
                segments, info = model.transcribe(
                    samples, task="translate", language=LANGUAGES[self.settings.language],
                    beam_size=3 if self.settings.model == "small" else 1,
                    best_of=1, temperature=0, condition_on_previous_text=False,
                    vad_filter=True,
                    vad_parameters={"min_silence_duration_ms": 350, "speech_pad_ms": 150},
                    no_speech_threshold=0.6, log_prob_threshold=-1.0,
                    compression_ratio_threshold=2.4, word_timestamps=False, max_new_tokens=192,
                )
                accepted = []
                for segment in segments:
                    if self.stop_event.is_set():
                        break
                    if segment.avg_logprob < -1.0 or segment.no_speech_prob > 0.65:
                        continue
                    if segment.compression_ratio > 2.4:
                        continue
                    accepted.append(segment.text.strip())
                if self.stop_event.is_set():
                    break
                text = " ".join(accepted).strip()
                if end_time - previous_end < self.settings.chunk_seconds + 2:
                    text = remove_overlap(previous, text)
                processing = time.monotonic() - started
                duration = len(samples) / 16000
                delay = time.monotonic() - end_time
                self.emit("metrics", {"processing": processing, "audio": duration,
                                       "delay": delay, "language": info.language,
                                       "probability": info.language_probability})
                if text:
                    previous, previous_end = text, end_time
                    self.emit("caption", text)
            if self.capture_error:
                raise RuntimeError(self.capture_error)
        except Exception as exc:
            LOG.exception("Recognition failed")
            self.emit("error", str(exc))
        finally:
            self.stop_event.set()
            if self.capture_thread:
                # Do not signal stopped while the capture device is still owned.
                self.capture_thread.join()
            del model
            gc.collect()
            self.emit("finished", None)


def download_model(root, model, emit):
    """The only model network operation. Validate before marking it ready."""
    try:
        from huggingface_hub import snapshot_download
        from faster_whisper import WhisperModel
        folder = root / "models" / model
        folder.mkdir(parents=True, exist_ok=True)
        (folder / "ready.json").unlink(missing_ok=True)
        emit("status", f"Downloading {model} from Hugging Face — this can take several minutes…")
        snapshot_download(
            repo_id=f"Systran/faster-whisper-{model}", local_dir=str(folder),
            allow_patterns=["config.json", "model.bin", "tokenizer.json", "vocabulary.*", "preprocessor_config.json"],
            max_workers=2,
        )
        emit("status", f"Checking {model} locally…")
        check = WhisperModel(str(folder), device="cpu", compute_type="int8", cpu_threads=1,
                             num_workers=1, local_files_only=True)
        del check
        gc.collect()
        if not all((folder / name).is_file() and (folder / name).stat().st_size > 0 for name in MODEL_FILES):
            raise RuntimeError("Model download incomplete. Click Download again to resume.")
        (folder / "ready.json").write_text(json.dumps({"model": model, "version": 1}), encoding="utf-8")
        emit("download_done", model)
    except Exception as exc:
        LOG.exception("Model download failed")
        emit("download_error", str(exc))


class SetupCancelled(Exception):
    pass


def runtime_dir(root):
    return root / "python-3.12.10"


def runtime_python(root):
    return runtime_dir(root) / "python.exe"


def runtime_complete(root):
    folder = runtime_dir(root)
    return all((folder / name).is_file() for name in (
        "python.exe", "python312.dll", "python312.zip", "python312._pth", ".runtime-complete"))


def setup_fingerprint():
    import hashlib
    payload = json.dumps({"revision": 2, "python": "3.12.10", "packages": DEPENDENCIES}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def setup_ready(root):
    try:
        data = json.loads((root / "setup-ready.json").read_text(encoding="utf-8"))
        return (data.get("fingerprint") == setup_fingerprint() and runtime_complete(root)
                and all(model_ready(root, model) for model in ("base", "small")))
    except (OSError, ValueError, TypeError, AttributeError):
        return False


def acquire_setup_lock(root):
    handle = (root / "setup.lock").open("a+b")
    try:
        handle.seek(0, 2)
        if handle.tell() == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        if sys.platform == "win32":
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        return handle
    except Exception:
        handle.close()
        raise


class SetupRunner:
    """Install into app data, never into the user's global Python environment."""
    def __init__(self, root, emit, repair=False):
        self.root = Path(root)
        self.emit = emit
        self.repair = repair
        self.cancelled = threading.Event()
        self.process = None
        self.process_lock = threading.Lock()
        self.tail = collections.deque(maxlen=24)

    def cancel(self):
        self.cancelled.set()
        with self.process_lock:
            if self.process is not None:
                try:
                    self.process.terminate()
                except OSError:
                    pass

    def check_cancel(self):
        if self.cancelled.is_set():
            raise SetupCancelled("Setup cancelled. Open the file again to resume.")

    def log(self, text):
        text = str(text)
        for line in text.splitlines():
            self.tail.append(line)
        self.emit(text if text.endswith("\n") else text + "\n")

    def command(self, args, title, allow_failure=False):
        self.check_cancel()
        self.log("\n" + title)
        process = subprocess.Popen(
            [str(arg) for arg in args], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            stdin=subprocess.DEVNULL, text=True, encoding="utf-8", errors="replace",
            env={**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"},
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        with self.process_lock:
            self.process = process
            if self.cancelled.is_set():
                process.terminate()
        try:
            for line in process.stdout:
                self.check_cancel()
                # Model-download subprocess emits JSON status messages.
                try:
                    event = json.loads(line)
                    if isinstance(event, list) and len(event) == 2:
                        line = str(event[1])
                except (ValueError, TypeError):
                    pass
                self.log(line)
            code = process.wait()
            self.check_cancel()
            if code and not allow_failure:
                details = "\n".join(self.tail)[-6000:]
                raise RuntimeError(f"{title} failed (exit {code}).\n\n{details}")
            return code
        finally:
            if process.poll() is None:
                try:
                    process.terminate()
                except OSError:
                    pass
            process.wait()
            if process.stdout:
                process.stdout.close()
            with self.process_lock:
                if self.process is process:
                    self.process = None

    def fetch(self, url, destination):
        """HTTPS download with size checks, atomic completion, retries and logging."""
        import urllib.request
        destination = Path(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        part = destination.with_suffix(destination.suffix + ".part")
        for attempt in range(3):
            self.check_cancel()
            try:
                self.log(f"Downloading {destination.name} (attempt {attempt + 1}/3)…")
                request = urllib.request.Request(url, headers={"User-Agent": f"OfflineLiveCaptions/{VERSION}"})
                with urllib.request.urlopen(request, timeout=30) as response, part.open("wb") as target:
                    total = int(response.headers.get("Content-Length", 0))
                    received = 0
                    last_report = 0.0
                    while True:
                        self.check_cancel()
                        block = response.read(256 * 1024)
                        if not block:
                            break
                        target.write(block)
                        received += len(block)
                        now = time.monotonic()
                        if now - last_report >= 1:
                            suffix = f" / {total / 1024**2:.1f} MB" if total else " MB"
                            self.log(f"  {destination.name}: {received / 1024**2:.1f}{suffix}")
                            last_report = now
                    if received == 0 or (total and received != total):
                        raise OSError("Download incomplete. The connection ended early.")
                os.replace(part, destination)
                return destination
            except SetupCancelled:
                raise
            except Exception as exc:
                self.log(f"Download error: {exc}")
                if attempt == 2:
                    raise RuntimeError(f"Could not download {destination.name}: {exc}.\n"
                                       "Check your internet connection and reopen the file to retry.") from exc
                if self.cancelled.wait(attempt + 1):
                    self.check_cancel()

    def extract_python(self, archive):
        import zipfile
        folder = runtime_dir(self.root)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / ".runtime-complete").unlink(missing_ok=True)
        base = folder.resolve()
        with zipfile.ZipFile(archive) as bundle:
            for entry in bundle.infolist():
                self.check_cancel()
                target = (folder / entry.filename).resolve()
                if target != base and base not in target.parents:
                    raise RuntimeError("Invalid path in Python download.")
                bundle.extract(entry, folder)
        # The official embeddable distribution isolates its library path. Enable
        # only this private environment's site-packages, including pip and Qt.
        site = folder / "Lib" / "site-packages"
        site.mkdir(parents=True, exist_ok=True)
        (folder / "python312._pth").write_text(
            "python312.zip\n.\nLib/site-packages\nimport site\n", encoding="utf-8")
        self.check_cancel()
        (folder / ".runtime-complete").write_text("3.12.10", encoding="ascii")

    def runtime_available(self):
        return self.command(
            [runtime_python(self.root), "-c",
             "import ctypes; [ctypes.WinDLL(n) for n in ('vcruntime140.dll','vcruntime140_1.dll','msvcp140.dll')]"],
            "Checking Microsoft C++ runtime", allow_failure=True,
        ) == 0

    def install_vc_runtime(self):
        from ctypes import wintypes
        installer = self.fetch("https://aka.ms/vs/17/release/vc_redist.x64.exe",
                               self.root / "setup-downloads" / "vc_redist.x64.exe")
        self.check_cancel()
        self.log("Installing Microsoft's C++ runtime. Approve the Windows administrator prompt if it appears.\n"
                 "This Microsoft installer must finish before setup can continue or close.")
        class ExecuteInfo(ctypes.Structure):
            _fields_ = [("cbSize", wintypes.DWORD), ("fMask", wintypes.ULONG), ("hwnd", wintypes.HWND),
                        ("lpVerb", wintypes.LPCWSTR), ("lpFile", wintypes.LPCWSTR),
                        ("lpParameters", wintypes.LPCWSTR), ("lpDirectory", wintypes.LPCWSTR),
                        ("nShow", ctypes.c_int), ("hInstApp", wintypes.HINSTANCE),
                        ("lpIDList", ctypes.c_void_p), ("lpClass", wintypes.LPCWSTR),
                        ("hkeyClass", wintypes.HKEY), ("dwHotKey", wintypes.DWORD),
                        ("hIcon", wintypes.HANDLE), ("hProcess", wintypes.HANDLE)]
        shell = ctypes.WinDLL("shell32", use_last_error=True)
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        shell.ShellExecuteExW.argtypes = [ctypes.POINTER(ExecuteInfo)]
        shell.ShellExecuteExW.restype = wintypes.BOOL
        kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        kernel.WaitForSingleObject.restype = wintypes.DWORD
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.GetExitCodeProcess.restype = wintypes.BOOL
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        info = ExecuteInfo()
        info.cbSize = ctypes.sizeof(info)
        info.fMask = 0x40 | 0x100  # SEE_MASK_NOCLOSEPROCESS | SEE_MASK_NOASYNC
        info.lpVerb = "runas"
        info.lpFile = str(installer)
        info.lpParameters = "/install /passive /norestart"
        info.nShow = 1
        if not shell.ShellExecuteExW(ctypes.byref(info)):
            code = ctypes.get_last_error()
            if code == 1223:
                raise RuntimeError("The Windows administrator prompt was declined.\n"
                                   "Run setup again and approve Microsoft's C++ runtime installer.")
            raise ctypes.WinError(code)
        try:
            while True:
                status = kernel.WaitForSingleObject(info.hProcess, 250)
                if status == 0:
                    break
                if status != 258:
                    raise ctypes.WinError(ctypes.get_last_error())
            code = wintypes.DWORD()
            if not kernel.GetExitCodeProcess(info.hProcess, ctypes.byref(code)):
                raise ctypes.WinError(ctypes.get_last_error())
            if code.value not in (0, 1638, 3010):
                raise RuntimeError(f"Microsoft's C++ runtime installer failed (exit {code.value}).")
            if code.value == 3010:
                self.log("Windows reports that a restart may be needed for the C++ runtime.")
        finally:
            kernel.CloseHandle(info.hProcess)
        self.check_cancel()

    def validate(self, allow_failure=False):
        modules = [name for name, _ in DEPENDENCIES]
        script = ("import sys, struct; assert sys.version_info[:2] == (3,12); "
                  "assert struct.calcsize('P') == 8; "
                  + "; ".join("__import__(" + repr(name) + ")" for name in modules)
                  + "; from PySide6 import QtWidgets; from scipy.signal import resample_poly; "
                  "from faster_whisper.vad import get_vad_model; get_vad_model(); "
                  "print('All application libraries loaded successfully.')")
        return self.command([runtime_python(self.root), "-c", script], "Checking installed libraries", allow_failure)

    def run(self):
        self.root.mkdir(parents=True, exist_ok=True)
        self.log(f"Live Captions {VERSION} automatic setup\n"
                 f"Private installation folder: {self.root}\n"
                 f"Starting Python: {sys.version.split()[0]} ({struct.calcsize('P') * 8}-bit)\n"
                 "Your existing Python installation will not be modified.")
        completed = setup_ready(self.root) and not self.repair
        if completed and self.validate(allow_failure=True) == 0:
            self.log("Setup is already complete. Starting Live Captions…")
            return
        self.check_cancel()
        (self.root / "setup-ready.json").unlink(missing_ok=True)
        if not runtime_complete(self.root) or self.repair or completed:
            import zipfile
            archive = self.root / "setup-downloads" / "python-3.12.10-embed-amd64.zip"
            valid_archive = False
            try:
                with zipfile.ZipFile(archive) as bundle:
                    valid_archive = bundle.testzip() is None and {
                        "python.exe", "python312.dll", "python312.zip", "_ssl.pyd"
                    }.issubset(bundle.namelist())
            except (OSError, zipfile.BadZipFile):
                pass
            if not valid_archive:
                archive = self.fetch(
                    "https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip", archive)
            self.log("Extracting private 64-bit Python 3.12.10…")
            self.extract_python(archive)
            if not runtime_complete(self.root):
                raise RuntimeError("The private Python download did not extract completely. Retry setup.")
        else:
            # Also repairs a missing/changed private library path file.
            (runtime_dir(self.root) / "Lib" / "site-packages").mkdir(parents=True, exist_ok=True)
            (runtime_dir(self.root) / "python312._pth").write_text(
                "python312.zip\n.\nLib/site-packages\nimport site\n", encoding="utf-8")
        if not self.runtime_available():
            self.install_vc_runtime()
            if not self.runtime_available():
                raise RuntimeError("Microsoft's C++ runtime is still unavailable. Restart Windows, then open this file again.")
        pip_exists = self.command([runtime_python(self.root), "-m", "pip", "--version"],
                                  "Checking package installer", allow_failure=True) == 0
        if not pip_exists:
            script = self.fetch("https://bootstrap.pypa.io/get-pip.py", self.root / "setup-downloads" / "get-pip.py")
            self.command([runtime_python(self.root), script, "pip>=24,<27", "setuptools>=70,<83", "wheel>=0.43,<1",
                          "--only-binary=:all:", "--disable-pip-version-check", "--no-warn-script-location"],
                         "Installing pip, setuptools and wheel")
        pip = [runtime_python(self.root), "-m", "pip", "--isolated", "--disable-pip-version-check", "--no-input",
               "install", "--upgrade", "--only-binary=:all:", "--retries", "3", "--timeout", "30",
               "--no-warn-script-location"]
        self.command(pip + ["pip>=24,<27", "setuptools>=70,<83", "wheel>=0.43,<1"], "Updating package installer")
        packages = [spec for _, spec in DEPENDENCIES]
        if self.repair or completed:
            packages = ["--force-reinstall"] + packages
        self.command(pip + packages, "Installing all speech, audio and interface libraries")
        self.validate()
        for model in ("small", "base"):
            if not model_ready(self.root, model):
                self.command([runtime_python(self.root), Path(__file__).resolve(), "--download-model", model,
                              "--data-dir", self.root], f"Downloading and checking {model} speech model")
            if not model_ready(self.root, model):
                raise RuntimeError(f"The {model} model did not finish downloading. Reopen the file to resume.")
        self.check_cancel()
        tmp = self.root / "setup-ready.tmp"
        tmp.write_text(json.dumps({"fingerprint": setup_fingerprint()}), encoding="utf-8")
        os.replace(tmp, self.root / "setup-ready.json")
        self.log("All components and both speech models are ready. Starting Live Captions…")


def native_error(title, detail):
    if sys.platform == "win32":
        ctypes.windll.user32.MessageBoxW(None, str(detail), str(title), 0x10)
    else:
        print(f"{title}: {detail}", file=sys.stderr)


def bootstrap_dependencies(root=None, repair=False):
    root = Path(root or app_dir())
    root.mkdir(parents=True, exist_ok=True)
    log_path = root / "setup.log"
    messages = queue.Queue()
    runner = SetupRunner(root, lambda value: messages.put(("line", value)), repair=repair)
    try:
        import tkinter as tk
        from tkinter import messagebox, scrolledtext
        window = tk.Tk()
    except Exception:
        # Some Python distributions omit Tk. Setup itself only needs the standard
        # library, and the finished application uses Qt, so it can still proceed.
        with log_path.open("w", encoding="utf-8", buffering=1) as setup_log:
            runner.emit = setup_log.write
            try:
                runner.run()
                return True
            except Exception as exc:
                setup_log.write(traceback.format_exc())
                native_error("Setup failed", f"{exc}\n\nDetails: {log_path}")
                return False
    window.title(f"Live Captions {VERSION} — automatic setup")
    window.geometry("850x620")
    window.minsize(650, 450)
    window.configure(bg="#111827")
    tk.Label(window, text="Setting up Live Captions", font=("Segoe UI", 21, "bold"),
             fg="white", bg="#111827").pack(anchor="w", padx=20, pady=(20, 8))
    tk.Label(window, text="Downloads compatible Python, all libraries and both speech models.\n"
             "Setup starts automatically. The caption window opens when it finishes.",
             fg="#cbd5e1", bg="#111827", justify="left").pack(anchor="w", padx=20)
    status = tk.StringVar(value="Preparing automatic setup…")
    tk.Label(window, textvariable=status, fg="#93c5fd", bg="#111827", anchor="w",
             wraplength=790, justify="left").pack(fill="x", padx=20, pady=(12, 0))
    output = scrolledtext.ScrolledText(window, height=18, bg="#080f1d", fg="#d1d5db", wrap="word")
    output.pack(fill="both", expand=True, padx=20, pady=14)
    state = {"running": False, "success": False, "closing": False, "log": None, "attempted": False}
    def emit(text):
        if state["log"]:
            state["log"].write(text)
            state["log"].flush()
        messages.put(("line", text))
    runner.emit = emit
    def install():
        if state["running"]:
            return
        state["running"] = True
        state["closing"] = False
        if state["attempted"]:
            runner.repair = True
        state["attempted"] = True
        runner.tail.clear()
        runner.cancelled.clear()
        try:
            state["log"] = log_path.open("w", encoding="utf-8")
        except OSError as exc:
            state["running"] = False
            messagebox.showerror("Cannot write setup log", f"{exc}\n\nCheck write access to {root} and retry setup.")
            return
        output.delete("1.0", "end")
        retry.configure(state="disabled")
        cancel.configure(state="normal", text="Cancel setup")
        def worker():
            try:
                runner.run()
                messages.put(("done", None))
            except SetupCancelled as exc:
                emit(str(exc) + "\n")
                messages.put(("cancelled", str(exc)))
            except Exception as exc:
                LOG.exception("Automatic setup failed")
                emit(traceback.format_exc())
                messages.put(("error", str(exc)))
        threading.Thread(target=worker, name="automatic-setup", daemon=True).start()
    def poll():
        try:
            while True:
                kind, value = messages.get_nowait()
                if kind == "line":
                    output.insert("end", value)
                    output.see("end")
                    lines = value.strip().splitlines()
                    if lines:
                        status.set(lines[-1][:240])
                else:
                    state["running"] = False
                    if state["log"]:
                        state["log"].close()
                        state["log"] = None
                    if state["closing"]:
                        window.destroy()
                        return
                    if kind == "done":
                        state["success"] = True
                        window.destroy()
                        return
                    retry.configure(state="normal", text="Retry / repair setup")
                    cancel.configure(text="Close")
                    if kind == "error":
                        status.set("Setup failed — the actual error is shown below and saved in setup.log")
                        messagebox.showerror("Setup failed", value[-4500:] + f"\n\nFull setup log: {log_path}")
                    else:
                        status.set("Setup cancelled. Retry to continue.")
        except queue.Empty:
            pass
        window.after(100, poll)
    def close():
        if state["running"]:
            state["closing"] = True
            runner.cancel()
            status.set("Cancelling… Waiting for any running installer to finish safely.")
            cancel.configure(state="disabled")
        else:
            window.destroy()
    def copy_log():
        window.clipboard_clear()
        try:
            window.clipboard_append(log_path.read_text(encoding="utf-8"))
            status.set("Setup log copied. You can paste it when asking for help.")
        except OSError:
            status.set("No setup log has been written yet.")
    row = tk.Frame(window, bg="#111827")
    row.pack(fill="x", padx=20, pady=(0, 16))
    retry = tk.Button(row, text="Install / repair", command=install, padx=16, pady=7)
    retry.pack(side="left")
    tk.Button(row, text="Copy setup log", command=copy_log, padx=16, pady=7).pack(side="left", padx=10)
    cancel = tk.Button(row, text="Cancel setup", command=close, padx=16, pady=7)
    cancel.pack(side="right")
    window.protocol("WM_DELETE_WINDOW", close)
    window.after(100, poll)
    window.after(300, install)
    window.mainloop()
    return state["success"]


def launch_managed(root):
    with (root / "launch.log").open("w", encoding="utf-8") as output:
        subprocess.Popen([str(runtime_python(root)), str(Path(__file__).resolve()), "--managed-run", "--data-dir", str(root)],
                         stdout=output, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))

HELP = """QUICK START
1. Open this Python file. Setup automatically installs a private Python runtime,
   app libraries and both caption models, then opens the caption window.
   If Windows asks, approve Microsoft's C++ runtime installer. First setup needs
   internet and several hundred MB of downloads. Later translation works offline.
2. Play browser audio through your normal speakers/headphones. Select that
   output device, or use the Windows default from the last refresh. Click Start.
3. The audio meter should move. English captions appear after a phrase is
   processed. Stop before changing model, language or capture settings.
4. Minimize to tray to keep captions running. Double-click the tray icon to
   restore settings. Exit in the tray menu stops capture and closes the app.

YOUR i5-6500 / 8 GB PC
Quality uses multilingual Whisper small, INT8 and beam size 3. Fast uses
multilingual base, INT8 and beam size 1. Start with two CPU threads. More threads
may improve recognition speed but compete with your browser. Use Fast if
processing time exceeds audio length or the skipped-audio counter rises.
These models cannot guarantee accurate, instant translation on this CPU.

LATENCY AND SKIPPED AUDIO
Chunk length controls how much speech is collected before translation. Shorter
chunks respond sooner but lose sentence context. Silence ends a chunk early.
Caption delay measures time since the chunk ended; total delay also includes
collecting the phrase. One pending chunk is kept; older pending chunks are
replaced when recognition is slow. This intentionally skips speech to stay
closer to live playback. Pause/seek in the browser is not detected by this app;
Stop/Start clears its queued audio. Captions are not synchronized to video time.

LANGUAGES
Output is always English. Select Japanese, Russian, Hindi or another spoken
language for more reliable recognition. Automatic detection runs on each chunk
and can misidentify short phrases. Quality varies by language and recording.
There is no separate translation model and no cloud translation service.

AUDIO TROUBLESHOOTING
WASAPI loopback captures all apps playing through the selected output, not just
the browser. Browser mute / very low volume / Windows per-app routing / Bluetooth
profile changes can prevent capture. If you connect headphones, change the
Windows default, or switch output while running: Stop, Refresh, then Start.
Protected audio and exclusive-mode apps may prevent capture. Do not select a
microphone: the list contains output-loopback devices only. For quiet dialogue,
lower the speech gate (more negative dB); a lower gate also admits more noise.
Silero VAD filters non-speech before recognition. Music may still cause mistakes.

CAPTION OVERLAY
Choose a display on the Captions tab. Drag the overlay to reposition it while
unlocked. Lock makes it click-through. Unlock or reset from this window or the
tray menu. Captions remain above ordinary windows and most borderless fullscreen
players; exclusive fullscreen may require switching to windowed/borderless.
Long phrases are split into pages, so the text is not silently clipped.

DATA AND SETUP
Only dependency/model setup uses the internet. Start loads the model
and tokenizer from local files with local_files_only=True. No audio is uploaded
or recorded. Caption history is in memory, limited to the latest 500 captions,
and can be saved as text with Export. Clear history removes that in-memory text.
Preferences, downloaded models and technical error logs are stored in:
%LOCALAPPDATA%\\OfflineLiveCaptions
Logs do not intentionally contain audio or recognized text. Redownloading a model
repairs an incomplete download. Windows 10/11 on an x64 PC is required. Setup uses
its own compatible 64-bit Python; your original Python version is not used to run
the caption engine. Setup errors include their actual details and are saved to
setup.log. Use Copy setup log if you need help. Retry / repair resumes setup.
You may rename the file to .pyw after setup to launch without a console window.
If a library reports "DLL load failed", install Microsoft's Visual C++ 2015–2022
Redistributable (x64), then reopen the app.
"""


def run_gui(root: Path, smoke_test=False, test_hook=None):
    from PySide6.QtCore import Qt, QObject, Signal, QTimer, QPoint, QRectF, QLockFile
    from PySide6.QtGui import QColor, QFont, QFontMetrics, QIcon, QPainter, QPixmap, QAction
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
        QFormLayout, QComboBox, QSpinBox, QDoubleSpinBox, QSlider, QCheckBox, QTabWidget,
        QPlainTextEdit, QProgressBar, QMessageBox, QFileDialog, QSystemTrayIcon, QMenu,
        QGroupBox, QScrollArea, QSizePolicy,
    )

    class Bridge(QObject):
        event = Signal(str, object)

    class Overlay(QWidget):
        def __init__(self, settings):
            super().__init__()
            self.settings = settings
            self.text = ""
            self.pages = collections.deque()
            self.active = False
            self.drag_offset = None
            self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint |
                                Qt.WindowType.Tool | Qt.WindowType.WindowDoesNotAcceptFocus)
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
            self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
            self.label = QLabel(self)
            self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.label.setTextFormat(Qt.TextFormat.PlainText)
            self.label.setWordWrap(True)
            self.label.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
            self.timer = QTimer(self)
            self.timer.setSingleShot(True)
            self.timer.timeout.connect(self.next_page)
            self.apply_settings(reposition=True)

        def screen(self):
            return next((s for s in app.screens() if s.name() == self.settings.screen_name), app.primaryScreen())

        def apply_settings(self, reposition=False):
            screen = self.screen()
            area = screen.availableGeometry()
            font = QFont("Segoe UI", self.settings.font_size)
            font.setWeight(QFont.Weight.DemiBold)
            self.label.setFont(font)
            metrics = QFontMetrics(font)
            width = max(280, int(area.width() * self.settings.width_percent / 100))
            height = metrics.lineSpacing() * 3 + 32
            self.setFixedSize(min(width, area.width()), min(height, area.height()))
            self.label.setGeometry(18, 12, self.width() - 36, self.height() - 24)
            self.label.setStyleSheet(f"color: white; background: transparent; font-size: {self.settings.font_size}pt; font-weight: 600;")
            self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, self.settings.locked)
            self.setWindowFlag(Qt.WindowType.WindowTransparentForInput, self.settings.locked)
            if reposition:
                self.move(area.x() + (area.width() - self.width()) // 2,
                          area.y() + area.height() - self.height() - int(area.height() * self.settings.bottom_percent / 100))
            else:
                self.move(max(area.left(), min(self.x(), area.right() - self.width() + 1)),
                          max(area.top(), min(self.y(), area.bottom() - self.height() + 1)))
            if self.active and self.text:
                self.show()
            self.update()

        def split_pages(self, text):
            # Use real font metrics, including long words, and never truncate a phrase.
            metrics = QFontMetrics(self.label.font())
            max_width = max(40, self.label.width() - 8)
            lines, line = [], ""
            for word in text.split():
                candidate = (line + " " + word).strip()
                if line and metrics.horizontalAdvance(candidate) > max_width:
                    lines.append(line)
                    line = ""
                if metrics.horizontalAdvance(word) > max_width:
                    for char in word:
                        if line and metrics.horizontalAdvance(line + char) > max_width:
                            lines.append(line)
                            line = ""
                        line += char
                else:
                    line = (line + " " + word).strip()
            if line:
                lines.append(line)
            return ["\n".join(lines[i:i + 3]) for i in range(0, len(lines), 3)]

        def set_caption(self, text):
            if not self.active:
                return
            new_pages = self.split_pages(text)
            # One phrase's pages only: an overlay backlog must not grow behind audio.
            self.pages.clear()
            self.pages.extend(new_pages)
            self.next_page()

        def next_page(self):
            if self.pages:
                self.text = self.pages.popleft()
                self.label.setText(self.text)
                self.show()
                self.raise_()
                seconds = min(self.settings.display_seconds, max(2.0, len(self.text.split()) / 3.2))
                if not self.pages:
                    seconds = self.settings.display_seconds
                self.timer.start(int(seconds * 1000))
                self.update()
            else:
                self.text = ""
                self.label.clear()
                self.hide()

        def clear(self):
            self.timer.stop()
            self.pages.clear()
            self.text = ""
            self.label.clear()
            self.hide()

        def paintEvent(self, event):
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(8, 12, 20, int(self.settings.opacity * 2.55)))
            painter.drawRoundedRect(QRectF(self.rect()), 12, 12)
            if not self.settings.locked:
                painter.setPen(QColor(130, 145, 165, 150))
                painter.drawLine(self.width() // 2 - 18, 6, self.width() // 2 + 18, 6)

        def mousePressEvent(self, event):
            if event.button() == Qt.MouseButton.LeftButton and not self.settings.locked:
                self.drag_offset = event.globalPosition().toPoint() - self.pos()

        def mouseMoveEvent(self, event):
            if self.drag_offset is not None:
                self.move(event.globalPosition().toPoint() - self.drag_offset)

        def mouseReleaseEvent(self, event):
            if self.drag_offset is not None:
                self.settings.overlay_x, self.settings.overlay_y = self.x(), self.y()
                self.drag_offset = None

    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.settings_path = root / "settings.json"
            self.settings = Settings.load(self.settings_path)
            self.bridge = Bridge(self)
            self.bridge.event.connect(self.handle_event)
            self.engine = None
            self.downloading = False
            self.download_process = None
            self.download_cancel = threading.Event()
            self.scanning = False
            self.want_close = False
            self.history = collections.deque(maxlen=500)
            self.notice = ""
            self.setWindowTitle(f"{APP_NAME} · Offline English translation")
            self.resize(860, 780)
            self.setMinimumSize(650, 600)
            self.setWindowIcon(icon)
            self.overlay = Overlay(self.settings)
            if self.settings.overlay_x != -1 or self.settings.overlay_y != -1:
                candidate = QPoint(self.settings.overlay_x, self.settings.overlay_y)
                # Restore a dragged position only if its center still fits a display.
                if any(s.availableGeometry().contains(candidate + QPoint(self.overlay.width() // 2, self.overlay.height() // 2))
                       for s in app.screens()):
                    self.overlay.move(candidate)
            main = QWidget()
            self.setCentralWidget(main)
            layout = QVBoxLayout(main)
            layout.setContentsMargins(26, 22, 26, 20)
            title = QLabel("Live Captions")
            title.setObjectName("title")
            layout.addWidget(title)
            subtitle = QLabel("Your browser audio. English captions. Entirely on your computer.")
            subtitle.setObjectName("muted")
            layout.addWidget(subtitle)
            self.tabs = QTabWidget()
            layout.addWidget(self.tabs, 1)
            self.make_translation_tab()
            self.make_caption_tab()
            self.make_history_tab()
            help_text = QPlainTextEdit()
            help_text.setPlainText(HELP)
            help_text.setReadOnly(True)
            self.tabs.addTab(help_text, "Help")
            self.status = QLabel("Ready — download a model, then start")
            self.status.setWordWrap(True)
            layout.addWidget(self.status)
            self.warning = QLabel("")
            self.warning.setWordWrap(True)
            self.warning.setStyleSheet("color: #fbbf24;")
            layout.addWidget(self.warning)
            row = QHBoxLayout()
            self.start_button = QPushButton("Start captions")
            self.start_button.setObjectName("primary")
            self.start_button.clicked.connect(self.start)
            self.stop_button = QPushButton("Stop")
            self.stop_button.setEnabled(False)
            self.stop_button.clicked.connect(self.stop)
            self.min_button = QPushButton("Minimize to tray")
            self.min_button.clicked.connect(self.minimize_tray)
            row.addWidget(self.start_button, 2)
            row.addWidget(self.stop_button)
            row.addWidget(self.min_button)
            layout.addLayout(row)
            self.tray = QSystemTrayIcon(icon, self)
            self.tray.setToolTip("Live Captions — offline English translation")
            menu = QMenu()
            action = menu.addAction("Show settings")
            action.triggered.connect(self.restore)
            self.tray_lock = menu.addAction("Lock captions (click-through)")
            self.tray_lock.setCheckable(True)
            self.tray_lock.setChecked(self.settings.locked)
            self.tray_lock.triggered.connect(self.lock_box.setChecked)
            menu.addAction("Reset caption position").triggered.connect(self.reset_overlay)
            menu.addAction("Stop captions").triggered.connect(self.stop)
            menu.addSeparator()
            menu.addAction("Exit").triggered.connect(self.close)
            self.tray.setContextMenu(menu)
            self.tray.activated.connect(lambda reason: self.restore() if reason == QSystemTrayIcon.ActivationReason.DoubleClick else None)
            if QSystemTrayIcon.isSystemTrayAvailable():
                self.tray.show()
            else:
                self.min_button.setText("Minimize")
            self.previous_time = time.monotonic()
            self.previous_cpu = time.process_time()
            self.resource_timer = QTimer(self)
            self.resource_timer.timeout.connect(self.update_resources)
            self.resource_timer.start(1500)
            self.update_model_status()
            self.update_controls()
            if model_ready(root, self.mode.currentData()):
                self.status.setText("Ready — select your spoken language and start captions")
            if not smoke_test:
                self.refresh_devices()
            app.screenAdded.connect(lambda _: self.refresh_screens())
            app.screenRemoved.connect(lambda _: self.refresh_screens())

        def form_page(self, label):
            page = QWidget()
            layout = QVBoxLayout(page)
            layout.setContentsMargins(18, 18, 18, 18)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setFrameShape(QScrollArea.Shape.NoFrame)
            scroll.setWidget(page)
            self.tabs.addTab(scroll, label)
            return layout

        def make_translation_tab(self):
            layout = self.form_page("Translation")
            group = QGroupBox("Speech and translation")
            form = QFormLayout(group)
            self.mode = QComboBox()
            for label, value in MODEL_LABELS.items():
                self.mode.addItem(label, value)
            self.mode.setCurrentIndex(self.mode.findData(self.settings.model))
            self.language = QComboBox()
            self.language.addItems(LANGUAGES)
            self.language.setCurrentText(self.settings.language)
            form.addRow("Mode", self.mode)
            form.addRow("Spoken language", self.language)
            self.model_status = QLabel()
            self.model_status.setWordWrap(True)
            self.download_button = QPushButton("Download selected model")
            self.download_button.clicked.connect(self.download)
            form.addRow("Local model", self.model_status)
            form.addRow("", self.download_button)
            self.download_progress = QProgressBar()
            self.download_progress.setRange(0, 0)
            self.download_progress.hide()
            form.addRow("", self.download_progress)
            self.cancel_download_button = QPushButton("Cancel download")
            self.cancel_download_button.clicked.connect(self.cancel_download)
            self.cancel_download_button.hide()
            form.addRow("", self.cancel_download_button)
            layout.addWidget(group)
            group = QGroupBox("Windows playback audio")
            form = QFormLayout(group)
            self.devices = QComboBox()
            self.devices.addItem("Windows default at last refresh", None)
            self.refresh_button = QPushButton("Refresh devices")
            self.refresh_button.clicked.connect(self.refresh_devices)
            form.addRow("Output device", self.devices)
            form.addRow("", self.refresh_button)
            self.meter = QProgressBar()
            self.meter.setRange(0, 80)
            self.meter.setValue(0)
            self.meter.setFormat("Audio level")
            form.addRow("Playback meter", self.meter)
            layout.addWidget(group)
            group = QGroupBox("Performance · stop captions before changing")
            form = QFormLayout(group)
            self.threads = QSpinBox()
            self.threads.setRange(1, min(4, os.cpu_count() or 2))
            self.threads.setValue(self.settings.threads)
            self.threads.setToolTip("Two threads is a useful starting point for an i5-6500 with a browser open.")
            self.chunk = QDoubleSpinBox()
            self.chunk.setRange(2, 10)
            self.chunk.setSingleStep(0.5)
            self.chunk.setSuffix(" seconds")
            self.chunk.setValue(self.settings.chunk_seconds)
            self.chunk.setToolTip("Maximum phrase length. Shorter is quicker; longer preserves more context.")
            self.silence = QDoubleSpinBox()
            self.silence.setRange(0.3, 1.5)
            self.silence.setSingleStep(0.1)
            self.silence.setSuffix(" seconds")
            self.silence.setValue(self.settings.silence_seconds)
            self.gate = QSpinBox()
            self.gate.setRange(-70, -20)
            self.gate.setSuffix(" dB")
            self.gate.setValue(self.settings.gate_db)
            self.gate.setToolTip("Lower (more negative) captures quieter dialogue, but admits more noise.")
            form.addRow("CPU threads", self.threads)
            form.addRow("Maximum phrase", self.chunk)
            form.addRow("End phrase after silence", self.silence)
            form.addRow("Speech gate", self.gate)
            layout.addWidget(group)
            self.metrics = QLabel("Processing —  |  Delay —  |  Skipped audio chunks: 0")
            self.metrics.setWordWrap(True)
            layout.addWidget(self.metrics)
            self.resources = QLabel("CPU —  |  App memory —")
            self.resources.setObjectName("muted")
            layout.addWidget(self.resources)
            layout.addStretch()
            self.mode.currentIndexChanged.connect(self.mode_changed)
            self.language.currentTextChanged.connect(self.save_preferences)
            for control in (self.threads, self.chunk, self.silence, self.gate):
                control.valueChanged.connect(self.save_preferences)

        def make_caption_tab(self):
            layout = self.form_page("Captions")
            group = QGroupBox("Appearance · changes apply immediately")
            form = QFormLayout(group)
            self.screen_combo = QComboBox()
            self.refresh_screens(initial=True)
            form.addRow("Display", self.screen_combo)
            self.caption_controls = {}
            specs = [("Font size", "font_size", 16, 56), ("Background opacity %", "opacity", 15, 100),
                     ("Width % of display", "width_percent", 35, 95),
                     ("Distance from bottom %", "bottom_percent", 0, 70),
                     ("Clear caption after seconds", "display_seconds", 3, 30)]
            for label, key, low, high in specs:
                control = QSpinBox()
                control.setRange(low, high)
                control.setValue(getattr(self.settings, key))
                control.valueChanged.connect(lambda _value, k=key: self.appearance_changed(k))
                self.caption_controls[key] = control
                form.addRow(label, control)
            self.lock_box = QCheckBox("Lock overlay and pass mouse clicks through it")
            self.lock_box.setChecked(self.settings.locked)
            self.lock_box.toggled.connect(self.lock_changed)
            form.addRow("", self.lock_box)
            layout.addWidget(group)
            row = QHBoxLayout()
            preview = QPushButton("Preview captions")
            preview.clicked.connect(self.preview)
            reset = QPushButton("Reset caption position")
            reset.clicked.connect(self.reset_overlay)
            row.addWidget(preview)
            row.addWidget(reset)
            layout.addLayout(row)
            note = QLabel("Drag the caption box while unlocked. Lock it when watching a video.\n"
                          "Use windowed or borderless fullscreen if exclusive fullscreen hides the overlay.")
            note.setWordWrap(True)
            note.setObjectName("muted")
            layout.addWidget(note)
            layout.addStretch()
            self.screen_combo.currentIndexChanged.connect(lambda _: self.appearance_changed("screen"))

        def make_history_tab(self):
            page = QWidget()
            layout = QVBoxLayout(page)
            note = QLabel("Last 500 captions · held in memory until you export them")
            note.setObjectName("muted")
            layout.addWidget(note)
            self.history_view = QPlainTextEdit()
            self.history_view.setReadOnly(True)
            self.history_view.setMaximumBlockCount(1000)
            layout.addWidget(self.history_view)
            row = QHBoxLayout()
            export = QPushButton("Export text…")
            export.clicked.connect(self.export_history)
            clear = QPushButton("Clear history")
            clear.clicked.connect(self.clear_history)
            row.addWidget(export)
            row.addWidget(clear)
            layout.addLayout(row)
            self.tabs.addTab(page, "History")

        def refresh_screens(self, initial=False):
            if not hasattr(self, "screen_combo"):
                return
            self.screen_combo.blockSignals(True)
            self.screen_combo.clear()
            for index, screen in enumerate(app.screens()):
                self.screen_combo.addItem(f"Display {index + 1} · {screen.name()}", screen.name())
            index = self.screen_combo.findData(self.settings.screen_name)
            self.screen_combo.setCurrentIndex(max(0, index))
            self.screen_combo.blockSignals(False)
            if not initial:
                self.appearance_changed("screen")

        def selected_settings(self):
            self.settings.model = self.mode.currentData()
            self.settings.language = self.language.currentText()
            self.settings.threads = self.threads.value()
            self.settings.chunk_seconds = self.chunk.value()
            self.settings.silence_seconds = self.silence.value()
            self.settings.gate_db = self.gate.value()
            data = self.devices.currentData()
            self.settings.device_name = data["name"] if data else ""
            return self.settings

        def save_preferences(self, *_):
            if not hasattr(self, "devices"):
                return
            self.selected_settings()
            try:
                self.settings.save(self.settings_path)
            except OSError:
                LOG.exception("Saving preferences")
                if hasattr(self, "warning"):
                    self.warning.setText("Could not save preferences. Check that the app data folder is writable.")

        def mode_changed(self, *_):
            self.save_preferences()
            self.update_model_status()
            self.update_controls()

        def update_model_status(self):
            ready = model_ready(root, self.mode.currentData())
            self.model_status.setText("Downloaded · ready for offline use" if ready else "Not downloaded yet")
            self.download_button.setText("Repair / redownload model" if ready else "Download selected model")

        def update_controls(self):
            busy = self.engine is not None or self.downloading
            for control in (self.mode, self.language, self.threads, self.chunk, self.silence, self.gate):
                control.setEnabled(not busy)
            self.devices.setEnabled(not busy and not self.scanning)
            self.refresh_button.setEnabled(not busy and not self.scanning)
            self.download_button.setEnabled(not busy)
            self.start_button.setEnabled(not busy and not self.scanning and model_ready(root, self.mode.currentData()))
            self.stop_button.setEnabled(self.engine is not None and not self.engine.stop_event.is_set())

        def refresh_devices(self):
            if self.engine or self.scanning or self.downloading:
                return
            self.scanning = True
            self.update_controls()
            desired = self.settings.device_name
            data = self.devices.currentData()
            if data:
                desired = data["name"]
            def worker():
                try:
                    self.bridge.event.emit("devices", (audio_devices(), desired))
                except Exception as exc:
                    LOG.exception("Scanning audio devices")
                    self.bridge.event.emit("device_error", str(exc))
            threading.Thread(target=worker, name="device-scan", daemon=True).start()

        def download(self):
            if self.engine or self.downloading:
                return
            self.downloading = True
            self.download_cancel.clear()
            self.warning.clear()
            self.download_progress.show()
            self.cancel_download_button.show()
            self.cancel_download_button.setEnabled(True)
            self.update_controls()
            model = self.mode.currentData()
            self.status.setText(f"Preparing {model} download…")
            # Separate process: downloads and model validation can be cancelled
            # without leaving a native library running inside the GUI process.
            def worker():
                try:
                    with (root / "download.log").open("w", encoding="utf-8") as errors:
                        process = subprocess.Popen(
                            [console_python(), str(Path(__file__).resolve()), "--download-model", model,
                             "--data-dir", str(root)],
                            stdout=subprocess.PIPE, stderr=errors, text=True, encoding="utf-8", errors="replace",
                            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                        )
                        self.download_process = process
                        if self.download_cancel.is_set():
                            process.terminate()
                        terminal_event = None
                        for line in process.stdout:
                            try:
                                kind, value = json.loads(line)
                                if kind in ("download_done", "download_error"):
                                    terminal_event = (kind, value)
                                elif kind == "status":
                                    self.bridge.event.emit(kind, value)
                            except (ValueError, TypeError):
                                LOG.debug("Non-protocol output from download subprocess")
                        code = process.wait()
                        self.download_process = None
                        if self.download_cancel.is_set():
                            self.bridge.event.emit("download_cancelled", None)
                        elif terminal_event:
                            self.bridge.event.emit(*terminal_event)
                        else:
                            self.bridge.event.emit("download_error", f"Download process exited ({code}). See {root / 'download.log'}.")
                except Exception as exc:
                    self.download_process = None
                    LOG.exception("Starting model download")
                    self.bridge.event.emit("download_error", str(exc))
            threading.Thread(target=worker, name="model-download", daemon=True).start()

        def cancel_download(self):
            if not self.downloading:
                return
            self.download_cancel.set()
            self.cancel_download_button.setEnabled(False)
            self.status.setText("Cancelling model download…")
            process = self.download_process
            if process is not None:
                try:
                    process.terminate()
                except OSError:
                    pass

        def start(self):
            if self.engine or self.downloading or self.scanning:
                return
            settings = dataclasses.replace(self.selected_settings())
            if not model_ready(root, settings.model):
                QMessageBox.information(self, "Model needed", "Download the selected model first.")
                return
            data = self.devices.currentData()
            if data is None:
                # The default is resolved on refresh. A device removal/reorder
                # is also checked again by the capture thread before opening it.
                data = getattr(self, "default_device", None)
            if not data:
                QMessageBox.warning(self, "No playback device", "Connect speakers/headphones, then Refresh devices.")
                return
            settings.device_name = data["name"]
            self.save_preferences()
            self.overlay.clear()
            self.overlay.active = True
            self.warning.clear()
            self.metrics.setText("Processing —  |  Delay —  |  Skipped audio chunks: 0")
            self.skip_count = 0
            self.engine = CaptionEngine(root, settings, data["index"], self.bridge.event.emit)
            self.engine.start()
            self.update_controls()

        def stop(self):
            if self.engine:
                self.engine.stop()
                self.status.setText("Stopping — waiting for the current recognition to finish…")
            self.overlay.active = False
            self.overlay.clear()
            self.update_controls()

        def preview(self):
            self.overlay.active = True
            self.overlay.set_caption("English captions appear here. Drag this box to place it over your video.")

        def reset_overlay(self):
            self.settings.overlay_x = self.settings.overlay_y = -1
            self.overlay.apply_settings(reposition=True)
            self.save_preferences()

        def appearance_changed(self, key):
            for name, control in self.caption_controls.items():
                setattr(self.settings, name, control.value())
            self.settings.screen_name = self.screen_combo.currentData() or ""
            reposition = key in ("width_percent", "bottom_percent", "screen")
            if reposition:
                self.settings.overlay_x = self.settings.overlay_y = -1
            self.overlay.apply_settings(reposition=reposition)
            if self.overlay.text:
                self.overlay.set_caption(self.overlay.text.replace("\n", " "))
            self.save_preferences()

        def lock_changed(self, locked):
            self.settings.locked = locked
            self.overlay.apply_settings()
            if hasattr(self, "tray_lock"):
                self.tray_lock.setChecked(locked)
            self.save_preferences()

        def clear_history(self):
            self.history.clear()
            self.history_view.clear()

        def export_history(self):
            if not self.history:
                QMessageBox.information(self, "History empty", "No captions to export yet.")
                return
            path, _ = QFileDialog.getSaveFileName(self, "Export English captions", "captions.txt", "Text files (*.txt)")
            if path:
                try:
                    Path(path).write_text("\n".join(self.history) + "\n", encoding="utf-8")
                except OSError as exc:
                    QMessageBox.warning(self, "Export failed", str(exc))

        def handle_event(self, kind, value):
            if kind == "status":
                if not self.want_close and not (self.engine and self.engine.stop_event.is_set()):
                    self.status.setText(value)
            elif kind == "warning":
                self.warning.setText(value)
            elif kind == "level":
                self.meter.setValue(round(value + 80))
                self.meter.setFormat(f"{value:.0f} dB")
            elif kind == "caption":
                if self.engine and not self.engine.stop_event.is_set():
                    self.overlay.set_caption(value)
                    line = f"[{dt.datetime.now().strftime('%H:%M:%S')}] {value}"
                    self.history.append(line)
                    self.history_view.appendPlainText(line)
            elif kind == "dropped":
                self.skip_count = value
                if value:
                    self.warning.setText(f"Skipped {value} audio chunk(s) to stay near live playback. Try Fast mode.")
            elif kind == "metrics":
                self.metrics.setText(f"Processing {value['processing']:.1f}s / {value['audio']:.1f}s audio  |  "
                                     f"Delay after phrase {value['delay']:.1f}s  |  "
                                     f"Detected: {value['language']}  |  Skipped: {getattr(self, 'skip_count', 0)}")
                if value["processing"] > value["audio"] * 1.1:
                    self.warning.setText("Recognition is slower than playback. Fast mode or longer phrases may help.")
            elif kind == "error":
                self.notice = value
                self.warning.setText(value)
                if not self.want_close:
                    QMessageBox.warning(self, "Caption error", value + "\n\nSee Help for troubleshooting. Technical details: " + str(root / "app.log"))
            elif kind == "finished":
                self.engine = None
                self.overlay.active = False
                self.overlay.clear()
                self.meter.setValue(0)
                self.meter.setFormat("Audio level")
                self.status.setText("Stopped — ready to start again")
                self.update_controls()
                if self.want_close:
                    QTimer.singleShot(0, self.close)
            elif kind in ("download_done", "download_error", "download_cancelled"):
                self.downloading = False
                self.download_progress.hide()
                self.cancel_download_button.hide()
                self.update_model_status()
                self.update_controls()
                if kind == "download_done":
                    self.status.setText(f"{value.capitalize()} model ready — you can disconnect the internet and start")
                elif kind == "download_cancelled":
                    self.status.setText("Download cancelled — click Download to resume later")
                else:
                    self.status.setText("Download failed — click Download to resume")
                    QMessageBox.warning(self, "Model download failed", value)
                if self.want_close:
                    QTimer.singleShot(0, self.close)
            elif kind == "devices":
                self.scanning = False
                devices, desired = value
                self.devices.blockSignals(True)
                self.devices.clear()
                self.devices.addItem("Windows default at last refresh", None)
                self.default_device = next((d for d in devices if d["default"]), None)
                for device in devices:
                    self.devices.addItem(device["name"], device)
                chosen = next((i + 1 for i, d in enumerate(devices) if d["name"] == desired), 0)
                self.devices.setCurrentIndex(chosen)
                self.devices.blockSignals(False)
                if not devices:
                    self.warning.setText("No WASAPI loopback devices found. Connect a playback device and Refresh.")
                self.update_controls()
            elif kind == "device_error":
                self.scanning = False
                self.warning.setText("Could not list playback devices: " + value)
                self.update_controls()

        def update_resources(self):
            now, cpu = time.monotonic(), time.process_time()
            percent = 100 * (cpu - self.previous_cpu) / max(0.01, now - self.previous_time) / (os.cpu_count() or 1)
            self.previous_time, self.previous_cpu = now, cpu
            memory = "—"
            if sys.platform == "win32":
                class Counters(ctypes.Structure):
                    _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong)] + [
                        (name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                        "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage", "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]
                try:
                    counts = Counters()
                    counts.cb = ctypes.sizeof(counts)
                    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
                    kernel.GetCurrentProcess.restype = ctypes.c_void_p
                    psapi = ctypes.WinDLL("psapi", use_last_error=True)
                    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(Counters), ctypes.c_ulong]
                    if psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counts), counts.cb):
                        memory = f"{counts.WorkingSetSize / 1024 ** 2:.0f} MB"
                except (OSError, AttributeError):
                    pass
            self.resources.setText(f"App CPU {min(100, percent):.0f}% of total capacity  |  App memory {memory}")

        def minimize_tray(self):
            self.save_preferences()
            if QSystemTrayIcon.isSystemTrayAvailable():
                self.hide()
                self.tray.showMessage("Live Captions", "Still running. Double-click the tray icon to restore settings.",
                                      QSystemTrayIcon.MessageIcon.Information, 2500)
            else:
                self.showMinimized()

        def restore(self):
            self.showNormal()
            self.raise_()
            self.activateWindow()

        def closeEvent(self, event):
            # Keep the Qt signal receiver alive until native recognition finishes.
            # Python threads cannot safely interrupt an active CTranslate2 call.
            if self.downloading:
                self.want_close = True
                self.cancel_download()
                event.ignore()
                return
            if self.engine:
                self.want_close = True
                self.stop()
                event.ignore()
                return
            self.save_preferences()
            self.overlay.close()
            self.tray.hide()
            event.accept()
            app.quit()

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName("OfflineLiveCaptions")
    app.setQuitOnLastWindowClosed(False)
    app.setStyle("Fusion")
    app_lock = QLockFile(str(root / "application.lock"))
    if not smoke_test and not app_lock.tryLock(0):
        QMessageBox.information(None, APP_NAME, "Live Captions is already running, or its data folder is locked.\n"
                                "Restore it from the CC icon in the Windows system tray.")
        return 0
    app.setStyleSheet("""
        QWidget { background: #111827; color: #e5e7eb; font-family: 'Segoe UI'; font-size: 13px; }
        QLabel#title { font-size: 29px; font-weight: 700; }
        QLabel#muted { color: #9ca3af; }
        QTabWidget::pane { border: 1px solid #334155; border-radius: 9px; }
        QTabBar::tab { padding: 11px 18px; background: #172033; color: #aab8ca; }
        QTabBar::tab:selected { background: #26364d; color: #ffffff; }
        QGroupBox { border: 1px solid #334155; border-radius: 8px; margin-top: 15px; padding: 16px 10px 10px; }
        QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 5px; color: #b9cbe5; }
        QPushButton { background: #26364d; border: 1px solid #41536d; border-radius: 6px; padding: 9px 14px; }
        QPushButton:hover { background: #344a68; }
        QPushButton:disabled { background: #182234; color: #64748b; border-color: #26364d; }
        QPushButton#primary { background: #2563eb; border-color: #3b82f6; color: white; font-weight: 600; }
        QPushButton#primary:disabled { background: #1e3456; border-color: #26364d; color: #8395ae; }
        QComboBox, QSpinBox, QDoubleSpinBox { background: #0c1424; border: 1px solid #41536d; border-radius: 5px; padding: 7px; min-height: 19px; }
        QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled { color: #64748b; }
        QPlainTextEdit { background: #0c1424; border: 1px solid #334155; border-radius: 6px; padding: 8px; }
        QProgressBar { border: 1px solid #334155; border-radius: 5px; text-align: center; background: #0c1424; min-height: 18px; }
        QProgressBar::chunk { background: #3b82f6; border-radius: 4px; }
        QCheckBox { spacing: 10px; padding: 7px 0; }
        QMenu { background: #172033; border: 1px solid #41536d; }
        QMenu::item { padding: 7px 20px; }
        QMenu::item:selected { background: #344a68; }
    """)
    pixmap = QPixmap(64, 64)
    pixmap.fill(QColor("#2563eb"))
    painter = QPainter(pixmap)
    painter.setPen(QColor("white"))
    painter.setFont(QFont("Segoe UI", 23, QFont.Weight.Bold))
    painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "CC")
    painter.end()
    icon = QIcon(pixmap)
    window = MainWindow()
    # Workers only interact with QObject via queued signals; no widgets off-thread.
    def exception_hook(kind, value, tb):
        LOG.error("Unhandled GUI error", exc_info=(kind, value, tb))
        QMessageBox.critical(window, "Unexpected error", f"{value}\n\nDetails were saved to {root / 'app.log'}")
    old_exception_hook = sys.excepthook
    sys.excepthook = exception_hook
    window.show()
    if test_hook:
        test_hook(window, app)
    if smoke_test:
        window.preview()
        app.processEvents()
        screenshot = os.environ.get("LIVE_CAPTIONS_SCREENSHOT")
        if screenshot:
            window.grab().save(screenshot)
        QTimer.singleShot(250, window.close)
    try:
        return app.exec()
    finally:
        sys.excepthook = old_exception_hook


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--managed-run", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--repair-setup", action="store_true", help="Repair the private Python environment and libraries")
    parser.add_argument("--ui-smoke-test", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--download-model", choices=("base", "small"), help=argparse.SUPPRESS)
    parser.add_argument("--data-dir", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    smoke = args.ui_smoke_test
    if sys.platform != "win32" and not smoke and not args.download_model:
        print("Live Captions requires Windows 10/11 for WASAPI system-audio capture.")
        return 1
    if (args.managed_run or args.download_model or smoke) and struct.calcsize("P") != 8:
        native_error("64-bit Python needed", "The app runtime must be 64-bit. Reopen the file normally to run automatic setup.")
        return 1
    root = args.data_dir or app_dir()
    try:
        init_logging(root)
        if args.download_model:
            result = {"success": False}
            def emit(kind, value):
                if kind == "download_done":
                    result["success"] = True
                print(json.dumps([kind, value]), flush=True)
            download_model(root, args.download_model, emit)
            return 0 if result["success"] else 1
        if not smoke and not args.managed_run:
            try:
                setup_lock = acquire_setup_lock(root)
            except OSError:
                native_error("Setup is already running", "Another Live Captions setup is using this installation.\n"
                             "Use the setup window that is already open, or wait for it to close.")
                return 1
            try:
                if bootstrap_dependencies(root, repair=args.repair_setup):
                    launch_managed(root)
            finally:
                setup_lock.close()
            return 0
        return run_gui(root, smoke)
    except Exception:
        LOG.exception("Startup failed")
        detail = traceback.format_exc()
        native_error("Live Captions could not start", detail + "\n\nRun the file with --repair-setup to repair its private environment.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
