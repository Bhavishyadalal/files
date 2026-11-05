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
VERSION = "1.2.0"
DEPENDENCIES = [
    ("PySide6", "PySide6-Essentials==6.8.3"),
    ("faster_whisper", "faster-whisper==1.2.1"),
    ("pyaudiowpatch", "PyAudioWPatch==0.2.12.7"),
    ("numpy", "numpy==1.26.4"),
    ("scipy", "scipy==1.13.1"),
    ("huggingface_hub", "huggingface-hub>=0.24,<1"),
    ("pkg_resources", "setuptools==80.9.0"),
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

    def stage(self, number, title):
        self.log(f"[STEP {number}/6] {title}")

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
        command_tail = collections.deque(maxlen=16)
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
                command_tail.extend(str(line).splitlines())
                self.log(line)
            code = process.wait()
            self.check_cancel()
            if code and not allow_failure:
                details = "\n".join(command_tail)[-4000:]
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
            self.stage(6, "Ready to launch")
            self.log("Setup is already complete. Starting Live Captions…")
            return
        self.check_cancel()
        (self.root / "setup-ready.json").unlink(missing_ok=True)
        self.stage(1, "Preparing private Python")
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
        self.stage(2, "Checking Windows components")
        if not self.runtime_available():
            self.install_vc_runtime()
            if not self.runtime_available():
                raise RuntimeError("Microsoft's C++ runtime is still unavailable. Restart Windows, then open this file again.")
        self.stage(3, "Installing application libraries")
        pip_exists = self.command([runtime_python(self.root), "-m", "pip", "--version"],
                                  "Checking package installer", allow_failure=True) == 0
        if not pip_exists:
            script = self.fetch("https://bootstrap.pypa.io/get-pip.py", self.root / "setup-downloads" / "get-pip.py")
            self.command([runtime_python(self.root), script, "pip>=24,<27", "setuptools==80.9.0", "wheel>=0.43,<1",
                          "--only-binary=:all:", "--disable-pip-version-check", "--no-warn-script-location"],
                         "Installing pip, setuptools and wheel")
        pip = [runtime_python(self.root), "-m", "pip", "--isolated", "--disable-pip-version-check", "--no-input",
               "install", "--upgrade", "--only-binary=:all:", "--retries", "3", "--timeout", "30",
               "--no-warn-script-location"]
        self.command(pip + ["pip>=24,<27", "setuptools==80.9.0", "wheel>=0.43,<1"], "Updating package installer")
        packages = [spec for _, spec in DEPENDENCIES]
        if self.repair or completed:
            packages = ["--force-reinstall"] + packages
        self.command(pip + packages, "Installing all speech, audio and interface libraries")
        self.stage(4, "Checking the caption engine")
        self.validate()
        self.stage(5, "Preparing both speech models")
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
        self.stage(6, "Ready to launch")
        self.log("All components and both speech models are ready. Starting Live Captions…")


def native_error(title, detail):
    if sys.platform == "win32":
        ctypes.windll.user32.MessageBoxW(None, str(detail), str(title), 0x10)
    else:
        print(f"{title}: {detail}", file=sys.stderr)


def setup_error_summary(error):
    lines = [line.strip() for line in str(error).splitlines() if line.strip()]
    for line in reversed(lines):
        if any(token in line for token in ("ModuleNotFoundError:", "ImportError:", "OSError:", "RuntimeError:", "ERROR:")):
            return line[-900:]
    return "\n".join(lines[-4:])[-1200:] or "Setup could not finish. Check the setup log for details."


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
    setup_font = "Segoe UI" if sys.platform == "win32" else "Arial"
    window.title(f"Live Captions {VERSION} — make yourself at home")
    window.geometry("960x660")
    window.minsize(800,560)
    window.configure(bg="#0b1220")
    # Setup uses the standard library until Qt has been installed.
    from tkinter import ttk
    style = ttk.Style(window)
    style.theme_use("clam")
    style.configure("Studio.Horizontal.TProgressbar", troughcolor="#233248", background="#91d9c9",
                    bordercolor="#233248", lightcolor="#91d9c9", darkcolor="#91d9c9", thickness=6)
    body = tk.Frame(window,bg="#0b1220"); body.pack(fill="both",expand=True)
    rail = tk.Frame(body,bg="#080f1b",width=215); rail.pack(side="left",fill="y"); rail.pack_propagate(False)
    tk.Label(rail,text="LC",font=(setup_font,22,"bold"),bg="#365a71",fg="#f0fffb",padx=14,pady=9).pack(anchor="w",padx=24,pady=(27,13))
    tk.Label(rail,text="Live Captions",font=(setup_font,17,"bold"),bg="#080f1b",fg="#ecf3ff").pack(anchor="w",padx=24)
    tk.Label(rail,text="A little closer to the world.",font=(setup_font,9),bg="#080f1b",fg="#859bb6").pack(anchor="w",padx=24,pady=(4,32))
    tk.Label(rail,text="YOUR SETUP",font=(setup_font,9,"bold"),bg="#080f1b",fg="#7088a8").pack(anchor="w",padx=24,pady=(0,12))
    stage_labels=[]
    for i,title in enumerate(("Private Python","Windows components","App libraries","Engine check","Speech models","Ready to listen"),1):
        label=tk.Label(rail,text=f"{i:02d}    {title}",font=(setup_font,10),bg="#080f1b",fg="#6f86a5",padx=8,pady=11,anchor="w")
        label.pack(fill="x",padx=16,pady=3);stage_labels.append(label)
    tk.Label(rail,text="YOUR AUDIO STAYS YOURS\nOffline after setup.\nNo account. No cloud API.",font=(setup_font,9),justify="left",
             bg="#080f1b",fg="#829ab8",padx=24,pady=25,anchor="w").pack(side="bottom",fill="x")
    workspace=tk.Frame(body,bg="#0b1220");workspace.pack(side="left",fill="both",expand=True,padx=28,pady=25)
    tk.Label(workspace,text="MAKE YOURSELF AT HOME",font=(setup_font,9,"bold"),fg="#82bdbe",bg="#0b1220").pack(anchor="w")
    tk.Label(workspace,text="Your next words,\nin English.",font=(setup_font,29,"bold"),justify="left",fg="#f0f4fc",bg="#0b1220").pack(anchor="w",pady=(9,7))
    tk.Label(workspace,text="We're getting your caption studio ready. Compatible Python,\nall the libraries, and both speech models — in one place.",
             font=(setup_font,10),justify="left",fg="#8ea5c1",bg="#0b1220").pack(anchor="w",pady=(0,22))
    progress_card=tk.Frame(workspace,bg="#152235",highlightbackground="#2a3f58",highlightthickness=1,padx=20,pady=19)
    progress_card.pack(fill="x")
    stage_title=tk.StringVar(value="Preparing your studio")
    tk.Label(progress_card,textvariable=stage_title,font=(setup_font,14,"bold"),fg="#e7f2fd",bg="#152235",anchor="w").pack(fill="x")
    stage_detail=tk.StringVar(value="Setup starts automatically. You can leave this window open.")
    status = tk.StringVar(value="Checking your installation…")
    tk.Label(progress_card,textvariable=stage_detail,font=(setup_font,9),fg="#86a2c0",bg="#152235",anchor="w").pack(fill="x",pady=(6,14))
    progress=ttk.Progressbar(progress_card,style="Studio.Horizontal.TProgressbar",mode="determinate",maximum=6,value=0)
    progress.pack(fill="x")
    status_label=tk.Label(progress_card,textvariable=status,font=(setup_font,9),fg="#a6bacf",bg="#152235",anchor="w",justify="left",wraplength=560)
    status_label.pack(fill="x",pady=(13,0))
    progress_card.bind("<Configure>",lambda event:status_label.configure(wraplength=max(160,event.width-45)))
    details_open=tk.BooleanVar(value=False)
    details_button=tk.Checkbutton(workspace,text="Show setup details",variable=details_open,bg="#0b1220",fg="#91a9c6",
                    activebackground="#0b1220",activeforeground="#c6e5ef",selectcolor="#1a2b40",font=(setup_font,9),highlightthickness=0)
    details_button.pack(anchor="w",pady=(14,7))
    output = scrolledtext.ScrolledText(workspace,height=6,bg="#080f1b",fg="#a7bdd5",insertbackground="white",
                                      relief="flat",borderwidth=0,wrap="word",font=("Consolas",9),padx=12,pady=10)
    def show_details():
        if details_open.get():
            output.pack(fill="both",expand=True,before=buttons,pady=(0,12))
        else:
            output.pack_forget()
    details_button.configure(command=show_details)
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
        output.delete("1.0", "end");progress["value"]=0
        stage_title.set("Preparing your studio")
        retry.configure(state="disabled");cancel.configure(state="normal",text="Cancel")
        def worker():
            try:
                runner.run();messages.put(("done",None))
            except SetupCancelled as exc:
                emit(str(exc)+"\n");messages.put(("cancelled",str(exc)))
            except Exception as exc:
                LOG.exception("Automatic setup failed");emit(traceback.format_exc());messages.put(("error",str(exc)))
        threading.Thread(target=worker,name="automatic-setup",daemon=True).start()
    def poll():
        try:
            while True:
                kind,value=messages.get_nowait()
                if kind=="line":
                    output.insert("end",value);output.see("end")
                    for line in value.strip().splitlines():
                        if line.startswith("[STEP "):
                            try:
                                number=int(line[6]);title=line.split("] ",1)[1]
                                stage_title.set(title);stage_detail.set(f"Step {number} of 6 · Automatic setup")
                                progress["value"]=number-1
                                for i,label in enumerate(stage_labels,1):
                                    label.configure(fg="#a2dacc" if i<number else "#ebf6ff" if i==number else "#6f86a5",
                                                    bg="#15253a" if i==number else "#080f1b")
                            except (ValueError,IndexError):
                                pass
                        else:
                            status.set(line[-190:])
                else:
                    state["running"]=False
                    if state["log"]:
                        state["log"].close();state["log"]=None
                    if state["closing"]:
                        window.destroy();return
                    if kind=="done":
                        state["success"]=True;progress["value"]=6;window.destroy();return
                    retry.configure(state="normal",text="Retry setup");cancel.configure(text="Close")
                    if kind=="error":
                        stage_title.set("Let's get this sorted")
                        stage_detail.set("Setup paused · Your progress is saved")
                        status.set("The actual error is in the details. Copy the log if you need help.")
                        details_open.set(True);show_details()
                        summary=setup_error_summary(value)
                        messagebox.showerror("Setup paused",summary+f"\n\nFull details: {log_path}\n\nUse Retry setup, or Copy setup log to share the error.")
                    else:
                        stage_title.set("Pick up where you left off")
                        status.set("Setup cancelled. Retry when you're ready.")
        except queue.Empty:
            pass
        window.after(100,poll)
    def close():
        if state["running"]:
            state["closing"]=True;runner.cancel()
            status.set("Cancelling… Waiting for any running installer to finish safely.")
            cancel.configure(state="disabled")
        else:
            window.destroy()
    def copy_log():
        window.clipboard_clear()
        try:
            window.clipboard_append(log_path.read_text(encoding="utf-8"))
            status.set("Log copied. Paste it when asking for help.")
        except OSError:
            status.set("Your setup log hasn't been written yet.")
    buttons=tk.Frame(workspace,bg="#0b1220");buttons.pack(side="bottom",fill="x",pady=(12,0))
    def button(text,command,primary=False):
        return tk.Button(buttons,text=text,command=command,font=(setup_font,10,"bold" if primary else "normal"),
                         bg="#95d9cc" if primary else "#1d2d43",fg="#132735" if primary else "#bfcee1",
                         activebackground="#b3ebdf" if primary else "#2d445f",activeforeground="#132735" if primary else "white",
                         disabledforeground="#597184",relief="flat",borderwidth=0,highlightthickness=0,padx=15,pady=10,cursor="hand2")
    retry=button("Retry setup",install,True);retry.pack(side="left")
    button("Copy setup log",copy_log).pack(side="left",padx=9)
    cancel=button("Cancel",close);cancel.pack(side="right")
    window.protocol("WM_DELETE_WINDOW",close)
    window.after(100,poll);window.after(300,install);window.mainloop()
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
    from PySide6.QtCore import Qt, QObject, Signal, QTimer, QPoint, QRectF, QLockFile, QSize
    from PySide6.QtGui import QColor, QFont, QFontMetrics, QIcon, QPainter, QPixmap, QAction, QPen, QLinearGradient, QRadialGradient, QPainterPath, QPalette
    from PySide6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout, QHBoxLayout,
        QFormLayout, QComboBox, QSpinBox, QDoubleSpinBox, QSlider, QCheckBox, QTabWidget,
        QPlainTextEdit, QProgressBar, QMessageBox, QFileDialog, QSystemTrayIcon, QMenu,
        QGroupBox, QScrollArea, QSizePolicy, QFrame, QGridLayout, QToolButton, QGraphicsDropShadowEffect, QButtonGroup,
    )

    class Bridge(QObject):
        event = Signal(str, object)

    def glyph(kind, color="#a8b5d0"):
        pixmap = QPixmap(28, 28)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(QPen(QColor(color), 1.7))
        if kind in ("studio", "captions"):
            painter.drawRoundedRect(QRectF(4, 6, 20, 15), 4, 4)
            if kind == "studio":
                for x, height in ((9, 4), (14, 9), (19, 6)):
                    painter.drawLine(x, 14 - height // 2, x, 14 + height // 2)
            else:
                painter.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
                painter.drawText(QRectF(4, 6, 20, 15), Qt.AlignmentFlag.AlignCenter, "CC")
        elif kind == "history":
            painter.drawEllipse(QRectF(5, 5, 18, 18))
            painter.drawLine(14, 9, 14, 14)
            painter.drawLine(14, 14, 18, 16)
        elif kind == "help":
            painter.drawEllipse(QRectF(5, 5, 18, 18))
            painter.setFont(QFont("Segoe UI", 12, QFont.Weight.DemiBold))
            painter.drawText(QRectF(5, 4, 18, 20), Qt.AlignmentFlag.AlignCenter, "?")
        elif kind == "play":
            path = QPainterPath()
            path.moveTo(10, 6); path.lineTo(22, 14); path.lineTo(10, 22); path.closeSubpath()
            painter.fillPath(path, QColor(color))
        elif kind == "stop":
            painter.setBrush(QColor(color))
            painter.drawRoundedRect(QRectF(8, 8, 12, 12), 3, 3)
        elif kind == "shield":
            path = QPainterPath()
            path.moveTo(14, 4); path.lineTo(23, 8); path.lineTo(22, 17)
            path.quadTo(19, 23, 14, 25); path.quadTo(7, 22, 6, 16)
            path.lineTo(5, 8); path.closeSubpath()
            painter.drawPath(path)
            painter.drawLine(10, 14, 13, 17); painter.drawLine(13, 17, 19, 11)
        painter.end()
        return QIcon(pixmap)

    class GlassCanvas(QWidget):
        def paintEvent(self, event):
            painter = QPainter(self)
            painter.fillRect(self.rect(), QColor("#090e1a"))
            for x, y, radius, color in (
                (self.width() * .92, 0, 730, QColor(39, 75, 107, 90)),
                (self.width() * .35, self.height(), 660, QColor(70, 43, 117, 66)),
                (self.width() * .58, self.height() * .4, 480, QColor(24, 86, 90, 26)),
            ):
                gradient = QRadialGradient(x, y, radius)
                gradient.setColorAt(0, color)
                gradient.setColorAt(1, QColor(color.red(), color.green(), color.blue(), 0))
                painter.fillRect(self.rect(), gradient)

    class CaptionPreview(QWidget):
        """Static vector artwork: no image download, web engine or animation loop."""
        def __init__(self, settings, parent=None):
            super().__init__(parent)
            self.settings = settings
            self.caption = "A world of voices.\nOne clear language."
            self.setMinimumHeight(185)
            self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        def paintEvent(self, event):
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            bounds = QRectF(self.rect()).adjusted(1, 1, -1, -1)
            clip = QPainterPath(); clip.addRoundedRect(bounds, 15, 15)
            painter.setClipPath(clip)
            sky = QLinearGradient(0, 0, self.width(), self.height())
            sky.setColorAt(0, QColor("#203a49")); sky.setColorAt(.5, QColor("#19243b")); sky.setColorAt(1, QColor("#393057"))
            painter.fillRect(self.rect(), sky)
            glow = QRadialGradient(self.width() * .7, self.height() * .2, self.width() * .55)
            glow.setColorAt(0, QColor(128, 185, 170, 70)); glow.setColorAt(1, QColor(58, 98, 128, 0))
            painter.fillRect(self.rect(), glow)
            # A quiet mountain silhouette gives the subtitle sample a cinematic setting.
            for offset, tint in ((.18, "#202e44"), (.35, "#172437"), (.50, "#121d2e")):
                path = QPainterPath(); path.moveTo(-10, self.height())
                for i, value in enumerate((.7, .35, .55, .13, .48, .25, .55, .3, .7)):
                    path.lineTo(i * self.width() / 8, self.height() * (value * .55 + offset))
                path.lineTo(self.width()+10, self.height()); path.closeSubpath()
                painter.fillPath(path, QColor(tint))
            painter.setFont(QFont("Segoe UI", 8, QFont.Weight.DemiBold))
            painter.setPen(QColor("#d2e0e7"))
            painter.drawText(QRectF(17, 13, self.width()-34, 18), Qt.AlignmentFlag.AlignLeft, "CAPTION PREVIEW")
            painter.setPen(QColor("#8cbbba"))
            painter.drawText(QRectF(17, 13, self.width()-34, 18), Qt.AlignmentFlag.AlignRight, "ENGLISH")
            caption_box = QRectF(17, self.height() * .48, self.width()-34, self.height() * .39)
            painter.setPen(QPen(QColor(255,255,255,35), 1))
            painter.setBrush(QColor(9, 16, 28, int(self.settings.opacity * 2.2)))
            painter.drawRoundedRect(caption_box, 10, 10)
            painter.setFont(QFont("Segoe UI", min(17, max(11, self.settings.font_size // 2)), QFont.Weight.DemiBold))
            painter.setPen(QColor("#f4f8fc"))
            text = self.caption
            if len(text) > 120:
                text = text[:117] + "…"  # this miniature is a preview; full text remains in the overlay/history
            painter.drawText(caption_box.adjusted(12, 5, -12, -5),
                             Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap, text)
            painter.setClipping(False)
            painter.setPen(QPen(QColor(217,234,255,35), 1)); painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(bounds, 15, 15)

    class AudioScope(QWidget):
        def __init__(self):
            super().__init__()
            self.values = collections.deque([0.02] * 48, maxlen=48)
            self.setFixedHeight(48)
        def push(self, db):
            self.values.append(max(.02, min(1., (db+70)/55)))
            self.update()
        def paintEvent(self, event):
            painter = QPainter(self)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            step = self.width() / 48
            for index, value in enumerate(self.values):
                height = max(2, value * (self.height()-6))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(97, 210, 190, 60 + int(value * 170)))
                painter.drawRoundedRect(QRectF(index*step+1, (self.height()-height)/2,
                                              max(1, step-3), height), 2, 2)

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
            glass = QLinearGradient(0,0,0,self.height())
            alpha = min(255, round(self.settings.opacity*2.55))
            glass.setColorAt(0, QColor(28,40,58,alpha))
            glass.setColorAt(1, QColor(7,13,24,alpha))
            painter.setPen(QPen(QColor(220,239,255,45),1))
            painter.setBrush(glass)
            painter.drawRoundedRect(QRectF(self.rect()).adjusted(1,1,-1,-1), 16, 16)
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
            self.setWindowTitle(f"{APP_NAME} · Translation studio")
            area = app.primaryScreen().availableGeometry()
            self.resize(min(1180, max(920, area.width()-60)), min(850, max(620, area.height()-60)))
            self.setMinimumSize(880, 600)
            self.setWindowIcon(icon)
            self.overlay = Overlay(self.settings)
            if self.settings.overlay_x != -1 or self.settings.overlay_y != -1:
                candidate = QPoint(self.settings.overlay_x, self.settings.overlay_y)
                if any(s.availableGeometry().contains(candidate + QPoint(self.overlay.width()//2, self.overlay.height()//2))
                       for s in app.screens()):
                    self.overlay.move(candidate)
            main = GlassCanvas()
            self.setCentralWidget(main)
            outer = QHBoxLayout(main); outer.setSpacing(0); outer.setContentsMargins(0,0,0,0)
            sidebar = QFrame(); sidebar.setObjectName("sidebar"); sidebar.setFixedWidth(186)
            side = QVBoxLayout(sidebar); side.setContentsMargins(18,28,18,22); side.setSpacing(8)
            brand = QLabel("LC"); brand.setObjectName("brandMark"); brand.setFixedSize(46,46)
            brand.setAlignment(Qt.AlignmentFlag.AlignCenter)
            side.addWidget(brand)
            brand_title = QLabel("Live Captions"); brand_title.setObjectName("brandTitle")
            side.addWidget(brand_title)
            tag = QLabel("A little closer to the world."); tag.setWordWrap(True); tag.setObjectName("muted")
            side.addWidget(tag); side.addSpacing(30)
            eyebrow = QLabel("WORKSPACE"); eyebrow.setObjectName("eyebrow"); side.addWidget(eyebrow)
            self.nav_buttons = []
            self.nav_group = QButtonGroup(self)
            for index, (label, kind) in enumerate((("Studio", "studio"), ("Captions", "captions"),
                                                    ("History", "history"), ("Help & setup", "help"))):
                button = QPushButton(label); button.setObjectName("navButton")
                button.setIcon(glyph(kind)); button.setIconSize(QSize(24,24)); button.setCheckable(True)
                button.clicked.connect(lambda checked=False, i=index: self.tabs.setCurrentIndex(i))
                self.nav_buttons.append(button); self.nav_group.addButton(button); side.addWidget(button)
            side.addStretch()
            privacy = QLabel("ON YOUR DEVICE"); privacy.setObjectName("eyebrow"); side.addWidget(privacy)
            private = QLabel("Your audio stays yours.\nNo account. No cloud API.")
            private.setObjectName("muted"); private.setWordWrap(True); side.addWidget(private)
            side.addSpacing(12)
            version = QLabel(f"DESKTOP  /  v{VERSION}"); version.setObjectName("version"); side.addWidget(version)
            outer.addWidget(sidebar)
            workspace = QWidget(); content = QVBoxLayout(workspace)
            content.setContentsMargins(26,24,26,22); content.setSpacing(14)
            top = QHBoxLayout()
            headings = QVBoxLayout(); headings.setSpacing(4)
            self.page_title = QLabel("Translation studio"); self.page_title.setObjectName("pageTitle")
            self.page_subtitle = QLabel("A world of voices. One clear language.")
            self.page_subtitle.setObjectName("subtitle"); self.page_subtitle.setWordWrap(True)
            headings.addWidget(self.page_title); headings.addWidget(self.page_subtitle); top.addLayout(headings,1)
            self.state_badge = QLabel("●  ON-DEVICE"); self.state_badge.setObjectName("stateBadge")
            top.addWidget(self.state_badge,0,Qt.AlignmentFlag.AlignTop); content.addLayout(top)
            self.tabs = QTabWidget(); self.tabs.tabBar().hide(); self.tabs.setObjectName("workspaceTabs")
            self.make_translation_tab(); self.make_caption_tab(); self.make_history_tab()
            help_page = QWidget(); help_layout = QVBoxLayout(help_page); help_layout.setContentsMargins(0,0,0,0)
            help_heading = QLabel("Make yourself at home"); help_heading.setObjectName("sectionTitle")
            help_layout.addWidget(help_heading)
            help_text = QPlainTextEdit(); help_text.setPlainText(HELP); help_text.setReadOnly(True)
            help_layout.addWidget(help_text); self.tabs.addTab(help_page, "Help")
            self.tabs.currentChanged.connect(self.navigate); self.navigate(0)
            content.addWidget(self.tabs,1)
            footer = QFrame(); footer.setObjectName("controlDock")
            dock = QVBoxLayout(footer); dock.setContentsMargins(16,13,16,13); dock.setSpacing(6)
            self.warning = QLabel(""); self.warning.setObjectName("warningLabel"); self.warning.setWordWrap(True)
            dock.addWidget(self.warning)
            row = QHBoxLayout(); row.setSpacing(9)
            self.status = QLabel("Ready — automatic setup prepares your models")
            self.status.setObjectName("statusText"); self.status.setWordWrap(True); row.addWidget(self.status,1)
            self.start_button = QPushButton("Start captions"); self.start_button.setObjectName("primary")
            self.start_button.setIcon(glyph("play", "#132535")); self.start_button.clicked.connect(self.start)
            self.stop_button = QPushButton("Stop"); self.stop_button.setObjectName("stopButton")
            self.stop_button.setIcon(glyph("stop")); self.stop_button.setEnabled(False); self.stop_button.clicked.connect(self.stop)
            self.min_button = QPushButton("To tray"); self.min_button.clicked.connect(self.minimize_tray)
            row.addWidget(self.start_button); row.addWidget(self.stop_button); row.addWidget(self.min_button)
            dock.addLayout(row); content.addWidget(footer); outer.addWidget(workspace,1)
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
            self.update_layout_density()

        def navigate(self, index):
            titles = (("Translation studio", "A world of voices. One clear language."),
                      ("Your captions, your way", "Find the perfect place for every word."),
                      ("A little context", "Revisit the words you just heard."),
                      ("Help & setup", "Everything you need to feel at home."))
            self.page_title.setText(titles[index][0]); self.page_subtitle.setText(titles[index][1])
            for i, button in enumerate(self.nav_buttons):
                button.setChecked(i == index)

        def card(self, title, subtitle=""):
            frame = QFrame(); frame.setObjectName("glassCard")
            layout = QVBoxLayout(frame); layout.setContentsMargins(18,16,18,17); layout.setSpacing(11)
            heading = QLabel(title); heading.setObjectName("sectionTitle"); layout.addWidget(heading)
            if subtitle:
                note = QLabel(subtitle); note.setWordWrap(True); note.setObjectName("muted"); layout.addWidget(note)
            return frame, layout

        def form_page(self, label):
            page = QWidget(); page.setObjectName("pageBody")
            layout = QVBoxLayout(page); layout.setContentsMargins(0,0,6,0); layout.setSpacing(14)
            scroll = QScrollArea(); scroll.setWidgetResizable(True); scroll.setFrameShape(QFrame.Shape.NoFrame)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
            scroll.setWidget(page); self.tabs.addTab(scroll, label)
            return layout

        def make_translation_tab(self):
            layout = self.form_page("Translation")
            grid = QGridLayout(); grid.setSpacing(14)
            self.studio_grid = grid
            left = QVBoxLayout(); left.setSpacing(14)
            right = QVBoxLayout(); right.setSpacing(14)
            self.studio_right = right
            frame, card = self.card("Translation engine", "Choose your balance of detail and speed.")
            self.mode = QComboBox(self); self.mode.hide()
            for label, value in MODEL_LABELS.items():
                self.mode.addItem(label, value)
            self.mode.setCurrentIndex(self.mode.findData(self.settings.model))
            choices = QHBoxLayout(); choices.setSpacing(8); self.mode_cards = {}
            self.mode_button_group = QButtonGroup(self)
            for name, value, hint in (("Quality", "small", "Richer context · Small"), ("Fast", "base", "Lighter & quicker · Base")):
                button = QPushButton(name + "\n" + hint); button.setObjectName("modeCard")
                button.setCheckable(True); button.setMinimumHeight(76)
                button.setToolTip("Whisper small, INT8" if value == "small" else "Whisper base, INT8")
                button.clicked.connect(lambda checked=False, m=value: self.mode.setCurrentIndex(self.mode.findData(m)))
                self.mode_cards[value] = button; self.mode_button_group.addButton(button); choices.addWidget(button,1)
            card.addLayout(choices)
            label = QLabel("SPOKEN LANGUAGE"); label.setObjectName("eyebrow"); card.addWidget(label)
            self.language = QComboBox(); self.language.addItems(LANGUAGES)
            self.language.setCurrentText(self.settings.language); card.addWidget(self.language)
            output = QLabel("↳  Translated into English"); output.setObjectName("accentText"); card.addWidget(output)
            self.model_status = QLabel(); self.model_status.setWordWrap(True); self.model_status.setObjectName("muted")
            card.addWidget(self.model_status)
            self.download_button = QPushButton("Download selected model"); self.download_button.setObjectName("quietButton")
            self.download_button.clicked.connect(self.download); card.addWidget(self.download_button)
            self.download_progress = QProgressBar(); self.download_progress.setRange(0,0); self.download_progress.hide()
            card.addWidget(self.download_progress)
            self.cancel_download_button = QPushButton("Cancel download"); self.cancel_download_button.clicked.connect(self.cancel_download)
            self.cancel_download_button.hide(); card.addWidget(self.cancel_download_button)
            left.addWidget(frame)
            frame, card = self.card("Audio source", "Listen to the sound playing on your PC.")
            self.devices = QComboBox(); self.devices.addItem("Windows default at last refresh", None)
            self.devices.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            self.devices.setMinimumContentsLength(12); card.addWidget(self.devices)
            self.refresh_button = QPushButton("Refresh devices"); self.refresh_button.setObjectName("quietButton")
            self.refresh_button.clicked.connect(self.refresh_devices); card.addWidget(self.refresh_button)
            left.addWidget(frame); left.addStretch()
            frame, card = self.card("The words, without the distance", "A floating caption layer, wherever you watch.")
            self.studio_preview = CaptionPreview(self.settings); card.addWidget(self.studio_preview,1)
            right.addWidget(frame)
            frame, card = self.card("Listening room")
            self.audio_scope = AudioScope(); card.addWidget(self.audio_scope)
            self.meter = QProgressBar(); self.meter.setRange(0,80); self.meter.setValue(0)
            self.meter.setFixedHeight(6); self.meter.setTextVisible(False); self.meter.setFormat("Audio level")
            card.addWidget(self.meter)
            self.audio_level_text = QLabel("Waiting for playback audio"); self.audio_level_text.setObjectName("muted")
            card.addWidget(self.audio_level_text)
            self.metrics = QLabel("Processing —  ·  Delay —  ·  Skipped: 0")
            self.metrics.setWordWrap(True); self.metrics.setObjectName("muted"); card.addWidget(self.metrics)
            self.resources = QLabel("CPU —  ·  App memory —"); self.resources.setWordWrap(True)
            self.resources.setObjectName("muted"); card.addWidget(self.resources)
            right.addWidget(frame); right.addStretch()
            grid.addLayout(left,0,0); grid.addLayout(right,0,1)
            grid.setColumnStretch(0,1); grid.setColumnStretch(1,1); layout.addLayout(grid)
            advanced = QFrame(); advanced.setObjectName("glassCard")
            advanced_layout = QVBoxLayout(advanced); advanced_layout.setContentsMargins(16,9,16,9)
            self.advanced_button = QToolButton(); self.advanced_button.setText("Performance tuning")
            self.advanced_button.setCheckable(True); self.advanced_button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
            self.advanced_button.setArrowType(Qt.ArrowType.RightArrow); self.advanced_button.setObjectName("advancedButton")
            advanced_layout.addWidget(self.advanced_button)
            details = QWidget(); form = QFormLayout(details); form.setContentsMargins(2,8,2,8)
            self.threads = QSpinBox(); self.threads.setRange(1,min(4,os.cpu_count() or 2)); self.threads.setValue(self.settings.threads)
            self.threads.setToolTip("Two threads is a good starting point with your browser open.")
            self.chunk = QDoubleSpinBox(); self.chunk.setRange(2,10); self.chunk.setSingleStep(.5)
            self.chunk.setSuffix(" seconds"); self.chunk.setValue(self.settings.chunk_seconds)
            self.silence = QDoubleSpinBox(); self.silence.setRange(.3,1.5); self.silence.setSingleStep(.1)
            self.silence.setSuffix(" seconds"); self.silence.setValue(self.settings.silence_seconds)
            self.gate = QSpinBox(); self.gate.setRange(-70,-20); self.gate.setSuffix(" dB"); self.gate.setValue(self.settings.gate_db)
            form.addRow("CPU threads",self.threads); form.addRow("Maximum phrase",self.chunk)
            form.addRow("End phrase after silence",self.silence); form.addRow("Speech gate",self.gate)
            advanced_layout.addWidget(details); details.hide()
            def expand(checked):
                details.setVisible(checked)
                self.advanced_button.setArrowType(Qt.ArrowType.DownArrow if checked else Qt.ArrowType.RightArrow)
            self.advanced_button.toggled.connect(expand); layout.addWidget(advanced); layout.addStretch()
            self.mode.currentIndexChanged.connect(self.mode_changed)
            self.language.currentTextChanged.connect(self.save_preferences)
            for control in (self.threads,self.chunk,self.silence,self.gate):
                control.valueChanged.connect(self.save_preferences)
            self.sync_mode_cards()

        def sync_mode_cards(self):
            for model, button in self.mode_cards.items():
                button.setChecked(model == self.mode.currentData())

        def make_caption_tab(self):
            layout = self.form_page("Captions")
            self.caption_grid = QGridLayout(); self.caption_grid.setSpacing(14)
            left = QVBoxLayout(); left.setSpacing(14)
            self.caption_right = QVBoxLayout(); self.caption_right.setSpacing(14)
            frame, card = self.card("Make it yours", "Adjustments apply immediately. Preview to position your floating captions.")
            form = QFormLayout(); form.setVerticalSpacing(12)
            self.screen_combo = QComboBox(); self.refresh_screens(initial=True); form.addRow("Display",self.screen_combo)
            self.caption_controls = {}; self.caption_sliders = {}
            specs = (("Text size", "font_size",16,56," pt"), ("Glass opacity", "opacity",15,100," %"),
                     ("Caption width", "width_percent",35,95," %"), ("Bottom spacing", "bottom_percent",0,70," %"),
                     ("Keep visible", "display_seconds",3,30," sec"))
            for label,key,low,high,suffix in specs:
                control = QSpinBox(); control.setRange(low,high); control.setValue(getattr(self.settings,key))
                control.setSuffix(suffix); control.setFixedWidth(84)
                slider = QSlider(Qt.Orientation.Horizontal); slider.setRange(low,high); slider.setValue(control.value())
                slider.valueChanged.connect(control.setValue); control.valueChanged.connect(slider.setValue)
                self.caption_controls[key] = control; self.caption_sliders[key] = slider
                control.valueChanged.connect(lambda _value,k=key: self.appearance_changed(k))
                row = QHBoxLayout(); row.setSpacing(9); row.addWidget(slider,1); row.addWidget(control)
                form.addRow(label,row)
            card.addLayout(form)
            self.lock_box = QCheckBox("Lock captions · let clicks pass through")
            self.lock_box.setChecked(self.settings.locked); self.lock_box.toggled.connect(self.lock_changed)
            card.addWidget(self.lock_box); left.addWidget(frame); left.addStretch()
            frame, preview_layout = self.card("A quieter way to understand", "Your captions above whatever you're watching.")
            self.style_preview = CaptionPreview(self.settings); self.style_preview.setFixedHeight(210)
            preview_layout.addWidget(self.style_preview); self.caption_right.addWidget(frame)
            frame, actions = self.card("Find your perfect spot")
            preview = QPushButton("Preview floating captions"); preview.setObjectName("primary"); preview.clicked.connect(self.preview)
            reset = QPushButton("Reset position"); reset.clicked.connect(self.reset_overlay)
            actions.addWidget(preview); actions.addWidget(reset)
            note = QLabel("Drag the preview while unlocked. Lock it when you're ready to watch.\n\nUse windowed or borderless playback if fullscreen hides your captions.")
            note.setWordWrap(True); note.setObjectName("muted"); actions.addWidget(note)
            self.caption_right.addWidget(frame); self.caption_right.addStretch()
            self.caption_grid.addLayout(left,0,0); self.caption_grid.addLayout(self.caption_right,0,1)
            self.caption_grid.setColumnStretch(0,1); self.caption_grid.setColumnStretch(1,1)
            layout.addLayout(self.caption_grid); layout.addStretch()
            self.screen_combo.currentIndexChanged.connect(lambda _: self.appearance_changed("screen"))

        def resizeEvent(self, event):
            super().resizeEvent(event)
            self.update_layout_density()

        def update_layout_density(self):
            if not hasattr(self,"caption_grid") or not hasattr(self,"studio_grid"):
                return
            compact = self.width() < 1020
            if getattr(self,"compact_layout",None) == compact:
                return
            self.compact_layout = compact
            for grid,right in ((self.studio_grid,self.studio_right),(self.caption_grid,self.caption_right)):
                grid.removeItem(right)
                grid.addLayout(right,1 if compact else 0,0 if compact else 1)
                grid.setColumnStretch(0,1); grid.setColumnStretch(1,0 if compact else 1)

        def make_history_tab(self):
            page = QWidget(); layout = QVBoxLayout(page); layout.setContentsMargins(0,0,0,0); layout.setSpacing(14)
            frame, card = self.card("The conversation so far", "Your latest 500 captions. Kept in memory until you choose to export.")
            self.history_view = QPlainTextEdit(); self.history_view.setReadOnly(True)
            self.history_view.setPlaceholderText("Your captions will appear here.\n\nStart listening in Studio, then come back to revisit a phrase.")
            self.history_view.setMaximumBlockCount(1000); card.addWidget(self.history_view,1)
            layout.addWidget(frame,1)
            row = QHBoxLayout()
            export = QPushButton("Export transcript…"); export.setObjectName("primary"); export.clicked.connect(self.export_history)
            clear = QPushButton("Clear history"); clear.clicked.connect(self.clear_history)
            row.addWidget(export); row.addWidget(clear); row.addStretch(); layout.addLayout(row)
            self.tabs.addTab(page,"History")

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
            self.sync_mode_cards()
            self.save_preferences()
            self.update_model_status()
            self.update_controls()

        def update_model_status(self):
            ready = model_ready(root, self.mode.currentData())
            self.model_status.setText("Downloaded · ready for offline use" if ready else "Not downloaded yet")
            self.download_button.setText("Repair / redownload model" if ready else "Download selected model")

        def update_controls(self):
            busy = self.engine is not None or self.downloading
            self.state_badge.setText("●  ACTIVE" if self.engine and not self.engine.stop_event.is_set() else "●  STOPPING" if self.engine else "●  MODEL SETUP" if self.downloading else "●  ON-DEVICE")
            for button in self.mode_cards.values():
                button.setEnabled(not busy)
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
            self.studio_preview.update()
            self.style_preview.update()
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
                self.audio_scope.push(value)
                self.audio_level_text.setText(f"Playback level  {value:.0f} dB" if value > -79 else "Waiting for playback audio")
                self.meter.setFormat(f"{value:.0f} dB")
            elif kind == "caption":
                if self.engine and not self.engine.stop_event.is_set():
                    self.overlay.set_caption(value)
                    self.studio_preview.caption = value
                    self.studio_preview.update()
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
                self.audio_scope.values = collections.deque([.02]*48, maxlen=48)
                self.audio_scope.update()
                self.audio_level_text.setText("Waiting for playback audio")
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
    palette = QPalette()
    for role, color in ((QPalette.ColorRole.Window, "#0c1423"), (QPalette.ColorRole.WindowText, "#e5ecf6"),
                        (QPalette.ColorRole.Base, "#101c2e"), (QPalette.ColorRole.AlternateBase, "#16243a"),
                        (QPalette.ColorRole.Text, "#e5ecf6"), (QPalette.ColorRole.Button, "#203149"),
                        (QPalette.ColorRole.ButtonText, "#d6e5f8"), (QPalette.ColorRole.Highlight, "#365576"),
                        (QPalette.ColorRole.HighlightedText, "#ffffff")):
        palette.setColor(role, QColor(color))
    app.setPalette(palette)

    app_lock = QLockFile(str(root / "application.lock"))
    if not smoke_test and not app_lock.tryLock(0):
        QMessageBox.information(None, APP_NAME, "Live Captions is already running, or its data folder is locked.\n"
                                "Restore it from the CC icon in the Windows system tray.")
        return 0
    app.setStyleSheet("""
        QWidget { color: #e5ecf6; font-family: 'Segoe UI'; font-size: 13px; background: transparent; }
        QMainWindow { background: #090e1a; }
        QFrame#sidebar { background: rgba(7,12,23,185); border-right: 1px solid rgba(183,208,249,18); }
        QLabel#brandMark { color: #efffff; font-size: 20px; font-weight: 800; border-radius: 14px;
          border: 1px solid rgba(216,255,244,85); background: qlineargradient(x1:0,y1:0,x2:1,y2:1,stop:0 #6b8abf,stop:.5 #5f7aa9,stop:1 #356d78); }
        QLabel#brandTitle { font-size: 17px; font-weight: 700; padding-top: 9px; }
        QLabel#pageTitle { font-size: 28px; font-weight: 700; color: #f3f6fc; }
        QLabel#subtitle { font-size: 13px; color: #94a5bf; }
        QLabel#sectionTitle { font-size: 14px; font-weight: 650; color: #e8effa; }
        QLabel#eyebrow { font-size: 10px; font-weight: 700; color: #8297b4; padding-top: 3px; }
        QLabel#muted { font-size: 12px; color: #93a6bf; }
        QLabel#version { font-size: 10px; color: #637a98; }
        QLabel#accentText { color: #91d9ca; font-size: 12px; }
        QLabel#stateBadge { border: 1px solid rgba(132,221,205,40); border-radius: 12px; color: #8ed8c9;
          font-size: 10px; font-weight: 650; background: rgba(80,165,147,15); padding: 7px 10px; }
        QFrame#glassCard { border: 1px solid rgba(197,220,255,27); border-radius: 17px;
          background: qlineargradient(x1:0,y1:0,x2:.4,y2:1,stop:0 rgba(46,65,89,130),stop:1 rgba(23,34,54,150)); }
        QFrame#controlDock { border: 1px solid rgba(195,218,255,32); border-radius: 15px; background: rgba(23,34,54,205); }
        QLabel#statusText { font-size: 12px; color: #b3c3d9; }
        QLabel#warningLabel { font-size: 12px; color: #e7bd84; }
        QTabWidget#workspaceTabs::pane { border: 0; background: transparent; }
        QPushButton { background: rgba(139,163,202,12); border: 1px solid rgba(184,211,250,38);
          border-radius: 10px; padding: 10px 14px; min-height: 18px; font-weight: 550; }
        QPushButton:hover { border-color: rgba(170,210,243,115); background: rgba(151,189,230,23); }
        QPushButton:pressed { background: rgba(132,166,222,42); border-color: #a2c0e4; }
        QPushButton:focus { border-color: #88b8d8; }
        QPushButton:disabled { color: #60718b; border-color: rgba(150,170,210,16); background: rgba(70,90,125,12); }
        QPushButton#navButton { text-align: left; border-color: transparent; color: #8ea2bf; background: transparent;
          padding: 10px 9px; border-radius: 10px; font-size: 13px; }
        QPushButton#navButton:hover { color: #dbe9fd; background: rgba(113,149,208,13); }
        QPushButton#navButton:checked { color: #eff6ff; border-color: rgba(139,177,235,43);
          background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 rgba(111,143,207,40),stop:1 rgba(80,115,172,14)); }
        QPushButton#primary { color: #101c2b; font-weight: 700; border-color: rgba(195,251,237,130);
          background: qlineargradient(x1:0,y1:0,x2:1,y2:1,stop:0 #a3e4d6,stop:1 #7cbed1); }
        QPushButton#primary:hover { background: #b7f0e3; border-color: #d0fff3; }
        QPushButton#primary:pressed { background: #79bcbe; }
        QPushButton#primary:disabled { background: rgba(117,164,175,30); color: #657e8c; border-color: rgba(140,190,192,25); }
        QPushButton#modeCard { text-align: left; border-radius: 11px; padding: 7px 11px; color: #91a3bd; font-size: 12px; }
        QPushButton#modeCard:checked { color: #e9eefb; border-color: rgba(163,174,234,130);
          background: qlineargradient(x1:0,y1:0,x2:1,y2:1,stop:0 rgba(124,139,204,65),stop:1 rgba(98,111,164,22)); }
        QPushButton#modeCard:disabled { color: #7889a4; }
        QPushButton#quietButton { border: 0; color: #9ebbd7; font-size: 11px; min-height: 12px; padding: 5px 0; text-align: left; background: transparent; }
        QPushButton#quietButton:hover { color: #b3ede5; }
        QToolButton#advancedButton { border: 0; color: #a1b4cf; padding: 6px; text-align: left; }
        QToolButton#advancedButton:hover { color: #e9f5ff; }
        QComboBox, QSpinBox, QDoubleSpinBox { background: rgba(6,14,27,130); border: 1px solid rgba(165,195,235,40);
          border-radius: 9px; padding: 9px 11px; min-height: 19px; selection-background-color: #354f71; }
        QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover { border-color: rgba(156,195,228,100); }
        QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus { border-color: #7babbc; }
        QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled { color: #5c6e88; }
        QComboBox::drop-down { border: 0; width: 27px; }
        QComboBox QAbstractItemView { background: #142136; color: #dce9fa; border: 1px solid #354966;
          selection-background-color: #304766; selection-color: white; outline: 0; padding: 5px; }
        QSpinBox::up-button, QDoubleSpinBox::up-button, QSpinBox::down-button, QDoubleSpinBox::down-button { width: 18px; border: 0; }
        QPlainTextEdit { background: rgba(8,16,30,140); border: 1px solid rgba(174,200,238,25); border-radius: 12px;
          padding: 14px; selection-background-color: #3a5777; font-size: 13px; }
        QProgressBar { border: 0; border-radius: 3px; background: rgba(121,155,195,17); text-align: center; max-height: 7px; }
        QProgressBar::chunk { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #6789b9,stop:1 #8edccc); border-radius: 3px; }
        QSlider::groove:horizontal { height: 4px; border-radius: 2px; background: rgba(133,163,198,32); }
        QSlider::sub-page:horizontal { background: #82bbbc; border-radius: 2px; }
        QSlider::handle:horizontal { width: 13px; height: 13px; margin: -5px 0; border-radius: 7px; background: #ade7db; border: 1px solid #d0f5eb; }
        QSlider::handle:horizontal:hover { background: #e0fff3; }
        QCheckBox { spacing: 10px; padding: 7px 0; color: #b3c5dc; }
        QCheckBox::indicator { width: 17px; height: 17px; border-radius: 5px; border: 1px solid #4a627e; background: #142237; }
        QCheckBox::indicator:checked { background: #90d8c8; border-color: #c2e9de; }
        QScrollArea { border: 0; background: transparent; }
        QScrollBar:vertical { width: 6px; background: transparent; margin: 3px 0; }
        QScrollBar::handle:vertical { background: rgba(142,169,210,62); border-radius: 3px; min-height: 35px; }
        QScrollBar::handle:vertical:hover { background: rgba(142,190,220,110); }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
        QMenu { background: #142136; border: 1px solid #354966; border-radius: 8px; padding: 5px; }
        QMenu::item { padding: 9px 20px; border-radius: 5px; }
        QMenu::item:selected { background: #304766; }
        QToolTip { background: #192a40; color: #dbe8fb; border: 1px solid #425d7c; padding: 7px; }
        QMessageBox { background: #142136; }
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
