"""
Jarvis AI Assistant — Hardened, Resilient & Multi-Modal
======================================================
Architecture:
  - Dual-Mode Engine: Adaptive Voice Mode (wake-word / speech-to-text) +
    Interactive Conversational CLI mode (auto-selected if mic is missing or via --cli).
  - Safety & Security: Zero-eval math engine, confirmed power controls, sanitized
    notes with automated backups, graceful process controls.
  - Multi-Modal Knowledge: Offline Wikipedia summarizer + OpenAI conversational fallback
    (works standalone, or via an optional external client.py module).
  - Advanced Diagnostics: Battery/CPU/RAM, disk health, top memory consumers, network ping.
  - Productivity: Non-blocking asynchronous timers, desktop app launcher, system audio controls.

New in this version (combined from two prior prototypes + original additions):
  - Unique Feature 7: Persistent, wall-clock Scheduled Reminders ("remind me to X at 5pm" /
    "...in 20 minutes") that survive restarts and are checked by a background thread.
  - Unique Feature 8: Custom Voice Routines/Macros — chain multiple commands under one name
    ("create routine morning: what's the time, weather in Kathmandu, news") and replay them
    with "run routine morning".
  - Unique Feature 9: Context-aware AI fallback — keeps a short rolling conversation history
    so follow-up questions to the AI assistant make sense without repeating context.
  - Unique Feature 10: Dual TTS backend — offline pyttsx3 (default) or online gTTS+pygame,
    selectable with --tts-backend, with automatic fallback if one is unavailable.
"""

import argparse
import ast
import json
import math
import operator
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import webbrowser
from datetime import datetime, timedelta
from urllib.parse import quote_plus

import requests

# ---------------------------------------------------------------------------
# Optional / Dynamic Dependencies with Graceful Fallbacks
# ---------------------------------------------------------------------------

try:
    from client import ask_openai, ask_openai_with_context, get_openai_client, is_openai_configured, load_env_file
    load_env_file()
except ImportError:
    def load_env_file():
        """Minimal .env loader so OPENAI_API_KEY etc. work without python-dotenv."""
        env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
        if not os.path.exists(env_path):
            return
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, value = line.partition("=")
                    os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))
        except Exception:
            pass

    load_env_file()

    def is_openai_configured():
        return bool(os.getenv("OPENAI_API_KEY", "").strip())

    try:
        from openai import OpenAI as _OpenAIClient
    except ImportError:
        _OpenAIClient = None

    def get_openai_client():
        if _OpenAIClient is None:
            return None
        key = os.getenv("OPENAI_API_KEY", "").strip()
        if not key:
            return None
        try:
            return _OpenAIClient(api_key=key)
        except Exception:
            return None

    def ask_openai(prompt):
        client = get_openai_client()
        if client is None:
            return "OpenAI client helper is not available."
        try:
            completion = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=[
                    {"role": "system", "content": "You are Jarvis, a concise, helpful virtual assistant."},
                    {"role": "user", "content": prompt},
                ],
            )
            return completion.choices[0].message.content
        except Exception as err:
            return f"Sorry, I couldn't reach OpenAI: {err}"

    def ask_openai_with_context(history):
        """Context-aware variant used by the conversational memory feature."""
        client = get_openai_client()
        if client is None:
            return ask_openai(history[-1]["content"]) if history else "I don't have anything to respond to."
        try:
            messages = [
                {"role": "system", "content": "You are Jarvis, a concise, helpful virtual assistant. "
                                               "Use the prior conversation for context when relevant."}
            ] + history
            completion = client.chat.completions.create(
                model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
                messages=messages,
            )
            return completion.choices[0].message.content
        except Exception as err:
            return f"Sorry, I couldn't reach OpenAI: {err}"
try:
    import sounddevice as sd
except ModuleNotFoundError:
    sd = None

try:
    import speech_recognition as sr
except ModuleNotFoundError:
    sr = None

try:
    import pyttsx3
except ModuleNotFoundError:
    pyttsx3 = None

try:
    from gtts import gTTS
except ModuleNotFoundError:
    gTTS = None

try:
    import pygame
except ModuleNotFoundError:
    pygame = None

try:
    import pywhatkit
except Exception:
    # Some pywhatkit releases perform a network check during import. Keep the
    # assistant usable offline when that optional check or dependency fails.
    pywhatkit = None

try:
    import psutil
except ModuleNotFoundError:
    psutil = None

try:
    import wikipedia
except ModuleNotFoundError:
    wikipedia = None

try:
    import importlib
    music_library = importlib.import_module("musiclibrary")
except ModuleNotFoundError:
    try:
        music_library = importlib.import_module("musicLibrary")
    except ModuleNotFoundError:
        class _MusicLibrary:
            music = {}
        music_library = _MusicLibrary()


# ---------------------------------------------------------------------------
# Configuration & Global State
# ---------------------------------------------------------------------------

SAMPLE_RATE = 16000
WAKE_WORD_DURATION = 3
COMMAND_DURATION = 4
QUESTION_DURATION = 7
MIC_DEVICE_INDEX = None

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
NOTES_FILE = os.path.join(BASE_DIR, "jarvis_notes.txt")
NOTES_BACKUP = os.path.join(BASE_DIR, "jarvis_notes.bak")
REMINDERS_FILE = os.path.join(BASE_DIR, "jarvis_reminders.json")
ROUTINES_FILE = os.path.join(BASE_DIR, "jarvis_routines.json")

EXIT_REQUESTED = False
CLI_MODE = False
QUIET_MODE = False
TTS_BACKEND = "pyttsx3"  # or "gtts"

# Rolling conversational memory for context-aware AI fallback answers.
CONVO_HISTORY = []
CONVO_HISTORY_MAX_TURNS = 6  # keeps last 6 user+assistant exchanges

_reminders_lock = threading.Lock()

# Speech synthesis engine initialization (safely guarded)
_tts_engine = None
_tts_lock = threading.Lock()


def get_tts_engine():
    global _tts_engine
    if _tts_engine is None and pyttsx3 is not None:
        try:
            _tts_engine = pyttsx3.init()
            _tts_engine.setProperty("rate", 185)
        except Exception as err:
            print(f"[Warning] Failed to initialize pyttsx3 engine: {err}")
            _tts_engine = None
    return _tts_engine


recognizer = sr.Recognizer() if sr else None
if recognizer:
    recognizer.dynamic_energy_threshold = True


# ---------------------------------------------------------------------------
# Core Audio & I/O Helpers
# ---------------------------------------------------------------------------

def _speak_gtts(text):
    """Online TTS backend using gTTS + pygame (mirrors the lightweight prototype)."""
    tmp_path = os.path.join(tempfile.gettempdir(), f"jarvis_tts_{int(time.time() * 1000)}.mp3")
    tts = gTTS(text)
    tts.save(tmp_path)

    pygame.mixer.init()
    pygame.mixer.music.load(tmp_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.Clock().tick(10)
    pygame.mixer.music.unload()
    try:
        os.remove(tmp_path)
    except OSError:
        pass


def speak(text):
    """Output speech audio safely and print to console. Backend chosen by TTS_BACKEND."""
    if not text:
        return
    text = str(text).strip()
    print(f"\nJarvis: {text}")

    if QUIET_MODE:
        return

    if TTS_BACKEND == "gtts" and gTTS is not None and pygame is not None:
        try:
            _speak_gtts(text)
            return
        except Exception as err:
            print(f"[gTTS Error]: {err} — falling back to offline TTS.")

    engine = get_tts_engine()
    if engine is None:
        return

    with _tts_lock:
        try:
            engine.say(text)
            engine.runAndWait()
        except Exception as err:
            print(f"[TTS Error]: {err}")


def has_working_microphone():
    """Detect if any working audio input device is available."""
    if sd is None:
        return False
    try:
        devices = sd.query_devices()
        if isinstance(devices, dict):
            return devices.get("max_input_channels", 0) > 0
        for dev in devices:
            if dev.get("max_input_channels", 0) > 0:
                return True
        return False
    except Exception:
        return False


def record_audio(duration, fs=SAMPLE_RATE):
    """Record audio with sounddevice and wrap for speech_recognition."""
    if sd is None or sr is None:
        raise OSError("Audio recording libraries are not available.")

    audio = sd.rec(
        int(duration * fs),
        samplerate=fs,
        channels=1,
        dtype='int16',
        device=MIC_DEVICE_INDEX
    )
    sd.wait()
    return sr.AudioData(audio.tobytes(), fs, 2)


_mic_warning_shown = False

def listen_and_transcribe(duration):
    """Listen for voice input with defensive error handling."""
    global _mic_warning_shown
    if CLI_MODE or not has_working_microphone():
        return None

    try:
        audio_data = record_audio(duration)
    except (OSError, getattr(sd, "PortAudioError", OSError)) as exc:
        if not _mic_warning_shown:
            print(f"[Audio Error] Microphone access failed: {exc}")
            speak("I can't access a microphone. Switching to keyboard input mode.")
            _mic_warning_shown = True
        return None

    if recognizer is None:
        return None

    try:
        return recognizer.recognize_google(audio_data)
    except getattr(sr, "UnknownValueError", Exception):
        return None
    except getattr(sr, "RequestError", Exception):
        print("[Network Warning] Speech recognition service unreachable.")
        return None
    except Exception as exc:
        print(f"[Speech Error] {exc}")
        return None


def prompt_user(prompt_text, duration=COMMAND_DURATION):
    """Prompt user via voice in Voice mode or console in CLI mode."""
    speak(prompt_text)
    if CLI_MODE:
        try:
            return input(f"{prompt_text} > ").strip()
        except (EOFError, KeyboardInterrupt):
            return None
    else:
        ans = listen_and_transcribe(duration)
        if not ans:
            try:
                print("(Microphone timed out. Type response or press Enter to skip)")
                ans = input("> ").strip()
            except Exception:
                ans = None
        return ans


# ---------------------------------------------------------------------------
# Browser & Power Management (Hardened & Confirmed)
# ---------------------------------------------------------------------------

def close_browser():
    """Safely ask for confirmation before terminating browser instances."""
    system = platform.system()
    speak("Closing the browser will close all active tabs and unsaved work. Are you sure? Say or type yes to confirm.")
    reply = prompt_user("Confirm browser close (yes/no):", COMMAND_DURATION) or ""
    if reply.strip().lower() != "yes":
        speak("Cancelled. Your browser was left open.")
        return False

    try:
        if system == "Windows":
            for exe in ("brave.exe", "chrome.exe", "msedge.exe", "firefox.exe"):
                subprocess.run(["taskkill", "/IM", exe], capture_output=True, text=True, check=False)
        elif system == "Darwin":
            subprocess.run(["osascript", "-e", 'quit app "Google Chrome"'], capture_output=True, text=True, check=False)
        else:
            subprocess.run(["pkill", "-f", "chrome"], capture_output=True, text=True, check=False)
        speak("Browser closed.")
        return True
    except Exception as err:
        print("Close browser error:", err)
        speak("Could not close the browser.")
        return False


def _run_power_command(args_by_os, action_label):
    system = platform.system()
    args = args_by_os.get(system)
    if not args:
        speak(f"I don't know how to {action_label} on this operating system yet.")
        return
    try:
        subprocess.run(args, check=False)
    except Exception as e:
        print(f"{action_label} error:", e)
        speak(f"Something went wrong trying to {action_label} the computer.")


def shutdown_computer():
    reply = prompt_user("Are you sure you want to shut down the computer? Say or type yes to confirm.") or ""
    if reply.strip().lower() != "yes":
        speak("Okay, cancelling shutdown.")
        return

    speak("Shutting down in one minute. Say or type 'cancel shutdown' to abort.")
    _run_power_command({
        "Windows": ["shutdown", "/s", "/t", "60"],
        "Darwin": ["osascript", "-e", 'tell app "System Events" to shut down'],
        "Linux": ["shutdown", "+1"],
    }, "shut down")


def cancel_shutdown():
    _run_power_command({
        "Windows": ["shutdown", "/a"],
        "Darwin": None,
        "Linux": ["shutdown", "-c"],
    }, "cancel the shutdown")
    speak("Shutdown command cancelled.")


def restart_computer():
    reply = prompt_user("Are you sure you want to restart the computer? Say or type yes to confirm.") or ""
    if reply.strip().lower() != "yes":
        speak("Okay, cancelling restart.")
        return
    _run_power_command({
        "Windows": ["shutdown", "/r", "/t", "10"],
        "Darwin": ["osascript", "-e", 'tell app "System Events" to restart'],
        "Linux": ["shutdown", "-r", "+1"],
    }, "restart")
    speak("Restarting the computer now.")


def sleep_computer():
    reply = prompt_user("Are you sure you want to put the computer to sleep? Say or type yes to confirm.") or ""
    if reply.strip().lower() != "yes":
        speak("Okay, computer will stay awake.")
        return

    speak("Putting the computer to sleep.")
    system = platform.system()
    if system == "Windows":
        try:
            import ctypes
            ctypes.windll.PowrProf.SetSuspendState(0, 1, 0)
        except Exception:
            _run_power_command({"Windows": ["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"]}, "sleep")
    elif system == "Darwin":
        _run_power_command({"Darwin": ["pmset", "sleepnow"]}, "sleep")
    else:
        _run_power_command({"Linux": ["systemctl", "suspend"]}, "sleep")


# ---------------------------------------------------------------------------
# Unique Feature 1: Offline Knowledge & Wikipedia Engine
# ---------------------------------------------------------------------------

def ask_wikipedia(query):
    """Retrieve concise encyclopedic answers without requiring paid API keys."""
    if not wikipedia:
        speak("Wikipedia package is not installed.")
        return False

    clean_query = query.strip()
    for prefix in ("tell me about", "who is", "who was", "what is", "what was", "define", "explain"):
        if clean_query.lower().startswith(prefix):
            clean_query = clean_query[len(prefix):].strip()
            break

    if not clean_query:
        speak("What topic would you like me to look up on Wikipedia?")
        return False

    try:
        search_results = wikipedia.search(clean_query)
        if not search_results:
            speak(f"I couldn't find any Wikipedia article about {clean_query}.")
            return False

        page_title = search_results[0]
        summary = wikipedia.summary(page_title, sentences=2, auto_suggest=False)
        speak(f"According to Wikipedia: {summary}")
        return True
    except getattr(wikipedia, "DisambiguationError", Exception) as e:
        options = getattr(e, "options", [])[:3]
        if options:
            try:
                summary = wikipedia.summary(options[0], sentences=2, auto_suggest=False)
                speak(f"According to Wikipedia: {summary}")
                return True
            except Exception:
                pass
        speak(f"There are multiple topics for {clean_query}. Please be more specific.")
        return True
    except getattr(wikipedia, "PageError", Exception):
        speak(f"I couldn't find a page matching {clean_query}.")
        return False
    except Exception as err:
        print("Wikipedia lookup error:", err)
        speak("Sorry, I had trouble reaching Wikipedia.")
        return False


# ---------------------------------------------------------------------------
# Unique Feature 2: Safe AST Math Calculator & Unit Converter (Zero-eval)
# ---------------------------------------------------------------------------

SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

SAFE_FUNCTIONS = {
    "sqrt": math.sqrt,
    "abs": abs,
    "round": round,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
}

MAX_MATH_EXPRESSION_LENGTH = 256
MAX_MATH_AST_NODES = 64
MAX_MATH_ABS_VALUE = 1e100


def _validate_math_tree(tree):
    """Reject oversized or computationally expensive calculator expressions."""
    nodes = list(ast.walk(tree))
    if len(nodes) > MAX_MATH_AST_NODES:
        raise ValueError("Expression is too complex.")
    for node in nodes:
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
                raise ValueError("Only numeric constants are supported.")
            if abs(node.value) > MAX_MATH_ABS_VALUE:
                raise ValueError("Number is too large.")
        elif isinstance(node, ast.Call) and node.keywords:
            raise ValueError("Keyword arguments are not supported.")


def _eval_ast_node(node):
    if isinstance(node, ast.Expression):
        return _eval_ast_node(node.body)
    elif isinstance(node, ast.Constant):
        if not isinstance(node.value, bool) and isinstance(node.value, (int, float)):
            if abs(node.value) > MAX_MATH_ABS_VALUE:
                raise ValueError("Number is too large.")
            return node.value
        raise ValueError(f"Unsafe constant type: {type(node.value)}")
    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in SAFE_OPERATORS:
            raise ValueError(f"Unsupported operator: {op_type}")
        left = _eval_ast_node(node.left)
        right = _eval_ast_node(node.right)
        if op_type is ast.Pow and (abs(right) > 100 or abs(left) > 1e10):
            raise ValueError("Exponent is too large.")
        return SAFE_OPERATORS[op_type](left, right)
    elif isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in SAFE_OPERATORS:
            raise ValueError(f"Unsupported unary operator: {op_type}")
        operand = _eval_ast_node(node.operand)
        return SAFE_OPERATORS[op_type](operand)
    elif isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in SAFE_FUNCTIONS:
            if node.keywords:
                raise ValueError("Keyword arguments are not supported.")
            args = [_eval_ast_node(arg) for arg in node.args]
            if len(args) > 2:
                raise ValueError("Too many function arguments.")
            result = SAFE_FUNCTIONS[node.func.id](*args)
            if isinstance(result, (int, float)) and (not math.isfinite(result) or abs(result) > MAX_MATH_ABS_VALUE):
                raise ValueError("Result is outside the supported range.")
            return result
        raise ValueError("Unsupported function call in math expression.")
    else:
        raise ValueError(f"Unsupported AST node: {type(node)}")


def calculate_expression(expression_str):
    """Safely calculate mathematical expression without using eval()."""
    if not isinstance(expression_str, str) or len(expression_str) > MAX_MATH_EXPRESSION_LENGTH:
        speak("That expression is too long to calculate safely.")
        return
    cleaned = expression_str.lower().strip()
    for word in ("calculate", "what is", "how much is", "solve", "equals", "="):
        cleaned = cleaned.replace(word, " ")

    cleaned = cleaned.replace("times", "*").replace("multiplied by", "*").replace("x", "*")
    cleaned = cleaned.replace("divided by", "/").replace("over", "/")
    cleaned = cleaned.replace("plus", "+").replace("add", "+")
    cleaned = cleaned.replace("minus", "-").replace("subtract", "-")
    cleaned = cleaned.replace("to the power of", "**").replace("power of", "**").replace("^", "**")
    cleaned = cleaned.replace("percent of", "* 0.01 *")

    cleaned = re.sub(r"[^0-9\+\-\*\/\.\(\)\s\%\^a-z_]", "", cleaned).strip()
    if not cleaned:
        speak("I couldn't find a valid mathematical expression to calculate.")
        return

    try:
        tree = ast.parse(cleaned, mode='eval')
        _validate_math_tree(tree)
        result = _eval_ast_node(tree)
        if isinstance(result, (int, float)) and (not math.isfinite(result) or abs(result) > MAX_MATH_ABS_VALUE):
            raise ValueError("Result is outside the supported range.")
        if isinstance(result, float) and result.is_integer():
            result = int(result)
        elif isinstance(result, float):
            result = round(result, 4)
        speak(f"The result is {result}")
    except ZeroDivisionError:
        speak("Cannot divide by zero.")
    except Exception:
        speak("Sorry, I could not calculate that expression safely.")


def convert_units(command):
    """Convert common physical units (temperature, distance, weight)."""
    c_to_f = re.search(r"(\-?\d+(?:\.\d+)?)\s*(?:degrees?)?\s*celsius\s+(?:in|to)\s+fahrenheit", command)
    if c_to_f:
        c = float(c_to_f.group(1))
        f = round((c * 9 / 5) + 32, 2)
        speak(f"{c} degrees Celsius is {f} degrees Fahrenheit.")
        return True

    f_to_c = re.search(r"(\-?\d+(?:\.\d+)?)\s*(?:degrees?)?\s*fahrenheit\s+(?:in|to)\s+celsius", command)
    if f_to_c:
        f = float(f_to_c.group(1))
        c = round((f - 32) * 5 / 9, 2)
        speak(f"{f} degrees Fahrenheit is {c} degrees Celsius.")
        return True

    km_to_mi = re.search(r"(\d+(?:\.\d+)?)\s*(?:kilometers?|km)\s+(?:in|to)\s+miles?", command)
    if km_to_mi:
        km = float(km_to_mi.group(1))
        mi = round(km * 0.621371, 2)
        speak(f"{km} kilometers is approximately {mi} miles.")
        return True

    mi_to_km = re.search(r"(\d+(?:\.\d+)?)\s*miles?\s+(?:in|to)\s*(?:kilometers?|km)", command)
    if mi_to_km:
        mi = float(mi_to_km.group(1))
        km = round(mi * 1.60934, 2)
        speak(f"{mi} miles is approximately {km} kilometers.")
        return True

    kg_to_lb = re.search(r"(\d+(?:\.\d+)?)\s*(?:kilograms?|kg)\s+(?:in|to)\s*(?:pounds?|lbs?)", command)
    if kg_to_lb:
        kg = float(kg_to_lb.group(1))
        lb = round(kg * 2.20462, 2)
        speak(f"{kg} kilograms is approximately {lb} pounds.")
        return True

    lb_to_kg = re.search(r"(\d+(?:\.\d+)?)\s*(?:pounds?|lbs?)\s+(?:in|to)\s*(?:kilograms?|kg)", command)
    if lb_to_kg:
        lb = float(lb_to_kg.group(1))
        kg = round(lb * 0.453592, 2)
        speak(f"{lb} pounds is approximately {kg} kilograms.")
        return True

    return False


# ---------------------------------------------------------------------------
# Unique Feature 3: Advanced System Diagnostics (Disk, Top Procs, Ping)
# ---------------------------------------------------------------------------

def system_status():
    if not psutil:
        speak("I need the psutil package installed to check system status.")
        return

    cpu = psutil.cpu_percent(interval=0.1)
    mem = psutil.virtual_memory()
    parts = [f"CPU usage is at {cpu} percent", f"RAM usage is at {mem.percent} percent"]

    battery = psutil.sensors_battery()
    if battery and battery.percent is not None:
        state = "charging" if battery.power_plugged else "not charging"
        parts.append(f"battery is at {round(battery.percent)} percent and {state}")

    speak(", ".join(parts) + ".")


def system_diagnostics():
    if not psutil:
        speak("System diagnostics require the psutil package.")
        return

    parts = []
    try:
        root_path = "C:\\" if platform.system() == "Windows" else "/"
        disk = psutil.disk_usage(root_path)
        free_gb = round(disk.free / (1024 ** 3), 1)
        total_gb = round(disk.total / (1024 ** 3), 1)
        parts.append(f"Primary storage has {free_gb} gigabytes free out of {total_gb} gigabytes")
    except Exception:
        pass

    try:
        top_procs = sorted(
            psutil.process_iter(['name', 'memory_info']),
            key=lambda p: (p.info['memory_info'].rss if p.info and p.info.get('memory_info') else 0),
            reverse=True
        )
        if top_procs:
            top_p = top_procs[0]
            top_name = top_p.info['name']
            top_mb = round(top_p.info['memory_info'].rss / (1024 * 1024))
            parts.append(f"top memory consumer is {top_name} using {top_mb} megabytes")
    except Exception:
        pass

    if parts:
        speak(". ".join(parts) + ".")
    else:
        speak("System diagnostics completed with no issues.")


def ping_network(host="8.8.8.8"):
    system = platform.system()
    param = "-n" if system == "Windows" else "-c"
    cmd = ["ping", param, "1", host]
    try:
        output = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
        if output.returncode == 0:
            match = re.search(r"time[=<](\d+)\s*ms", output.stdout, re.IGNORECASE)
            latency = f"with {match.group(1)} milliseconds latency" if match else "with good latency"
            speak(f"Internet connection is active {latency}.")
        else:
            speak("Internet connection appears to be offline.")
    except Exception:
        speak("Could not verify internet connection.")


# ---------------------------------------------------------------------------
# Unique Feature 4: Safe Desktop Application Launcher
# ---------------------------------------------------------------------------

APP_COMMANDS = {
    "notepad": ["notepad.exe"],
    "calculator": ["calc.exe"],
    "terminal": ["cmd.exe"],
    "command prompt": ["cmd.exe"],
    "task manager": ["taskmgr.exe"],
    "file explorer": ["explorer.exe"],
    "settings": ["explorer.exe", "ms-settings:"],
    "visual studio code": ["code"],
    "vs code": ["code"],
}


def launch_application(app_name):
    app_key = app_name.lower().strip()
    cmd = APP_COMMANDS.get(app_key)
    if not cmd:
        for name, target in APP_COMMANDS.items():
            if name in app_key:
                cmd = target
                break

    if not cmd:
        speak(f"I don't have a configured shortcut for {app_name}.")
        return False

    try:
        subprocess.Popen(cmd)
        speak(f"Opening {app_name}.")
        return True
    except Exception as err:
        print(f"App launch error for {app_name}:", err)
        speak(f"Could not open {app_name}.")
        return False


# ---------------------------------------------------------------------------
# Unique Feature 5: Non-blocking Background Timers
# ---------------------------------------------------------------------------

_active_timers = []


def set_timer(seconds, label="timer"):
    def _timer_worker():
        time.sleep(seconds)
        speak(f"Ding ding! Time's up! Your {label} for {seconds} seconds is complete.")
        if threading.current_thread() in _active_timers:
            _active_timers.remove(threading.current_thread())

    t = threading.Thread(target=_timer_worker, daemon=True)
    _active_timers.append(t)
    t.start()
    speak(f"Timer set for {seconds} seconds.")


def parse_and_set_timer(command):
    match = re.search(r"(\d+)\s*(second|minute|hour)s?", command)
    if not match:
        speak("How many seconds or minutes should I set the timer for?")
        return False

    amount = int(match.group(1))
    unit = match.group(2)
    multiplier = 1
    if unit == "minute":
        multiplier = 60
    elif unit == "hour":
        multiplier = 3600

    total_seconds = amount * multiplier
    set_timer(total_seconds, f"{amount} {unit}")
    return True


# ---------------------------------------------------------------------------
# Unique Feature 6: Master Volume & Audio Controls
# ---------------------------------------------------------------------------

def control_volume(action):
    system = platform.system()
    try:
        if system == "Windows":
            import ctypes
            if action == "mute" or action == "unmute":
                ctypes.windll.user32.keybd_event(0xAD, 0, 0, 0)
                ctypes.windll.user32.keybd_event(0xAD, 0, 2, 0)
                speak("Toggled audio mute.")
            elif action == "up":
                for _ in range(5):
                    ctypes.windll.user32.keybd_event(0xAF, 0, 0, 0)
                    ctypes.windll.user32.keybd_event(0xAF, 0, 2, 0)
                speak("Volume increased.")
            elif action == "down":
                for _ in range(5):
                    ctypes.windll.user32.keybd_event(0xAE, 0, 0, 0)
                    ctypes.windll.user32.keybd_event(0xAE, 0, 2, 0)
                speak("Volume decreased.")
        elif system == "Darwin":
            if action in ("mute", "unmute"):
                subprocess.run(["osascript", "-e", "set volume output muted not (output muted of (get volume settings))"])
                speak("Toggled audio mute.")
            elif action == "up":
                subprocess.run(["osascript", "-e", "set volume output volume ((output volume of (get volume settings)) + 10)"])
                speak("Volume increased.")
            elif action == "down":
                subprocess.run(["osascript", "-e", "set volume output volume ((output volume of (get volume settings)) - 10)"])
                speak("Volume decreased.")
        else:
            speak("System audio control is not supported on this platform yet.")
    except Exception as err:
        print("Volume control error:", err)
        speak("Could not adjust the volume.")


# ---------------------------------------------------------------------------
# Time & Date
# ---------------------------------------------------------------------------

def tell_time():
    now = datetime.now().strftime("%I:%M %p").lstrip("0")
    speak(f"It's {now} right now.")


def tell_date():
    today = datetime.now().strftime("%A, %B %d, %Y")
    speak(f"Today is {today}.")


# ---------------------------------------------------------------------------
# Weather (Open-Meteo, Defensive Parsing)
# ---------------------------------------------------------------------------

WEATHER_CODES = {
    0: "clear sky", 1: "mostly clear", 2: "partly cloudy", 3: "overcast",
    45: "foggy", 48: "foggy with frost", 51: "light drizzle", 53: "drizzle",
    55: "heavy drizzle", 61: "light rain", 63: "rain", 65: "heavy rain",
    71: "light snow", 73: "snow", 75: "heavy snow", 80: "rain showers",
    81: "rain showers", 82: "violent rain showers", 95: "thunderstorms",
    96: "thunderstorms with hail", 99: "severe thunderstorms with hail",
}


def get_weather(city):
    if not city:
        speak("Which city's weather would you like?")
        city = prompt_user("Enter city name:")
        if not city:
            speak("I didn't catch a city name.")
            return

    city = str(city).strip()
    if not city or len(city) > 100 or not re.fullmatch(r"[\w .,'-]+", city, re.UNICODE):
        speak("Please provide a valid city name.")
        return

    try:
        geo_response = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1}, timeout=6
        )
        geo_response.raise_for_status()
        geo = geo_response.json()
        results = geo.get("results")
        if not results:
            speak(f"I couldn't find a place called {city}.")
            return

        place = results[0]
        lat, lon = place["latitude"], place["longitude"]
        label = place.get("name", city)

        forecast_response = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat, "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
            }, timeout=6
        )
        forecast_response.raise_for_status()
        forecast = forecast_response.json()

        current = forecast.get("current", {})
        temp = current.get("temperature_2m")
        humidity = current.get("relative_humidity_2m")
        wind = current.get("wind_speed_10m")
        code = current.get("weather_code")
        condition = WEATHER_CODES.get(code, "fair conditions")

        if temp is not None:
            details = [f"It's currently {temp} degrees Celsius in {label} with {condition}."]
            if humidity is not None:
                details.append(f"Humidity is {humidity} percent")
            if wind is not None:
                details.append(f"wind speed is {wind} kilometers per hour")
            speak(" and ".join(details) + ".")
        else:
            speak(f"Weather conditions in {label} are currently {condition}.")
    except Exception as e:
        print("Weather error:", e)
        speak("Sorry, I couldn't fetch the weather right now.")


# ---------------------------------------------------------------------------
# Notes & Reminders (Sanitized & Confirmed)
# ---------------------------------------------------------------------------

def save_note(text):
    if not text:
        text = prompt_user("What would you like me to remember?")
        if not text:
            speak("I didn't catch that, so I didn't save anything.")
            return

    sanitized = " ".join(text.split()).strip()
    if not sanitized:
        speak("Note was empty.")
        return

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    try:
        with open(NOTES_FILE, "a", encoding="utf-8") as f:
            f.write(f"[{timestamp}] {sanitized}\n")
        speak("Got it, I saved that.")
    except Exception as err:
        print("Save note error:", err)
        speak("Could not write note to file.")


def read_notes():
    if not os.path.exists(NOTES_FILE):
        speak("You don't have any saved notes yet.")
        return

    try:
        with open(NOTES_FILE, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        if not lines:
            speak("You don't have any saved notes yet.")
            return

        speak(f"You have {len(lines)} saved note{'s' if len(lines) != 1 else ''}. Here are the most recent ones.")
        for line in lines[-5:]:
            content = line.split("] ", 1)[-1]
            speak(content)
    except Exception as err:
        print("Read notes error:", err)
        speak("Could not read notes.")


def clear_notes():
    if not os.path.exists(NOTES_FILE):
        speak("You don't have any notes to clear.")
        return

    reply = prompt_user("Are you sure you want to delete all your saved notes? Say or type yes to confirm.") or ""
    if reply.strip().lower() != "yes":
        speak("Notes kept safe.")
        return

    try:
        shutil.copyfile(NOTES_FILE, NOTES_BACKUP)
        os.remove(NOTES_FILE)
        speak("All notes have been cleared. A backup copy was saved.")
    except Exception as err:
        print("Clear notes error:", err)
        speak("Could not clear notes.")


# ---------------------------------------------------------------------------
# Unique Feature 7: Persistent Scheduled Reminders (survive restarts)
# ---------------------------------------------------------------------------

def _load_json(path, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, type(default)) else default
    except Exception:
        return default


def _save_json(path, data):
    """Atomically save assistant state so interruption cannot truncate it."""
    temp_path = None
    try:
        fd, temp_path = tempfile.mkstemp(prefix=".jarvis-", suffix=".tmp", dir=os.path.dirname(path))
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp_path, path)
        return True
    except Exception as err:
        print("Save JSON error:", err)
        if temp_path:
            try:
                os.remove(temp_path)
            except OSError:
                pass
        return False


def parse_reminder_time(text):
    """Parse phrases like 'at 5pm', 'at 17:30', 'in 10 minutes', 'in 2 hours'."""
    now = datetime.now()
    text = text.strip().lower()

    m = re.search(r"in\s+(\d+)\s*(minute|hour)s?", text)
    if m:
        amount = int(m.group(1))
        unit = m.group(2)
        delta_seconds = amount * (3600 if unit == "hour" else 60)
        return now + timedelta(seconds=delta_seconds)

    m2 = re.search(r"at\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", text)
    if m2:
        hour = int(m2.group(1))
        minute = int(m2.group(2)) if m2.group(2) else 0
        ampm = m2.group(3)
        if ampm == "pm" and hour != 12:
            hour += 12
        elif ampm == "am" and hour == 12:
            hour = 0
        if not (0 <= hour <= 23 and 0 <= minute <= 59):
            return None
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        return target

    return None


def add_reminder(command):
    """Handles: 'remind me to <task> at <time>' / 'remind me to <task> in <duration>'."""
    m = re.search(r"remind me to (.+?)\s+(at|in)\s+(.+)", command, re.IGNORECASE)
    if not m:
        speak("Please say it like: remind me to call mom at 5 PM, or remind me to take a break in 20 minutes.")
        return False

    task = m.group(1).strip()
    if not task or len(task) > 500:
        speak("Reminder text must be between 1 and 500 characters.")
        return False
    time_phrase = f"{m.group(2)} {m.group(3)}".strip()
    when = parse_reminder_time(time_phrase)
    if not when:
        speak("I couldn't understand that time. Try 'at 5pm' or 'in 20 minutes'.")
        return False

    with _reminders_lock:
        reminders = _load_json(REMINDERS_FILE, [])
        if len(reminders) >= 1000:
            speak("You have reached the limit of 1000 saved reminders.")
            return False
        reminders.append({"task": task, "time": when.isoformat(), "done": False})
        if not _save_json(REMINDERS_FILE, reminders):
            speak("I couldn't save that reminder.")
            return False

    speak(f"Okay, I'll remind you to {task} at {when.strftime('%I:%M %p on %B %d').lstrip('0')}.")
    return True


def list_reminders():
    reminders = _load_json(REMINDERS_FILE, [])
    pending = [r for r in reminders if not r.get("done")]
    if not pending:
        speak("You have no upcoming reminders.")
        return
    speak(f"You have {len(pending)} upcoming reminder{'s' if len(pending) != 1 else ''}.")
    for r in pending[:5]:
        when = datetime.fromisoformat(r["time"])
        speak(f"{r['task']} at {when.strftime('%I:%M %p').lstrip('0')}")


def reminder_checker_loop():
    """Background daemon thread: checks reminders periodically and announces due ones."""
    while not EXIT_REQUESTED:
        try:
            with _reminders_lock:
                reminders = _load_json(REMINDERS_FILE, [])
                now = datetime.now()
                changed = False
                for r in reminders:
                    if not r.get("done"):
                        when = datetime.fromisoformat(r["time"])
                        if now >= when:
                            speak(f"Reminder: {r['task']}")
                            r["done"] = True
                            changed = True
                if changed:
                    _save_json(REMINDERS_FILE, reminders)
        except Exception as err:
            print("Reminder checker error:", err)
        time.sleep(20)


# ---------------------------------------------------------------------------
# Unique Feature 8: Custom Voice Routines / Macros
# ---------------------------------------------------------------------------

def create_routine(command):
    """Handles: 'create routine <name>: <step1>, <step2>, ...'"""
    m = re.search(r"create routine (\w+)\s*:\s*(.+)", command, re.IGNORECASE)
    if not m:
        speak("Say it like: create routine morning: what's the time, weather in Kathmandu, news.")
        return False

    name = m.group(1).strip().lower()
    steps = [s.strip() for s in m.group(2).strip().split(",") if s.strip()]
    if not steps or len(steps) > 20 or any(len(step) > 300 for step in steps):
        speak("A routine must have 1 to 20 steps, each no longer than 300 characters.")
        return False

    routines = _load_json(ROUTINES_FILE, {})
    routines[name] = steps
    if not _save_json(ROUTINES_FILE, routines):
        speak("I couldn't save that routine.")
        return False
    speak(f"Routine '{name}' saved with {len(steps)} steps.")
    return True


_active_routines = set()


def run_routine(name):
    routines = _load_json(ROUTINES_FILE, {})
    name = name.strip().lower()
    steps = routines.get(name)
    if not steps:
        speak(f"I don't have a routine called {name}.")
        return False

    if name in _active_routines:
        speak(f"Routine '{name}' cannot run because it creates a cycle.")
        return False

    _active_routines.add(name)
    try:
        speak(f"Running routine {name}.")
        for step in steps:
            process_command(step)
    finally:
        _active_routines.remove(name)
    return True


def list_routines():
    routines = _load_json(ROUTINES_FILE, {})
    if not routines:
        speak("You don't have any saved routines.")
        return
    speak(f"You have {len(routines)} routines: {', '.join(routines.keys())}.")


# ---------------------------------------------------------------------------
# Social & WhatsApp Messaging
# ---------------------------------------------------------------------------

def send_whatsapp_message(command):
    """Open WhatsApp Web with recipient and message safely validated."""
    remainder = command
    for marker in ("send whatsapp message to", "send a whatsapp message to"):
        if marker in remainder.lower():
            idx = remainder.lower().find(marker) + len(marker)
            remainder = remainder[idx:].strip()
            break

    separator = " saying " if " saying " in remainder else " message "
    if separator not in remainder:
        speak("Say, send WhatsApp message to a phone number saying your message.")
        return

    phone, message = (part.strip() for part in remainder.split(separator, 1))
    clean_phone = re.sub(r"[^\d\+]", "", phone).lstrip("+")
    if not clean_phone or not clean_phone.isdigit() or not 7 <= len(clean_phone) <= 15 or not message or len(message) > 2000:
        speak("I need a valid phone number and a message.")
        return

    webbrowser.open(
        f"https://web.whatsapp.com/send?phone={quote_plus(clean_phone)}&text={quote_plus(message)}"
    )
    speak("WhatsApp is open with your message ready.")


# ---------------------------------------------------------------------------
# News Headlines
# ---------------------------------------------------------------------------

def read_news():
    if not NEWS_API_KEY:
        speak("I don't have a news API key set up, so I can't fetch headlines right now.")
        return
    try:
        r = requests.get(
            "https://newsapi.org/v2/top-headlines",
            params={"country": "us", "apiKey": NEWS_API_KEY},
            timeout=5
        )
        if r.status_code != 200:
            speak("Sorry, I couldn't reach the news service.")
            return
        articles = r.json().get("articles", [])[:5]
        if not articles:
            speak("I couldn't find any headlines right now.")
            return
        speak("Here are the top headlines.")
        for article in articles:
            speak(article.get("title", ""))
    except Exception as e:
        print("News error:", e)
        speak("Sorry, something went wrong fetching the news.")


# ---------------------------------------------------------------------------
# Unique Feature 9: Context-Aware AI Fallback
# ---------------------------------------------------------------------------

def get_ai_response(prompt):
    """Query OpenAI (with rolling conversational memory) if configured;
    otherwise fall back to Wikipedia for informative answers."""
    global CONVO_HISTORY

    if is_openai_configured():
        CONVO_HISTORY.append({"role": "user", "content": prompt})
        CONVO_HISTORY[:] = CONVO_HISTORY[-CONVO_HISTORY_MAX_TURNS * 2:]
        try:
            response = ask_openai_with_context(CONVO_HISTORY)
        except NameError:
            response = ask_openai(prompt)
        CONVO_HISTORY.append({"role": "assistant", "content": response})
        CONVO_HISTORY[:] = CONVO_HISTORY[-CONVO_HISTORY_MAX_TURNS * 2:]
        return response

    if wikipedia:
        search_res = wikipedia.search(prompt)
        if search_res:
            try:
                return wikipedia.summary(search_res[0], sentences=2, auto_suggest=False)
            except Exception:
                pass

    return "I don't have an OpenAI API key set up, and I couldn't find a direct answer. Set OPENAI_API_KEY in your .env file."


# ---------------------------------------------------------------------------
# Command Router (Intent Parsing)
# ---------------------------------------------------------------------------

def process_command(raw_command):
    """Route voice/text input to the correct feature handler."""
    global EXIT_REQUESTED
    if not raw_command:
        return

    raw_command = raw_command.strip()
    command = raw_command.lower()
    words = command.split()

    # --- Exit commands ---
    if command in {"exit jarvis", "quit jarvis", "close jarvis", "shutdown jarvis", "exit", "quit"}:
        EXIT_REQUESTED = True
        speak("Goodbye sir. Have a wonderful day.")
        return

    # --- Volume ---
    if any(phrase in command for phrase in ("mute audio", "mute sound", "mute")):
        control_volume("mute")
        return
    elif any(phrase in command for phrase in ("unmute audio", "unmute sound", "unmute")):
        control_volume("unmute")
        return
    elif any(phrase in command for phrase in ("volume up", "increase volume", "louder")):
        control_volume("up")
        return
    elif any(phrase in command for phrase in ("volume down", "decrease volume", "lower volume", "quieter")):
        control_volume("down")
        return

    # --- Reminders (must come before generic "timer"/"remember" checks) ---
    if command.startswith("remind me to"):
        add_reminder(raw_command)
        return
    elif "list reminders" in command or "my reminders" in command or "upcoming reminders" in command:
        list_reminders()
        return

    # --- Routines ---
    if command.startswith("create routine"):
        create_routine(raw_command)
        return
    elif command.startswith("run routine"):
        name = command.replace("run routine", "", 1).strip()
        run_routine(name)
        return
    elif "list routines" in command or "my routines" in command:
        list_routines()
        return

    # --- Timers ---
    if "timer" in command:
        parse_and_set_timer(command)
        return

    # --- Unit Conversion ---
    if any(k in command for k in ("celsius", "fahrenheit", "kilometers", "miles", "pounds", "kilograms")):
        if convert_units(command):
            return

    # --- Calculator (AST Safe Math) ---
    if (
        command.startswith("calculate")
        or command.startswith("what is")
        and any(op in command for op in ("plus", "minus", "times", "divided", "*", "/", "+", "%", "power"))
        or "math" in command
    ):
        calculate_expression(command)
        return

    # --- System Diagnostics & Status ---
    if any(k in command for k in ("system diagnostics", "diagnostics", "disk usage", "storage", "top process", "top processes")):
        system_diagnostics()
        return
    elif any(k in command for k in ("ping test", "check internet", "test connection", "network latency")):
        ping_network()
        return
    elif "system status" in command or "how is my computer" in command or "battery" in command:
        system_status()
        return

    # --- Desktop App Launcher ---
    if command.startswith("open app") or (command.startswith("open ") and any(app in command for app in APP_COMMANDS)):
        target_app = command.replace("open app", "").replace("open", "").strip()
        if launch_application(target_app):
            return

    # --- Wikipedia Knowledge Lookup ---
    if any(command.startswith(p) for p in ("tell me about", "who is", "who was", "define", "explain", "wikipedia")):
        if ask_wikipedia(command):
            return

    # --- Conversational / Questions ---
    if "i have some question" in command or command in ("hey", "hi", "hello", "jarvis"):
        question = prompt_user("Sure, what would you like to ask?")
        if question:
            speak(get_ai_response(question))
        return

    # --- Time & Date ---
    if re.search(r"\b(time|what time|current time)\b", command):
        tell_time()
        return
    elif re.search(r"\b(date|today's date|current date)\b", command):
        tell_date()
        return

    # --- Weather ---
    if "weather" in command:
        city = None
        m = re.search(r"weather\s+(?:in|for|at|like in)\s+([a-zA-Z\s]+)", command)
        if m:
            city = m.group(1).strip().title()
        else:
            m2 = re.search(r"([a-zA-Z\s]+)\s+weather", command)
            if m2 and m2.group(1).strip() not in ("what", "how", "the", "current"):
                city = m2.group(1).strip().title()
        get_weather(city)
        return

    # --- Power Controls ---
    if "cancel shutdown" in command:
        cancel_shutdown()
        return
    elif "shut down" in command or "shutdown" in command:
        shutdown_computer()
        return
    elif "restart" in command or "reboot" in command:
        restart_computer()
        return
    elif "sleep" in words and "computer" in words:
        sleep_computer()
        return

    # --- Notes ---
    if command.startswith("remember that") or command.startswith("remember this"):
        note = command.split("remember that", 1)[-1].split("remember this", 1)[-1].strip()
        save_note(note)
        return
    elif "take a note" in command or "save a note" in command:
        save_note(None)
        return
    elif "read my notes" in command or "what did i ask you to remember" in command:
        read_notes()
        return
    elif "clear my notes" in command or "delete my notes" in command:
        clear_notes()
        return

    # --- News ---
    if "news" in words or "headlines" in words:
        read_news()
        return

    # --- Google ---
    if command.startswith("open google") or "search google for" in command or "google search for" in command:
        if "search google for" in command:
            query = command.split("search google for", 1)[1].strip()
        elif "google search for" in command:
            query = command.split("google search for", 1)[1].strip()
        else:
            query = command.replace("open google", "", 1).strip()

        if query:
            webbrowser.open(f"https://www.google.com/search?q={quote_plus(query)}")
            speak(f"Searching Google for {query}")
        else:
            webbrowser.open("https://www.google.com")
            speak("Opening Google.")
        return
    elif "close google" in command or "close browser" in command:
        close_browser()
        return

    # --- YouTube ---
    if command.startswith("search youtube for") or command.startswith("youtube search for"):
        query = command.replace("search youtube for", "", 1).replace("youtube search for", "", 1).strip()
        if query:
            speak(f"Playing {query} on YouTube")
            if pywhatkit:
                pywhatkit.playonyt(query)
            else:
                webbrowser.open(f"https://www.youtube.com/results?search_query={quote_plus(query)}")
        else:
            webbrowser.open("https://www.youtube.com")
            speak("Opening YouTube.")
        return
    elif "open youtube" in command:
        webbrowser.open("https://www.youtube.com")
        speak("Opening YouTube.")
        return
    elif "close youtube" in command:
        close_browser()
        return

    # --- Social Apps ---
    if "open facebook" in command:
        webbrowser.open("https://www.facebook.com")
        speak("Opening Facebook.")
        return
    elif "close facebook" in command:
        close_browser()
        return
    elif "open instagram" in command:
        webbrowser.open("https://www.instagram.com")
        speak("Opening Instagram.")
        return
    elif "close instagram" in command:
        close_browser()
        return
    elif "open whatsapp" in command:
        webbrowser.open("https://web.whatsapp.com")
        speak("Opening WhatsApp.")
        return
    elif "close whatsapp" in command:
        close_browser()
        return
    elif command.startswith("send whatsapp message to") or command.startswith("send a whatsapp message to"):
        send_whatsapp_message(raw_command)
        return
    elif "open linkedin" in command:
        webbrowser.open("https://www.linkedin.com")
        speak("Opening LinkedIn.")
        return

    # --- Music Playback ---
    if command.startswith("play"):
        try:
            song = re.sub(r"^play\s+(?:song\s+)?", "", command).strip(" .!?")
            if not song:
                song = prompt_user("What song would you like to play?") or ""
                song = song.strip(" .!?")

            if song in music_library.music:
                webbrowser.open(music_library.music[song])
                speak(f"Playing {song}")
            elif song:
                speak(f"Playing {song} on YouTube")
                if pywhatkit:
                    pywhatkit.playonyt(song)
                else:
                    webbrowser.open(f"https://www.youtube.com/results?search_query={quote_plus(song)}")
            else:
                speak("I didn't catch a song name.")
        except Exception as e:
            print("Music error:", e)
            speak("Something went wrong while playing the song.")
        return

    # --- Fallback: AI or Knowledge Engine ---
    speak(get_ai_response(raw_command))


# ---------------------------------------------------------------------------
# Interactive Shell & Execution Loops
# ---------------------------------------------------------------------------

def run_cli_loop():
    """Interactive conversational terminal shell."""
    global EXIT_REQUESTED
    print("\n" + "=" * 60)
    print("      JARVIS A.I. ASSISTANT (Interactive Shell)")
    print("=" * 60)
    print("Type your commands below. Type 'help' for examples or 'exit' to quit.")
    print("=" * 60 + "\n")
    speak("Yes sir, Jarvis is online and ready for your command.")

    while not EXIT_REQUESTED:
        try:
            cmd = input("\nJarvis > ").strip()
            if not cmd:
                continue

            if cmd.lower() in ("help", "?"):
                print("\nAvailable Features:")
                print("  • Time/Date: 'what's the time', 'what is today's date'")
                print("  • Weather: 'weather in London', 'Tokyo weather'")
                print("  • Math (AST safe): 'calculate 45 * 12', 'what is 15 percent of 200'")
                print("  • Units: 'convert 100 celsius to fahrenheit', 'convert 5 miles to km'")
                print("  • Wikipedia: 'who was Isaac Newton', 'define quantum computing'")
                print("  • Diagnostics: 'system diagnostics', 'system status', 'ping test'")
                print("  • Desktop Apps: 'open notepad', 'open calculator', 'open terminal'")
                print("  • Timers: 'set a timer for 10 seconds'")
                print("  • Reminders: 'remind me to call mom at 5pm', 'list reminders'")
                print("  • Routines: 'create routine morning: what's the time, weather in Kathmandu, news'")
                print("              'run routine morning', 'list routines'")
                print("  • Audio: 'mute', 'volume up', 'volume down'")
                print("  • Notes: 'remember that my password is safe', 'read my notes'")
                print("  • Web/Media: 'search google for cats', 'play believer'")
                print("  • Power: 'shut down', 'restart', 'sleep computer'")
                print("  • Exit: 'exit', 'quit'")
                continue

            process_command(cmd)
        except (KeyboardInterrupt, EOFError):
            speak("Goodbye sir.")
            break
        except Exception as err:
            print("Execution error:", err)
            speak("An error occurred while processing that command.")


def run_voice_loop():
    """Voice-driven loop with wake-word detection and robust fallback."""
    global EXIT_REQUESTED
    speak("Yes sir, Jarvis voice system is online. How can I help you.")

    while not EXIT_REQUESTED:
        try:
            print("\nWaiting for wake word ('Jarvis')...")
            wake_text = None
            while not wake_text or "jarvis" not in wake_text.lower():
                wake_text = listen_and_transcribe(WAKE_WORD_DURATION)
                if wake_text:
                    print(f"Heard: {wake_text}")
                if EXIT_REQUESTED:
                    break

            if EXIT_REQUESTED:
                break

            speak("Yes Sir, I'm listening.")
            active = True
            while active and not EXIT_REQUESTED:
                print("Listening for command...")
                command = listen_and_transcribe(COMMAND_DURATION)
                if command:
                    print(f"Command heard: {command}")
                    cmd_lower = command.lower()
                    if any(phrase in cmd_lower for phrase in ("stop jarvis", "go to sleep")):
                        speak("Going back to sleep. Say Jarvis to wake me up again.")
                        active = False
                    else:
                        try:
                            process_command(command)
                        except Exception as e:
                            print("Command error:", e)
                            speak("Sorry, something went wrong with that command.")

        except KeyboardInterrupt:
            speak("Goodbye, have a great day.")
            break
        except Exception as e:
            print("System error:", e)

    print("Jarvis stopped.")


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------

def main():
    global CLI_MODE, QUIET_MODE, TTS_BACKEND

    parser = argparse.ArgumentParser(description="Jarvis AI Virtual Assistant")
    parser.add_argument("--cli", "--text", action="store_true", help="Force interactive CLI/text mode")
    parser.add_argument("--voice", action="store_true", help="Force voice recognition mode")
    parser.add_argument("--quiet", action="store_true", help="Mute text-to-speech engine")
    parser.add_argument(
        "--tts-backend", choices=["pyttsx3", "gtts"],
        default=os.getenv("JARVIS_TTS_BACKEND", "pyttsx3"),
        help="Choose offline (pyttsx3) or online (gTTS) text-to-speech backend"
    )
    args = parser.parse_args()

    QUIET_MODE = args.quiet
    TTS_BACKEND = args.tts_backend

    if args.cli:
        CLI_MODE = True
    elif args.voice:
        CLI_MODE = False
    else:
        mic_available = has_working_microphone()
        if not mic_available:
            print("[Notice] No audio input device detected. Defaulting to Interactive Shell.")
            CLI_MODE = True
        else:
            CLI_MODE = False

    # Background reminder checker runs regardless of CLI/voice mode.
    threading.Thread(target=reminder_checker_loop, daemon=True).start()

    if CLI_MODE:
        run_cli_loop()
    else:
        run_voice_loop()


if __name__ == "__main__":
    main()
