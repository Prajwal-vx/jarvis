"""
Jarvis Voice Assistant (combined + extended)
=============================================

This merges the two Jarvis scripts you had into one file and adds:
  - Time / date announcements
  - Weather lookup (no API key needed - uses Open-Meteo)
  - Shutdown / restart / sleep the computer (with spoken confirmation, safe)
  - System status: battery + CPU + RAM ("unique" feature #1)
  - Voice notes/reminders you can save and recall later ("unique" feature #2)
  - News headlines (optional, needs a NewsAPI key)
  - Everything from before: wake word, Google/YouTube/Facebook/Instagram/
    WhatsApp, music playback, and OpenAI fallback for open-ended questions

--------------------------------------------------------------------------
SETUP
--------------------------------------------------------------------------
1) Install dependencies:
   python -m pip install SpeechRecognition sounddevice numpy pyttsx3 openai
   requests pywhatkit psutil

2) Set your OpenAI key as an environment variable (do NOT hardcode it):
   Windows (PowerShell):  $env:OPENAI_API_KEY="sk-..."
   macOS/Linux (bash):    export OPENAI_API_KEY="sk-..."

3) (Optional) For news headlines, set NEWS_API_KEY the same way.

4) (Optional) Create a local `musiclibrary.py` next to this file with a
   `music = {"song name": "https://..."}` dict for direct-link playback.
   If it's missing, Jarvis just falls back to searching/playing on YouTube.

5) Run it, then say "Jarvis" to wake it up, wait for the "I'm listening"
   response, then give your command.

--------------------------------------------------------------------------
SAFETY NOTE ON SHUTDOWN
--------------------------------------------------------------------------
Shutdown/restart is a real, irreversible OS action, so Jarvis always asks
for a spoken "yes" before doing it, and gives you a few seconds + a spoken
"cancel shutdown" window to abort. Nothing destructive happens silently.
"""

import os
import platform
import subprocess
import webbrowser
from datetime import datetime
from urllib.parse import quote_plus

import numpy as np
import requests
import sounddevice as sd

try:
    import speech_recognition as sr  # pyright: ignore[reportMissingImports]
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "The 'SpeechRecognition' package is required. Install it with: "
        "python -m pip install SpeechRecognition"
    ) from exc

import pyttsx3
import pywhatkit

try:
    import psutil  # for battery / CPU / RAM status
except ModuleNotFoundError:
    psutil = None  # system-status command will just explain it's missing

try:
    import openai
except ModuleNotFoundError:
    openai = None

try:
    import importlib
    music_library = importlib.import_module("musiclibrary")
except ModuleNotFoundError:
    class _MusicLibrary:
        music = {}
    music_library = _MusicLibrary()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

engine = pyttsx3.init()
recognizer = sr.Recognizer()

SAMPLE_RATE = 16000
WAKE_WORD_DURATION = 3
COMMAND_DURATION = 4
QUESTION_DURATION = 7
MIC_DEVICE_INDEX = None

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ai_client = openai.OpenAI(api_key=OPENAI_API_KEY) if (openai and OPENAI_API_KEY) else None
AI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

NEWS_API_KEY = os.getenv("NEWS_API_KEY")

NOTES_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "jarvis_notes.txt")
EXIT_REQUESTED = False


# ---------------------------------------------------------------------------
# Core I/O helpers
# ---------------------------------------------------------------------------

def speak(text):
    print("Jarvis:", text)
    engine.say(text)
    engine.runAndWait()


def record_audio(duration, fs=SAMPLE_RATE):
    """Record audio with sounddevice (no PyAudio needed) and wrap it for
    speech_recognition."""
    audio = sd.rec(
        int(duration * fs),
        samplerate=fs,
        channels=1,
        dtype='int16',
        device=MIC_DEVICE_INDEX
    )
    sd.wait()
    return sr.AudioData(audio.tobytes(), fs, 2)


def listen_and_transcribe(duration):
    try:
        audio_data = record_audio(duration)
    except (OSError, sd.PortAudioError) as exc:
        print("Microphone error:", exc)
        speak("I can't access the microphone. Check the microphone and try again.")
        return None
    try:
        return recognizer.recognize_google(audio_data)
    except sr.UnknownValueError:
        return None
    except sr.RequestError:
        speak("Speech recognition service is unavailable.")
        return None


def list_microphones():
    """Run once to find the correct MIC_DEVICE_INDEX if audio isn't picked up."""
    print(sd.query_devices())


def close_browser():
    """Best-effort close of the default browser window (cross-platform)."""
    system = platform.system()
    try:
        if system == "Windows":
            for exe in ("brave.exe", "chrome.exe", "msedge.exe", "firefox.exe"):
                subprocess.run(["taskkill", "/F", "/IM", exe], capture_output=True, text=True)
        elif system == "Darwin":
            subprocess.run(["osascript", "-e", 'quit app "Google Chrome"'], capture_output=True, text=True)
        else:
            subprocess.run(["pkill", "-f", "chrome"], capture_output=True, text=True)
        return True
    except Exception as e:
        print("Close browser error:", e)
        return False


# ---------------------------------------------------------------------------
# AI fallback
# ---------------------------------------------------------------------------

def get_ai_response(prompt):
    if not ai_client:
        return "I don't have an API key set up, so I can't think of an answer right now."
    try:
        response = ai_client.chat.completions.create(
            model=AI_MODEL,
            messages=[
                {"role": "system", "content": (
                    "You are a virtual assistant named Jarvis. Keep answers short "
                    "and conversational, since they will be spoken out loud."
                )},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print("OpenAI error:", e)
        return "Sorry, I had trouble getting an answer for that."


# ---------------------------------------------------------------------------
# New feature: time & date
# ---------------------------------------------------------------------------

def tell_time():
    now = datetime.now().strftime("%I:%M %p").lstrip("0")
    speak(f"It's {now} right now.")


def tell_date():
    today = datetime.now().strftime("%A, %B %d, %Y")
    speak(f"Today is {today}.")


# ---------------------------------------------------------------------------
# New feature: weather (Open-Meteo, no API key required)
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
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1}, timeout=5
        ).json()
        results = geo.get("results")
        if not results:
            speak(f"I couldn't find a place called {city}.")
            return

        place = results[0]
        lat, lon = place["latitude"], place["longitude"]
        label = place.get("name", city)

        forecast = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat, "longitude": lon,
                "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m",
            }, timeout=5
        ).json()

        current = forecast.get("current", {})
        temp = current.get("temperature_2m")
        humidity = current.get("relative_humidity_2m")
        wind = current.get("wind_speed_10m")
        code = current.get("weather_code")
        condition = WEATHER_CODES.get(code, "unusual conditions")

        speak(
            f"It's currently {temp} degrees Celsius in {label} with {condition}. "
            f"Humidity is {humidity} percent and wind speed is {wind} kilometers per hour."
        )
    except Exception as e:
        print("Weather error:", e)
        speak("Sorry, I couldn't fetch the weather right now.")


# ---------------------------------------------------------------------------
# New feature: shutdown / restart / sleep (with confirmation, cross-platform)
# ---------------------------------------------------------------------------

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
    speak("Are you sure you want me to shut down the computer? Say yes to confirm.")
    reply = listen_and_transcribe(COMMAND_DURATION) or ""
    if "yes" not in reply.lower():
        speak("Okay, cancelling shutdown.")
        return

    speak("Shutting down in one minute. Say 'cancel shutdown' now if you change your mind.")
    reply2 = listen_and_transcribe(COMMAND_DURATION) or ""
    if "cancel" in reply2.lower():
        _run_power_command({
            "Windows": ["shutdown", "/a"],
            "Darwin": None,
            "Linux": ["shutdown", "-c"],
        }, "cancel the shutdown")
        speak("Shutdown cancelled.")
        return

    _run_power_command({
        "Windows": ["shutdown", "/s", "/t", "60"],
        "Darwin": ["sudo", "shutdown", "-h", "+1"],
        "Linux": ["shutdown", "+1"],
    }, "shut down")


def restart_computer():
    speak("Are you sure you want me to restart the computer? Say yes to confirm.")
    reply = listen_and_transcribe(COMMAND_DURATION) or ""
    if "yes" not in reply.lower():
        speak("Okay, cancelling restart.")
        return
    _run_power_command({
        "Windows": ["shutdown", "/r", "/t", "10"],
        "Darwin": ["sudo", "shutdown", "-r", "+1"],
        "Linux": ["shutdown", "-r", "+1"],
    }, "restart")
    speak("Restarting the computer now.")


def sleep_computer():
    _run_power_command({
        "Windows": ["rundll32.exe", "powrprof.dll,SetSuspendState", "0,1,0"],
        "Darwin": ["pmset", "sleepnow"],
        "Linux": ["systemctl", "suspend"],
    }, "sleep")
    speak("Putting the computer to sleep.")


# ---------------------------------------------------------------------------
# Unique feature #1: system status (battery / CPU / RAM)
# ---------------------------------------------------------------------------

def system_status():
    if not psutil:
        speak("I need the psutil package installed to check system status. "
              "Install it with pip install psutil.")
        return

    cpu = psutil.cpu_percent(interval=1)
    mem = psutil.virtual_memory()
    parts = [f"CPU usage is at {cpu} percent", f"memory usage is at {mem.percent} percent"]

    battery = psutil.sensors_battery()
    if battery:
        state = "charging" if battery.power_plugged else "not charging"
        parts.append(f"battery is at {round(battery.percent)} percent and {state}")

    speak(", ".join(parts) + ".")


# ---------------------------------------------------------------------------
# Unique feature #2: voice notes / reminders you can save and recall
# ---------------------------------------------------------------------------

def save_note(text):
    if not text:
        speak("What would you like me to remember?")
        text = listen_and_transcribe(QUESTION_DURATION)
        if not text:
            speak("I didn't catch that, so I didn't save anything.")
            return

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    with open(NOTES_FILE, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {text}\n")
    speak("Got it, I saved that.")


def read_notes():
    if not os.path.exists(NOTES_FILE):
        speak("You don't have any saved notes yet.")
        return

    with open(NOTES_FILE, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]

    if not lines:
        speak("You don't have any saved notes yet.")
        return

    speak(f"You have {len(lines)} saved note{'s' if len(lines) != 1 else ''}. Here are the most recent ones.")
    for line in lines[-5:]:
        # Strip the leading "[timestamp] " for a more natural read-out.
        content = line.split("] ", 1)[-1]
        speak(content)


def clear_notes():
    if os.path.exists(NOTES_FILE):
        os.remove(NOTES_FILE)
    speak("All your notes have been cleared.")


# ---------------------------------------------------------------------------
# Social messaging
# ---------------------------------------------------------------------------

def send_whatsapp_message(command):
    """Open WhatsApp Web with a recipient and message prefilled."""
    remainder = command
    for marker in ("send whatsapp message to", "send a whatsapp message to"):
        if marker in remainder:
            remainder = remainder.split(marker, 1)[1].strip()
            break

    separator = " saying " if " saying " in remainder else " message "
    if separator not in remainder:
        speak("Say, send WhatsApp message to a phone number saying your message.")
        return

    phone, message = (part.strip() for part in remainder.split(separator, 1))
    phone = phone.replace(" ", "").lstrip("+")
    if not phone or not phone.lstrip("+").isdigit() or not message:
        speak("I need an international phone number and a message.")
        return

    webbrowser.open(
        f"https://web.whatsapp.com/send?phone={quote_plus(phone)}&text={quote_plus(message)}"
    )
    speak("WhatsApp is open with your message ready. Review it and press send.")


# ---------------------------------------------------------------------------
# Optional: news headlines (needs NEWS_API_KEY)
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
# Command router
# ---------------------------------------------------------------------------

def process_command(command):
    global EXIT_REQUESTED
    raw_command = command.strip()
    command = raw_command.lower()
    words = command.split()

    if command in {"exit jarvis", "quit jarvis", "close jarvis", "shutdown jarvis"}:
        EXIT_REQUESTED = True
        speak("Goodbye.")
        return

    # --- Open-ended question mode ---
    if "i have some question" in command or "hey" in words or "hi" in words:
        speak("Sure, what would you like to ask?")
        question = listen_and_transcribe(QUESTION_DURATION)
        if question:
            speak(get_ai_response(question))
        else:
            speak("I didn't catch that. Please try again.")

    # --- Time / date ---
    elif "time" in words and "what" in words:
        tell_time()
    elif "date" in words or "today's date" in command:
        tell_date()

    # --- Weather ---
    elif "weather" in command:
        city = None
        for marker in ("weather in", "weather for", "weather at"):
            if marker in command:
                city = command.split(marker, 1)[1].strip()
                break
        if not city:
            speak("Which city's weather would you like?")
            city = listen_and_transcribe(COMMAND_DURATION)
        if city:
            get_weather(city)
        else:
            speak("I didn't catch a city name.")

    # --- Power controls ---
    elif "shut down" in command or "shutdown" in command:
        shutdown_computer()
    elif "restart" in command or "reboot" in command:
        restart_computer()
    elif "sleep" in words and "computer" in words:
        sleep_computer()

    # --- System status (unique feature) ---
    elif "system status" in command or "how is my computer" in command or "battery" in command:
        system_status()

    # --- Notes / reminders (unique feature) ---
    elif command.startswith("remember that") or command.startswith("remember this"):
        note = command.split("remember that", 1)[-1].split("remember this", 1)[-1].strip()
        save_note(note)
    elif "take a note" in command or "save a note" in command:
        save_note(None)
    elif "read my notes" in command or "what did i ask you to remember" in command:
        read_notes()
    elif "clear my notes" in command or "delete my notes" in command:
        clear_notes()

    # --- News ---
    elif "news" in words:
        read_news()

    # --- Google ---
    elif command.startswith("open google") or "search google for" in command or "google search for" in command:
        if "search google for" in command:
            query = command.split("search google for", 1)[1].strip()
        elif "google search for" in command:
            query = command.split("google search for", 1)[1].strip()
        else:
            query = command.replace("open google", "", 1).strip()

        if query:
            webbrowser.open(f"https://www.google.com/search?q={quote_plus(query)}")
            speak(f"Searching Google for {query}")
            speak(get_ai_response(query))
        else:
            webbrowser.open("https://www.google.com")
            speak("Opening Google.")
    elif "close google" in command:
        close_browser()
        speak("Closing Google.")

    # --- YouTube ---
    elif command.startswith("search youtube for") or command.startswith("youtube search for"):
        query = command.replace("search youtube for", "", 1).replace("youtube search for", "", 1).strip()
        if query:
            speak(f"Playing {query} on YouTube")
            pywhatkit.playonyt(query)
        else:
            webbrowser.open("https://www.youtube.com")
            speak("Opening YouTube.")
    elif "open youtube" in command:
        webbrowser.open("https://www.youtube.com")
        speak("Opening YouTube.")
    elif "close youtube" in command:
        close_browser()
        speak("Closing YouTube.")

    # --- Social apps ---
    elif "open facebook" in command:
        webbrowser.open("https://www.facebook.com")
        speak("Opening Facebook.")
    elif "close facebook" in command:
        close_browser()
        speak("Closing Facebook.")
    elif "open instagram" in command:
        webbrowser.open("https://www.instagram.com")
        speak("Opening Instagram.")
    elif "close instagram" in command:
        close_browser()
        speak("Closing Instagram.")
    elif "open whatsapp" in command:
        webbrowser.open("https://web.whatsapp.com")
        speak("Opening WhatsApp.")
    elif "close whatsapp" in command:
        close_browser()
        speak("Closing WhatsApp.")
    elif command.startswith("send whatsapp message to") or command.startswith("send a whatsapp message to"):
        send_whatsapp_message(raw_command)
    elif "open linkedin" in command:
        webbrowser.open("https://www.linkedin.com")
        speak("Opening LinkedIn.")

    # --- Music ---
    elif command.startswith("play"):
        try:
            song = raw_command[4:].strip()
            if not song:
                speak("What song would you like to play?")
                song = listen_and_transcribe(QUESTION_DURATION) or ""
            if song in music_library.music:
                webbrowser.open(music_library.music[song])
                speak(f"Playing {song}")
            elif song:
                speak(f"Playing {song} on YouTube")
                pywhatkit.playonyt(song)
            else:
                speak("I didn't catch a song name.")
        except Exception as e:
            print(e)
            speak("Something went wrong while playing the song.")

    # --- Fallback: let the AI handle it ---
    else:
        speak(get_ai_response(command))


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    speak("Yes sir, how can I help you.")

    while not EXIT_REQUESTED:
        try:
            print("Waiting for wake word...")

            wake_text = None
            while not wake_text or "jarvis" not in wake_text.lower():
                wake_text = listen_and_transcribe(WAKE_WORD_DURATION)
                if wake_text:
                    print("You said:", wake_text)

            speak("Yes Sir. I'm listening.")

            active = True
            print("Listening for your command...")
            while active and not EXIT_REQUESTED:
                command = listen_and_transcribe(COMMAND_DURATION)

                if command:
                    print("Command:", command)
                    command_lower = command.lower()

                    if any(phrase in command_lower for phrase in ("stop jarvis", "go to sleep")):
                        speak("Going back to sleep. Say Jarvis to wake me up again.")
                        active = False
                    else:
                        speak(f"You said, {command}")
                        try:
                            process_command(command)
                        except Exception as e:
                            print("Command error:", e)
                            speak("Sorry, something went wrong with that command.")

                        if active and not EXIT_REQUESTED:
                            print("Listening for your command...")

        except KeyboardInterrupt:
            speak("Goodbye, have a great day.")
            break
        except Exception as e:
            print("Error:", e)

    print("Jarvis stopped.")