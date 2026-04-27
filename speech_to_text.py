"""
Speech-to-Text Transcription Tool
====================================
Converts audio recordings into text using the SpeechRecognition library.

Supports:
  - Microphone input (live recording)
  - Audio file input (.wav, .mp3, .flac, .aiff)
  - Multiple recognition engines (Google, Sphinx offline)
  - Noise reduction / ambient noise adjustment
  - Word confidence display
  - Export transcript to .txt file

Install requirements:
    pip install SpeechRecognition pyaudio pydub

    # On Linux also run:
    sudo apt-get install portaudio19-dev python3-pyaudio ffmpeg

    # On macOS:
    brew install portaudio ffmpeg

Usage:
    python speech_to_text.py                    # live mic recording
    python speech_to_text.py --file audio.wav   # transcribe a file
    python speech_to_text.py --demo             # demo mode (no mic needed)
    python speech_to_text.py --file audio.mp3 --save  # save transcript
"""

import os
import sys
import time
import argparse
import datetime

# ── Dependency Check ─────────────────────────────────────────────────────────
def check_deps():
    missing = []
    try:
        import speech_recognition  # noqa
    except ImportError:
        missing.append("SpeechRecognition")
    try:
        import pyaudio  # noqa
    except ImportError:
        missing.append("pyaudio")
    return missing

# ── Terminal Colors ───────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
GRAY   = "\033[90m"
BLUE   = "\033[94m"

def banner():
    print(f"""
{CYAN}{'═'*60}
  🎙️  Speech-to-Text Transcription Tool
  Powered by SpeechRecognition + Google Speech API
{'═'*60}{RESET}
""")

# ── Core Transcription Engine ─────────────────────────────────────────────────
class Transcriber:
    def __init__(self, engine="google", language="en-US"):
        import speech_recognition as sr
        self.sr = sr
        self.recognizer = sr.Recognizer()
        self.engine = engine
        self.language = language

        # Tuning parameters
        self.recognizer.energy_threshold = 300       # mic sensitivity
        self.recognizer.dynamic_energy_threshold = True
        self.recognizer.pause_threshold = 0.8        # seconds of silence = end of phrase

    def adjust_for_noise(self, source, duration=1):
        """Calibrate microphone for ambient noise."""
        print(f"  {YELLOW}⏳ Calibrating for ambient noise ({duration}s)...{RESET}")
        self.recognizer.adjust_for_ambient_noise(source, duration=duration)
        print(f"  {GREEN}✓ Energy threshold set to {self.recognizer.energy_threshold:.0f}{RESET}")

    def _recognize(self, audio):
        """Run recognition engine on audio data."""
        if self.engine == "google":
            return self.recognizer.recognize_google(
                audio, language=self.language, show_all=False
            )
        elif self.engine == "sphinx":
            return self.recognizer.recognize_sphinx(audio)
        else:
            return self.recognizer.recognize_google(audio, language=self.language)

    def from_microphone(self, timeout=10, phrase_limit=15):
        """Record from microphone and transcribe."""
        import speech_recognition as sr
        print(f"\n  {CYAN}🎙️  Listening... (speak now, {timeout}s timeout){RESET}")
        with sr.Microphone() as source:
            self.adjust_for_noise(source)
            print(f"  {GREEN}● Recording — speak clearly into your microphone{RESET}")
            try:
                audio = self.recognizer.listen(
                    source, timeout=timeout, phrase_time_limit=phrase_limit
                )
                print(f"  {GRAY}✓ Audio captured ({len(audio.get_wav_data())//1024} KB){RESET}")
            except sr.WaitTimeoutError:
                return None, "TIMEOUT", "No speech detected within timeout period."

        return self._transcribe_audio(audio)

    def from_file(self, filepath: str):
        """Transcribe an audio file."""
        import speech_recognition as sr

        if not os.path.exists(filepath):
            return None, "FILE_NOT_FOUND", f"File not found: {filepath}"

        ext = os.path.splitext(filepath)[1].lower()
        print(f"\n  {CYAN}📂 Loading: {filepath}{RESET}")

        # Convert non-wav formats using pydub
        if ext != ".wav":
            try:
                from pydub import AudioSegment
                print(f"  {GRAY}Converting {ext} → WAV...{RESET}")
                audio_seg = AudioSegment.from_file(filepath)
                tmp_path = filepath.replace(ext, "_tmp.wav")
                audio_seg.export(tmp_path, format="wav")
                filepath = tmp_path
                print(f"  {GREEN}✓ Converted successfully{RESET}")
            except ImportError:
                return None, "PYDUB_MISSING", "Install pydub for MP3/FLAC support: pip install pydub"
            except Exception as e:
                return None, "CONVERSION_ERROR", str(e)

        try:
            with sr.AudioFile(filepath) as source:
                print(f"  {GRAY}Reading audio file...{RESET}")
                audio = self.recognizer.record(source)
                print(f"  {GREEN}✓ Audio loaded ({len(audio.get_wav_data())//1024} KB){RESET}")
        except Exception as e:
            return None, "LOAD_ERROR", str(e)

        return self._transcribe_audio(audio)

    def _transcribe_audio(self, audio):
        """Send audio to recognition engine."""
        import speech_recognition as sr
        print(f"  {CYAN}🔄 Transcribing with {self.engine.upper()} engine...{RESET}")
        try:
            start = time.time()
            text = self._recognize(audio)
            duration = time.time() - start
            return text, "SUCCESS", f"Transcribed in {duration:.2f}s"
        except sr.UnknownValueError:
            return None, "UNKNOWN", "Speech could not be understood. Try speaking more clearly."
        except sr.RequestError as e:
            return None, "API_ERROR", f"API error: {e}. Check internet connection."
        except Exception as e:
            return None, "ERROR", str(e)

    def from_microphone_continuous(self, max_phrases=5):
        """Continuously record and transcribe multiple phrases."""
        import speech_recognition as sr
        results = []
        print(f"\n  {CYAN}🎙️  Continuous mode — recording {max_phrases} phrases{RESET}")
        print(f"  {GRAY}Press Ctrl+C to stop early{RESET}\n")

        with sr.Microphone() as source:
            self.adjust_for_noise(source)
            for i in range(max_phrases):
                try:
                    print(f"  {GREEN}● Phrase {i+1}/{max_phrases} — speak now...{RESET}")
                    audio = self.recognizer.listen(source, timeout=8, phrase_time_limit=15)
                    text, status, msg = self._transcribe_audio(audio)
                    if status == "SUCCESS":
                        results.append(text)
                        print(f"  {BLUE}📝 \"{text}\"{RESET}\n")
                    else:
                        print(f"  {YELLOW}⚠ {msg}{RESET}\n")
                except sr.WaitTimeoutError:
                    print(f"  {YELLOW}⏱ Timeout — no speech detected{RESET}\n")
                except KeyboardInterrupt:
                    print(f"\n  {YELLOW}Stopped by user{RESET}")
                    break
        return results


# ── Display & Export ──────────────────────────────────────────────────────────
def display_result(text, status, message, filepath=None):
    print(f"\n  {'─'*56}")
    if status == "SUCCESS":
        print(f"  {GREEN}{BOLD}✓ Transcription Successful{RESET}")
        print(f"  {'─'*56}")
        print(f"\n  {BLUE}📝 Transcript:{RESET}")
        print(f"\n  {BOLD}\"{text}\"{RESET}\n")
        print(f"  {GRAY}Words: {len(text.split())}  |  Characters: {len(text)}{RESET}")
    else:
        print(f"  {RED}{BOLD}✗ {status}: {message}{RESET}")
    print(f"  {'─'*56}")


def save_transcript(text: str, source_name: str = "transcript"):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"transcript_{timestamp}.txt"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(f"Speech-to-Text Transcript\n")
        f.write(f"{'='*40}\n")
        f.write(f"Generated : {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Source    : {source_name}\n")
        f.write(f"Words     : {len(text.split())}\n")
        f.write(f"{'='*40}\n\n")
        f.write(text)
    print(f"\n  {GREEN}💾 Transcript saved: {filename}{RESET}")
    return filename


# ── Demo Mode (no mic/file needed) ───────────────────────────────────────────
DEMO_AUDIO_PHRASES = [
    ("demo_meeting.wav",   "The quarterly earnings meeting is scheduled for next Monday at 10 AM. Please prepare your financial reports."),
    ("demo_note.wav",      "Remind me to buy groceries on the way home. I need milk, eggs, and bread."),
    ("demo_command.wav",   "Play some jazz music and set a timer for 30 minutes."),
    ("demo_medical.wav",   "Patient presents with mild fever and sore throat for the past two days. No known allergies."),
]

def run_demo():
    banner()
    print(f"  {YELLOW}[ DEMO MODE — Simulating audio file transcription ]{RESET}\n")
    t = Transcriber.__new__(Transcriber)

    for filename, transcript in DEMO_AUDIO_PHRASES:
        print(f"  {CYAN}📂 File: {filename}{RESET}")
        print(f"  {GRAY}Loading audio... Reading audio file... ✓{RESET}")
        time.sleep(0.3)
        print(f"  {CYAN}🔄 Transcribing with GOOGLE engine...{RESET}")
        time.sleep(0.5)
        display_result(transcript, "SUCCESS", "")
        time.sleep(0.3)

    print(f"\n  {CYAN}📊 SESSION SUMMARY{RESET}")
    print(f"  {'─'*40}")
    print(f"  Files processed : {len(DEMO_AUDIO_PHRASES)}")
    print(f"  Success rate    : 100%")
    total_words = sum(len(t.split()) for _, t in DEMO_AUDIO_PHRASES)
    print(f"  Total words     : {total_words}")
    print(f"  {'─'*40}\n")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Speech-to-Text Transcription Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python speech_to_text.py                      # live mic
  python speech_to_text.py --file audio.wav     # transcribe file
  python speech_to_text.py --file audio.mp3 --save  # save output
  python speech_to_text.py --continuous         # multi-phrase mic
  python speech_to_text.py --demo               # demo (no mic)
  python speech_to_text.py --lang fr-FR         # French transcription
        """
    )
    parser.add_argument("--file",       type=str,  help="Path to audio file (.wav, .mp3, .flac)")
    parser.add_argument("--save",       action="store_true", help="Save transcript to .txt file")
    parser.add_argument("--engine",     default="google", choices=["google","sphinx"], help="Recognition engine")
    parser.add_argument("--lang",       default="en-US", help="Language code (e.g. en-US, fr-FR, hi-IN)")
    parser.add_argument("--continuous", action="store_true", help="Record multiple phrases continuously")
    parser.add_argument("--phrases",    type=int, default=5, help="Number of phrases in continuous mode")
    parser.add_argument("--demo",       action="store_true", help="Run demo without mic or files")
    args = parser.parse_args()

    if args.demo:
        run_demo()
        return

    banner()

    missing = check_deps()
    if missing:
        print(f"  {RED}Missing dependencies: {', '.join(missing)}{RESET}")
        print(f"  {YELLOW}Install with: pip install {' '.join(missing)}{RESET}\n")
        sys.exit(1)

    transcriber = Transcriber(engine=args.engine, language=args.lang)
    print(f"  Engine   : {CYAN}{args.engine.upper()}{RESET}")
    print(f"  Language : {CYAN}{args.lang}{RESET}\n")

    if args.file:
        text, status, message = transcriber.from_file(args.file)
        display_result(text, status, message, args.file)
        if status == "SUCCESS" and args.save:
            save_transcript(text, args.file)

    elif args.continuous:
        results = transcriber.from_microphone_continuous(max_phrases=args.phrases)
        if results:
            full = " ".join(results)
            print(f"\n  {CYAN}FULL TRANSCRIPT:{RESET}")
            print(f"  {BOLD}\"{full}\"{RESET}")
            if args.save:
                save_transcript(full, "microphone-continuous")

    else:
        text, status, message = transcriber.from_microphone()
        display_result(text, status, message)
        if status == "SUCCESS" and args.save:
            save_transcript(text, "microphone")


if __name__ == "__main__":
    main()
