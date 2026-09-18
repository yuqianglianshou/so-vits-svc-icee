import asyncio
import random
import sys

import edge_tts
from edge_tts import VoicesManager
from langdetect import DetectorFactory, detect

DetectorFactory.seed = 0


def normalize_locale(lang: str) -> str:
    lang = (lang or "").strip()
    if lang.lower() in {"zh-cn", "zh-tw"}:
        return lang[:-2] + lang[-2:].upper()
    if "-" in lang and len(lang) >= 5:
        left, right = lang.split("-", 1)
        return f"{left.lower()}-{right.upper()}"
    return lang


def resolve_voice(voices: VoicesManager, lang: str, gender: str | None = None):
    normalized = normalize_locale(lang)
    if gender is not None:
        matches = voices.find(Gender=gender, Locale=normalized) or voices.find(Gender=gender, Language=lang)
        if matches:
            voice = random.choice(matches)["Name"]
            print(f"Using random {normalized or lang} voice: {voice}")
            return voice

    locale_matches = voices.find(Locale=normalized) if normalized else []
    if locale_matches:
        voice = random.choice(locale_matches)["Name"]
        print(f"Using random locale voice: {voice}")
        return voice

    language_matches = voices.find(Language=lang) if lang else []
    if language_matches:
        voice = random.choice(language_matches)["Name"]
        print(f"Using random language voice: {voice}")
        return voice

    return lang

async def synthesize_tts(text: str, lang: str, rate: str, volume: str, gender: str | None = None, output_file: str = "tts.wav") -> None:
    voices = await VoicesManager.create()
    voice = resolve_voice(voices, lang, gender)
    communicate = edge_tts.Communicate(text=text, voice=voice, rate=rate, volume=volume)
    await communicate.save(output_file)


def parse_cli_args(argv):
    text = argv[1]
    lang = detect(text) if argv[2] == "Auto" else argv[2]
    rate = argv[3]
    volume = argv[4]
    gender = argv[5] if len(argv) >= 6 else None
    output_file = argv[6] if len(argv) >= 7 else "tts.wav"
    return text, lang, rate, volume, gender, output_file


def main(argv=None) -> None:
    argv = argv or sys.argv
    text, lang, rate, volume, gender, output_file = parse_cli_args(argv)
    print("Running TTS...")
    print(f"Text: {text}, Language: {lang}, Gender: {gender}, Rate: {rate}, Volume: {volume}")

    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        asyncio.run(synthesize_tts(text, lang, rate, volume, gender, output_file))
    else:
        loop = asyncio.get_event_loop_policy().get_event_loop()
        try:
            loop.run_until_complete(synthesize_tts(text, lang, rate, volume, gender, output_file))
        finally:
            loop.close()


if __name__ == "__main__":
    main()
