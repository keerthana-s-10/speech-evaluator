"""
transcriber.py
--------------
Wraps OpenAI Whisper so the rest of the app only calls one function
and gets back a plain transcript string.

The model is cached at module level so it is only loaded once
per Streamlit session, regardless of how many files are analysed.
"""

import whisper

_model_cache: dict = {}


def transcribe_audio(filepath: str, model_size: str = "base") -> str:
    """
    Transcribe an audio file using OpenAI Whisper.

    Parameters
    ----------
    filepath   : absolute or relative path to the audio file on disk.
    model_size : Whisper model variant.
                 "tiny"   — fastest, least accurate
                 "base"   — good speed/accuracy balance (recommended)
                 "small"  — better accuracy, slower
                 "medium" — high accuracy, needs more RAM
                 "large"  — best accuracy, slowest

    Returns
    -------
    Plain-text transcript string with no timestamps.
    """
    if model_size not in _model_cache:
        _model_cache[model_size] = whisper.load_model(model_size)

    model  = _model_cache[model_size]
    result = model.transcribe(filepath)
    return result["text"].strip()