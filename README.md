# Multi-Context Speech Performance Evaluator
A Python application that evaluates speech quality across different contexts using acoustic signal processing and NLP.

## Contexts
- Job Interview
- Idea Pitch
- Public Speaking

## Features
- Acoustic analysis: pitch stability, speech rate, energy consistency, pause ratio
- Semantic analysis: sentiment, keyword density, clarity
- OpenAI Whisper transcription
- Streamlit dashboard with radar chart visualisation

## Setup

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python -m textblob.download_corpora
streamlit run app.py
```

## Requirements
- Python 3.10+
- ffmpeg installed on your system
