"""
speech_evaluator.py
-------------------
Core OOP evaluation engine.
Loads a JSON context config and dynamically scores a speech sample.

Also contains all semantic feature extractors (spaCy + TextBlob)
so the app only needs to import from this one file.

Place at the project root — same level as app.py.
"""

import json
import numpy as np
from pathlib import Path
from dataclasses import dataclass
from typing import Dict

import spacy
from textblob import TextBlob

# Load spaCy model once at import time
nlp = spacy.load("en_core_web_sm")


# ===========================================================================
# Data Containers
# ===========================================================================

@dataclass
class AcousticFeatures:
    pitch_stability:    float   # 0-1  stable fundamental frequency
    speech_rate:        float   # 0-1  normalised WPM
    energy_consistency: float   # 0-1  RMS energy variance
    pause_ratio:        float   # 0-1  meaningful-pause proportion


@dataclass
class SemanticFeatures:
    sentiment_score: float   # 0-1  positive polarity
    keyword_density: float   # 0-1  context-relevant keyword ratio
    clarity_score:   float   # 0-1  sentence structure + lexical diversity


@dataclass
class EvaluationResult:
    context_name:     str
    raw_scores:       Dict[str, float]   # normalised 0-1 per feature
    weighted_scores:  Dict[str, float]   # raw x weight
    final_score:      float              # 0-100
    grade:            str                # A / B / C / D / F
    feedback:         Dict[str, str]     # per-feature coaching text
    dimension_scores: Dict[str, float]   # 0-100, used by radar chart


# ===========================================================================
# Context Config Loader
# ===========================================================================

class ContextConfig:
    """
    Loads a JSON context file from the /contexts directory.

    To add a new context (e.g. "podcast"), drop a new
    contexts/podcast.json following the same schema.
    Zero code changes needed anywhere else.

    Expected JSON schema
    --------------------
    {
      "context_name": "...",
      "description":  "...",
      "weights":      { "feature": float, ... },        must sum to 1.0
      "thresholds":   { "feature": { "min", "ideal", "max" }, ... },
      "feedback_templates": { "feature": "... {value:.2f} ..." }   optional
    }
    """

    CONTEXTS_DIR = Path(__file__).parent / "contexts"

    def __init__(self, context_key: str):
        config_path = self.CONTEXTS_DIR / f"{context_key}.json"
        if not config_path.exists():
            raise FileNotFoundError(
                f"No config found for context '{context_key}' at {config_path}\n"
                f"Available contexts: {self.available_contexts()}"
            )

        with open(config_path, "r", encoding="utf-8") as f:
            raw: dict = json.load(f)

        self.context_name:       str              = raw["context_name"]
        self.description:        str              = raw["description"]
        self.weights:            Dict[str, float] = raw["weights"]
        self.thresholds:         Dict[str, dict]  = raw["thresholds"]
        self.feedback_templates: Dict[str, str]   = raw.get("feedback_templates", {})

        self._validate_weights()

    def _validate_weights(self) -> None:
        total = sum(self.weights.values())
        if not (0.999 < total < 1.001):
            raise ValueError(
                f"[{self.context_name}] weights must sum to 1.0, got {total:.4f}. "
                "Check your JSON config."
            )

    @classmethod
    def available_contexts(cls) -> list:
        return sorted(p.stem for p in cls.CONTEXTS_DIR.glob("*.json"))


# ===========================================================================
# Semantic Feature Extractors
# ===========================================================================

CONTEXT_KEYWORDS: Dict[str, list] = {
    "interview": [
        "experience", "led", "achieved", "improved", "delivered",
        "managed", "developed", "collaborated", "results", "impact",
    ],
    "pitch": [
        "problem", "solution", "market", "revenue", "scale",
        "traction", "unique", "opportunity", "growth", "vision",
    ],
    "public_speaking": [
        "together", "imagine", "future", "change", "believe",
        "community", "story", "challenge", "inspire", "action",
    ],
}


def extract_sentiment_score(transcript: str) -> float:
    """
    TextBlob sentence-level polarity mapped from [-1, +1] to [0, 1].
    Neutral speech (0.0) maps to 0.5; fully positive (+1.0) maps to 1.0.
    """
    blob       = TextBlob(transcript)
    sentiments = [s.sentiment.polarity for s in blob.sentences]
    if not sentiments:
        return 0.5
    mean_pol = float(np.mean(sentiments))
    return round((mean_pol + 1.0) / 2.0, 4)


def extract_keyword_density(transcript: str, context_key: str) -> float:
    """
    spaCy lemmatises every content word (NOUN/VERB/ADJ/ADV, non-stop),
    then measures overlap with the context keyword bank.

    density = matched / total_content_words
    Ideal density >= 0.15 maps to score 1.0; scales linearly below.
    """
    doc      = nlp(transcript.lower())
    keywords = set(CONTEXT_KEYWORDS.get(context_key, []))

    content  = [
        t for t in doc
        if t.pos_ in {"NOUN", "VERB", "ADJ", "ADV"}
        and not t.is_stop
        and not t.is_punct
    ]
    if not content:
        return 0.0

    matched = sum(1 for t in content if t.lemma_ in keywords)
    raw_den = matched / len(content)
    return round(float(np.clip(raw_den / 0.15, 0.0, 1.0)), 4)


def extract_clarity_score(transcript: str) -> float:
    """
    Composite clarity from three sub-scores:

    (a) Sentence length score  — target 10-20 words/sentence   weight 0.50
    (b) Lexical diversity      — type-token ratio, ideal >= 0.6  weight 0.30
    (c) Filler word penalty    — um / uh / like / basically ...  weight 0.20
    """
    doc       = nlp(transcript)
    sentences = list(doc.sents)
    if not sentences:
        return 0.0

    # (a) sentence length
    lengths  = [len([t for t in s if not t.is_space]) for s in sentences]
    mean_len = float(np.mean(lengths))
    LO, HI   = 10, 20
    if LO <= mean_len <= HI:
        length_score = 1.0
    elif mean_len < LO:
        length_score = mean_len / LO
    else:
        length_score = max(0.0, 1.0 - (mean_len - HI) / 20.0)

    # (b) lexical diversity (type-token ratio)
    tokens    = [t.text.lower() for t in doc if t.is_alpha]
    ttr       = len(set(tokens)) / (len(tokens) + 1e-9)
    div_score = float(np.clip(ttr / 0.6, 0.0, 1.0))

    # (c) filler word penalty
    FILLERS   = {"um", "uh", "like", "you know", "basically",
                 "literally", "right", "so"}
    fillers   = sum(1 for t in tokens if t in FILLERS)
    fil_ratio = fillers / (len(tokens) + 1e-9)
    fil_score = float(np.clip(1.0 - fil_ratio * 5, 0.0, 1.0))

    return round(0.5 * length_score + 0.3 * div_score + 0.2 * fil_score, 4)


# ===========================================================================
# Core Evaluator
# ===========================================================================

class SpeechEvaluator:
    """
    Scores a speech sample against a loaded ContextConfig.

    Scoring pipeline (per feature)
    -------------------------------
    1. Clamp raw value to [0, 1]
    2. Piecewise-linear map through (min, ideal, max) thresholds → [0, 1]
    3. Multiply by context weight
    4. Sum all weighted scores → [0, 1] → scale to 0-100
    """

    GRADE_THRESHOLDS = [
        (90, "A"),
        (80, "B"),
        (70, "C"),
        (60, "D"),
        (0,  "F"),
    ]

    def __init__(self, context_key: str):
        self.config      = ContextConfig(context_key)
        self.context_key = context_key

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def evaluate(
        self,
        acoustic: AcousticFeatures,
        semantic: SemanticFeatures,
    ) -> EvaluationResult:

        all_features = {
            "pitch_stability":    acoustic.pitch_stability,
            "speech_rate":        acoustic.speech_rate,
            "energy_consistency": acoustic.energy_consistency,
            "pause_ratio":        acoustic.pause_ratio,
            "sentiment_score":    semantic.sentiment_score,
            "keyword_density":    semantic.keyword_density,
            "clarity_score":      semantic.clarity_score,
        }

        raw_scores:      Dict[str, float] = {}
        weighted_scores: Dict[str, float] = {}

        for feature, value in all_features.items():
            threshold = self.config.thresholds[feature]
            weight    = self.config.weights[feature]
            norm      = self._threshold_score(value, threshold)
            raw_scores[feature]      = round(norm, 4)
            weighted_scores[feature] = round(norm * weight, 4)

        final = sum(weighted_scores.values()) * 100

        return EvaluationResult(
            context_name     = self.config.context_name,
            raw_scores       = raw_scores,
            weighted_scores  = weighted_scores,
            final_score      = round(final, 2),
            grade            = self._assign_grade(final),
            feedback         = self._generate_feedback(raw_scores),
            dimension_scores = {k: round(v * 100, 1) for k, v in raw_scores.items()},
        )

    # -----------------------------------------------------------------------
    # Internal Helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _threshold_score(value: float, threshold: dict) -> float:
        """
        Fully transparent piecewise linear scorer.

          value < min            → linear 0.0 to 0.5
          min  <= value < ideal  → linear 0.5 to 1.0
          ideal <= value <= max  → 1.0 plateau
          value > max            → mild penalty, floor 0.85
        """
        lo    = threshold["min"]
        ideal = threshold["ideal"]
        hi    = threshold["max"]
        v     = float(np.clip(value, 0.0, 1.0))

        if v >= ideal:
            if v <= hi:
                return 1.0
            overshoot = (v - hi) / (1.0 - hi + 1e-9)
            return max(0.85, 1.0 - 0.15 * overshoot)
        if v >= lo:
            return 0.5 + 0.5 * ((v - lo) / (ideal - lo + 1e-9))
        return 0.5 * (v / (lo + 1e-9))

    def _assign_grade(self, score: float) -> str:
        for threshold, grade in self.GRADE_THRESHOLDS:
            if score >= threshold:
                return grade
        return "F"

    def _generate_feedback(self, raw_scores: Dict[str, float]) -> Dict[str, str]:
        feedback  = {}
        templates = self.config.feedback_templates
        for feature, score in raw_scores.items():
            if feature in templates:
                feedback[feature] = templates[feature].format(value=score)
            else:
                status = "strong" if score >= 0.75 else "needs improvement"
                feedback[feature] = (
                    f"{feature.replace('_', ' ').title()} scored "
                    f"{score:.2f} — {status}."
                )
        return feedback