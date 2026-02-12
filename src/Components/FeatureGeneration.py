import re
import unicodedata
import html
import numpy as np
from bs4 import BeautifulSoup
from collections import Counter

# NLP libs
import nltk
import spacy
import textstat
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from flair.models import SequenceTagger
from flair.data import Sentence
from sentence_transformers import SentenceTransformer


class TextFeatureGenerator:
    """
    Unified feature generation class for news article analysis.
    Each sub-task uses the most appropriate NLP library.
    """

    def __init__(
        self,
        enable_ner: bool = True,
        enable_semantic: bool = True,
        max_ner_chars: int = 3000
    ):
        # NLTK
        nltk.download("punkt", quiet=True)
        nltk.download("stopwords", quiet=True)
        self.stop_words = set(stopwords.words("english"))

        # spaCy (POS only)
        self.nlp_pos = spacy.load("en_core_web_sm", disable=["ner", "parser"])

        # Flair NER
        self.enable_ner = enable_ner
        self.max_ner_chars = max_ner_chars
        if enable_ner:
            self.ner_tagger = SequenceTagger.load("flair/ner-english")

        # Transformer for semantic density
        self.enable_semantic = enable_semantic
        if enable_semantic:
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")

    # -------------------------
    # Canonical cleaning (SAFE)
    # -------------------------
    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            return ""

        text = html.unescape(text)
        text = unicodedata.normalize("NFKC", text)
        text = BeautifulSoup(text, "lxml").get_text(separator=" ")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n", text)
        return text.strip()

    # -------------------------
    # Length & structure
    # -------------------------
    def _length_features(self, text: str) -> dict:
        return {
            "char_count": len(text),
            "char_count_no_spaces": len(text.replace(" ", "")),
            "paragraph_count": text.count("\n") + 1
        }

    # -------------------------
    # Lexical richness
    # -------------------------
    def _lexical_features(self, text: str) -> dict:
        tokens = [t.lower() for t in word_tokenize(text) if t.isalpha()]
        counts = Counter(tokens)

        return {
            "word_count": len(tokens),
            "unique_word_count": len(counts),
            "lexical_diversity": len(counts) / max(len(tokens), 1),
            "hapax_ratio": sum(1 for c in counts.values() if c == 1) / max(len(tokens), 1)
        }

    # -------------------------
    # Stopword stats
    # -------------------------
    def _stopword_features(self, text: str) -> dict:
        tokens = [t.lower() for t in word_tokenize(text) if t.isalpha()]
        stopword_count = sum(1 for t in tokens if t in self.stop_words)

        return {
            "stopword_count": stopword_count,
            "stopword_ratio": stopword_count / max(len(tokens), 1),
            "content_word_count": len(tokens) - stopword_count
        }

    # -------------------------
    # POS statistics
    # -------------------------
    def _pos_features(self, text: str) -> dict:
        doc = self.nlp_pos(text)
        pos_counts = Counter(tok.pos_ for tok in doc if tok.is_alpha)

        return {
            "noun_count": pos_counts.get("NOUN", 0),
            "verb_count": pos_counts.get("VERB", 0),
            "adj_count": pos_counts.get("ADJ", 0),
            "adv_count": pos_counts.get("ADV", 0),
            "pronoun_count": pos_counts.get("PRON", 0)
        }

    # -------------------------
    # Named Entity features (Flair)
    # -------------------------
    def _ner_features(self, text: str) -> dict:
        if not self.enable_ner:
            return {}

        sentence = Sentence(text[: self.max_ner_chars])
        self.ner_tagger.predict(sentence)

        labels = [ent.get_label("ner").value for ent in sentence.get_spans("ner")]
        counts = Counter(labels)

        return {
            "person_count": counts.get("PER", 0),
            "org_count": counts.get("ORG", 0),
            "location_count": counts.get("LOC", 0),
            "misc_entity_count": counts.get("MISC", 0),
            "total_entity_count": len(labels)
        }

    # -------------------------
    # Readability metrics
    # -------------------------
    def _readability_features(self, text: str) -> dict:
        return {
            "flesch_reading_ease": textstat.flesch_reading_ease(text),
            "flesch_kincaid_grade": textstat.flesch_kincaid_grade(text),
            "gunning_fog": textstat.gunning_fog(text)
        }

    # -------------------------
    # Semantic density (Transformer)
    # -------------------------
    def _semantic_features(self, text: str) -> dict:
        if not self.enable_semantic:
            return {}

        sentences = [s.strip() for s in text.split(".") if len(s.strip()) > 20]

        if len(sentences) < 2:
            return {"semantic_variance": 0.0}

        embeddings = self.embedder.encode(sentences[:20])
        variance = float(np.mean(np.var(embeddings, axis=0)))

        return {"semantic_variance": variance}

    # -------------------------
    # Public API
    # -------------------------
    def generate(self, raw_text: str) -> dict:
        text = self.clean_text(raw_text)

        features = {}
        features.update(self._length_features(text))
        features.update(self._lexical_features(text))
        features.update(self._stopword_features(text))
        features.update(self._pos_features(text))
        features.update(self._ner_features(text))
        features.update(self._readability_features(text))
        features.update(self._semantic_features(text))

        return features
