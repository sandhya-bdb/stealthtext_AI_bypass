import logging
import random
import re
import os
import ssl
import nltk
from groq import Groq
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# ── NLTK bootstrap ───────────────────────────────────────────────────────────
# macOS often rejects default SSL certs for NLTK downloads.
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    logger.info("Downloading NLTK corpora…")
    nltk.download("wordnet", quiet=True)
    nltk.download("averaged_perceptron_tagger", quiet=True)
    nltk.download("punkt", quiet=True)

# ── Input validation constants ───────────────────────────────────────────────
MIN_CHARS = 10    # Prevent trivially short rewrites
MAX_CHARS = 8000  # Match detector limit; avoids massive Groq token bills

# ── AI buzzword replacements (post-processing safety net) ────────────────────
AI_STOP_WORDS = {
    "delve":          ["dig", "look", "explore"],
    "crucial":        ["important", "key", "vital"],
    "moreover":       ["plus", "also", "besides"],
    "furthermore":    ["also", "then", "plus"],
    "utilize":        ["use", "take"],
    "leverage":       ["use", "rely on"],
    "meticulous":     ["careful", "detailed"],
    "showcase":       ["show", "present"],
    "pivotal":        ["key", "major"],
    "paramount":      ["top", "critical"],
    "transformative": ["game-changing", "big"],
}

SYSTEM_PROMPT_CASUAL = """You are a skilled human writer. Rewrite the given text so it sounds like a real person wrote it in a casual, conversational style.

STRICT RULES you MUST follow:
1. COMPLETELY restructure sentences. Do NOT copy the original structure.
2. MIX sentence lengths: some very short (3-6 words), some long and winding (25-40 words).
3. Use casual, conversational tone. Add hedges like "honestly", "basically", "kind of", "I'd say".
4. Replace formal words: "utilize"→"use", "leverage"→"rely on", "implement"→"set up", "achieve"→"get to", "enable"→"let".
5. NEVER use: delve, crucial, moreover, furthermore, meticulous, showcase, realm, tapestry, pivotal, paramount, seamless, robust, comprehensive, innovative, groundbreaking, transformative, notably, consequently, in conclusion.
6. Use dashes, ellipses, and rhetorical questions naturally.
7. Keep the core meaning 100% intact.
8. Return ONLY the rewritten text. No explanations, no preamble, no quotes."""

SYSTEM_PROMPT_PROFESSIONAL = """You are a highly skilled professional writer and academic editor. Rewrite the given text so it sounds like a real expert wrote it — NOT an AI chatbot. It must be polished, professional, and suitable for official or educational contexts. Do NOT make it overly emotional, informal, or use slang.

STRICT RULES you MUST follow:
1. COMPLETELY restructure sentences to avoid predictable AI writing patterns.
2. MIX sentence lengths: some short and punchy, some longer and complex. AI writing is overly uniform; human writing varies in rhythm.
3. Maintain a formal, authoritative, and objective tone. Do NOT add informal hedges like "honestly", "basically", "kind of", "you know", or exclamation marks.
4. Keep the vocabulary sophisticated but clear. Replace overused AI transition words and buzzwords.
5. NEVER use: delve, crucial, moreover, furthermore, meticulous, showcase, realm, tapestry, pivotal, paramount, seamless, robust, comprehensive, innovative, groundbreaking, transformative, notably, consequently, in conclusion, leverage, utilize. Replace them with standard professional alternatives (e.g. use 'explore/examine' instead of 'delve', 'important/essential' instead of 'crucial', 'also/in addition' instead of 'furthermore/moreover').
6. Do NOT use overly dramatic punctuation like ellipses (...) or multiple exclamation marks.
7. Keep the core meaning 100% intact.
8. Return ONLY the rewritten text. No explanations, no preamble, no quotes."""

load_dotenv()
_GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "").strip()


class TextHumanizer:
    def __init__(self):
        logger.info("Initializing TextHumanizer (Groq SDK / llama-3.3-70b-versatile)…")
        if not _GROQ_API_KEY:
            logger.error(
                "GROQ_API_KEY is not set in environment. "
                "Humanizer will return original text unchanged."
            )
            self.client = None
        else:
            try:
                self.client = Groq(api_key=_GROQ_API_KEY)
                self.model  = "llama-3.3-70b-versatile"
                logger.info("Groq client initialised successfully.")
            except Exception as exc:
                logger.exception("Failed to initialise Groq client: %s", exc)
                self.client = None

    # ── Validation ───────────────────────────────────────────────────────────

    @staticmethod
    def _validate(text: str) -> None:
        """Raise ValueError for inputs that are clearly unsuitable."""
        if not isinstance(text, str):
            raise TypeError(f"text must be a str, got {type(text).__name__}")
        stripped = text.strip()
        if len(stripped) < MIN_CHARS:
            raise ValueError(
                f"Input text is too short ({len(stripped)} chars). "
                f"Please provide at least {MIN_CHARS} characters."
            )
        if len(stripped) > MAX_CHARS:
            raise ValueError(
                f"Input text is too long ({len(stripped)} chars). "
                f"Maximum allowed is {MAX_CHARS} characters."
            )

    # ── Post-processing ──────────────────────────────────────────────────────

    @staticmethod
    def _strip_ai_words(text: str) -> str:
        """Replace known AI buzzwords that slipped through the prompt."""
        for word, replacements in AI_STOP_WORDS.items():
            if word in text.lower():
                replacement = random.choice(replacements)
                text = re.sub(re.escape(word), replacement, text, flags=re.IGNORECASE)
        return text

    # ── Public API ───────────────────────────────────────────────────────────

    def rewrite(self, text: str, tone: str = "casual") -> str:
        """
        Rewrite *text* using Groq LLM to sound more human.

        Returns the rewritten text, or the original text unchanged if:
        - The Groq client is not initialised (missing API key).
        - The API call fails for any reason.

        Raises:
            ValueError: if text is too short or too long.
            TypeError:  if text is not a str.
        """
        self._validate(text)

        if not self.client:
            logger.warning("Groq client not available — returning original text.")
            return text

        system_prompt = SYSTEM_PROMPT_PROFESSIONAL if tone == "professional" else SYSTEM_PROMPT_CASUAL
        logger.info("Sending %d chars to Groq for rewriting (tone=%s)…", len(text), tone)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": f"Rewrite this text:\n\n{text}"},
                ],
                temperature=0.85 if tone == "professional" else 0.9,
                max_tokens=2048,
            )
            result = response.choices[0].message.content.strip()
            logger.info("Groq rewrite complete — output %d chars.", len(result))

            result = self._strip_ai_words(result)
            return result

        except Exception as exc:
            logger.exception("Groq API call failed: %s", exc)
            return text

    def advanced_rewrite(self, text: str, tone: str = "casual", api_key: str = None) -> str:
        """Alias for rewrite() — kept for backwards compatibility."""
        return self.rewrite(text, tone=tone)
