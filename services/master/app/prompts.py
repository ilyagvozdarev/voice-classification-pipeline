"""Prompt templates the master applies before calling the LLM service.

The LLM service is prompt-agnostic (it just completes text), so the master owns
the two prompts: one to *restore* the raw ASR transcript, one to *classify* the
restored text. Tweak these freely — they are the main knob for output quality.
"""

# LLM #1: clean up the raw ASR output.
RESTORATION_PROMPT = """You are a transcript post-processor. Given a raw ASR \
transcript, produce a cleaned version by:
- diarizing the text (attribute utterances to speakers),
- restoring punctuation, casing and obvious recognition errors,
- tagging each utterance with its message type (question, statement, request, ...).

Return only the restored transcript.

Raw transcript:
{transcript}
"""

# LLM #2: multi-label classification of the restored text (LLM's own take,
# used alongside the BERT classifier for comparison).
CLASSIFICATION_PROMPT = """Assign all applicable category labels to the message \
below (multi-label classification). Respond with a comma-separated list of \
labels and nothing else.

Message:
{text}
"""


def build_restoration_prompt(transcript: str) -> str:
    """Return the restoration prompt for a raw transcript."""
    return RESTORATION_PROMPT.format(transcript=transcript)


def build_classification_prompt(text: str) -> str:
    """Return the classification prompt for a restored text."""
    return CLASSIFICATION_PROMPT.format(text=text)
