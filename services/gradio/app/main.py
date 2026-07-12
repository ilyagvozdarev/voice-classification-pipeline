import logging

import gradio as gr
import requests

from .client import master_client
from .config import settings

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger("gradio")


def handle_audio(audio_path: str | None):
    """Send the recorded/uploaded audio to master and unpack the result."""
    if not audio_path:
        return "", "", "Please record or upload an audio clip first."

    try:
        result = master_client.process(audio_path)
    except requests.HTTPError as exc:
        detail = exc.response.text if exc.response is not None else str(exc)
        logger.warning("Master returned an error: %s", detail)
        return "", "", f"Error from pipeline: {detail}"
    except requests.RequestException as exc:
        logger.exception("Could not reach master")
        return "", "", f"Could not reach master service: {exc}"

    sentiment = result["sentiment"]
    sentiment_text = f"{sentiment['label']} ({sentiment['score']:.2%})"
    return result["transcript"], sentiment_text, result["answer"]


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="Voice ML Pipeline") as demo:
        gr.Markdown(
            "# 🎙️ Voice ML Pipeline\n"
            "Record or upload audio. It is transcribed (ASR), then analyzed "
            "for sentiment (BERT) and answered (LLM) in parallel."
        )
        audio_in = gr.Audio(
            sources=["microphone", "upload"], type="filepath", label="Audio input"
        )
        submit = gr.Button("Process", variant="primary")

        transcript_out = gr.Textbox(label="Transcript (ASR)", interactive=False)
        sentiment_out = gr.Textbox(label="Sentiment (BERT)", interactive=False)
        answer_out = gr.Textbox(label="Answer (LLM)", interactive=False, lines=4)

        submit.click(
            fn=handle_audio,
            inputs=audio_in,
            outputs=[transcript_out, sentiment_out, answer_out],
        )
    return demo


demo = build_interface()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
