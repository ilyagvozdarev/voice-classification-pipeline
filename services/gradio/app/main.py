import logging

import gradio as gr
import requests

from .client import master_client
from .config import settings

logging.basicConfig(level=settings.log_level)
logger = logging.getLogger("gradio")


def _format_labels(labels: list[dict]) -> str:
    if not labels:
        return "(no labels above threshold)"
    return "\n".join(f"{item['label']}  {item['score']:.1%}" for item in labels)


def handle_audio(audio_path: str | None):
    """
    Send the recorded/uploaded audio to master and unpack the result.

    Returns
    -------
    tuple of str
        (transcript, restored_text, bert_labels, llm_classification) for the
        four output boxes. On error, the message is placed in the last box.
    """
    if not audio_path:
        return "", "", "", "Please record or upload an audio clip first."

    try:
        result = master_client.process(audio_path)
    except requests.HTTPError as exc:
        detail = exc.response.text if exc.response is not None else str(exc)
        logger.warning("Master returned an error: %s", detail)
        return "", "", "", f"Error from pipeline: {detail}"
    except requests.RequestException as exc:
        logger.exception("Could not reach master")
        return "", "", "", f"Could not reach master service: {exc}"

    return (
        result["transcript"],
        result["restored_text"],
        _format_labels(result["bert_labels"]),
        result["llm_classification"],
    )


def build_interface() -> gr.Blocks:
    with gr.Blocks(title="Voice Classification Pipeline") as demo:
        gr.Markdown(
            "# Voice Classification Pipeline\n"
            "Record or upload audio. It is transcribed (ASR), **restored** by "
            "the LLM (diarization, punctuation, message types), then classified "
            "two ways: multi-label by BERT and by the LLM."
        )
        audio_in = gr.Audio(
            sources=["microphone", "upload"], type="filepath", label="Audio input"
        )
        submit = gr.Button("Process", variant="primary")

        transcript_out = gr.Textbox(label="Transcript — raw (ASR)", interactive=False)
        restored_out = gr.Textbox(
            label="Restored text (LLM)", interactive=False, lines=4
        )
        bert_out = gr.Textbox(
            label="Multi-label classification (BERT)", interactive=False, lines=4
        )
        llm_class_out = gr.Textbox(
            label="Classification (LLM)", interactive=False, lines=3
        )

        submit.click(
            fn=handle_audio,
            inputs=audio_in,
            outputs=[transcript_out, restored_out, bert_out, llm_class_out],
        )
    return demo


demo = build_interface()

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
