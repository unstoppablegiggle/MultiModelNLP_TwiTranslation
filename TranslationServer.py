"""
Flask Speech‑Translation Server
--------------------------------
Accepts a WAV file
translates (en2tw | tw2en)
Returns JSON:
    {
      "source_text": "<original>",
      "translated_text": "<twi or english>",
      "audio": "<base64‑wav>"
     }
"""


import base64
import os
from flask import Flask, request, jsonify
import whisper
from local_translate import translate # NLLB
from TTS.api import TTS
from transformers import pipeline
from huggingface_hub import snapshot_download

LANGUAGE_CONFIDENCE_THRESHOLD = 0.85  # switch to other model is below this

# ─── Models ────────────────────────────────────────────────────────────────

# STT
TWI_MODEL_DIR = snapshot_download("jackiejoe45/twi_trained_whisper")

asr_twi = pipeline(
    "automatic-speech-recognition",
    model=TWI_MODEL_DIR,
    device=0,  # GPU
    torch_dtype="auto",)  # Twi

asr_en = whisper.load_model("small", device="cuda")  # English fallback

# TTS #
tts_en = TTS(model_name="tts_models/en/ljspeech/vits")
tts_twi = TTS("tts_models/tw_asante/openbible/vits")

app = Flask(__name__)


# --- Pipeline ---
def tts_speak(text: str, lang: str, out_path: str):
    voice = tts_en if lang == "en" else tts_twi
    voice.tts_to_file(text=text, file_path=out_path)


def process(wav_path: str):
    """
    Returns (src_lang, tgt_lang, src_text, tgt_text, base64_wav)
    """

    # Detect language with Whisper
    audio = whisper.load_audio(wav_path)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(asr_en.device)
    _, probs = asr_en.detect_language(mel)
    lang_code = max(probs, key=probs.get)

    # display conf and lang in server
    app.logger.info(f"[LANG DETECT] Whisper detected: {lang_code} (conf: {probs[lang_code]:.2f})")

    confidence = probs.get("en", 0.0)

    # Switch to twi if below threshold
    if lang_code == "en" and confidence >= LANGUAGE_CONFIDENCE_THRESHOLD:
        app.logger.info(f"[LANG OK] English detected with confidence {confidence:.2f}")
        result = asr_en.transcribe(wav_path, language="en")
        text = result["text"].strip()
        src_lang = "en"
    else:
        app.logger.warning(f"[LANG FALLBACK] Switching to Twi model (lang='{lang_code}', conf={confidence:.2f})")
        result = asr_twi(wav_path, chunk_length_s=30)
        text = result["text"].strip()
        src_lang = "tw"

    tgt_lang = "tw" if src_lang == "en" else "en"
    translated = translate(text, src_lang, tgt_lang)

    out_wav = "output.wav"
    tts_speak(translated, tgt_lang, out_wav)

    with open(out_wav, "rb") as f:
        audio_b64 = base64.b64encode(f.read()).decode()

    return src_lang, tgt_lang, text, translated, audio_b64



# --- Route ---
@app.route("/translate_audio", methods=["POST"])
def translate_audio():
    if "audio" not in request.files:
        return jsonify(error="Missing audio"), 400

    wav_path = "input.wav"
    request.files["audio"].save(wav_path)

    try:
        src, tgt, src_text, tgt_text, a64 = process(wav_path)
    except Exception as e:
        # print full traceback to the console (needed for debug with app)
        app.logger.exception("translate_audio failed")
        return jsonify(error=str(e)), 500
    finally:
        if os.path.exists(wav_path):
            os.remove(wav_path)

    return jsonify(
        source_lang=src,
        target_lang=tgt,
        source_text=src_text,
        translated_text=tgt_text,
        audio=a64,
    )


if __name__ == "__main__":
    #app.run(host="0.0.0.0", port=5000, debug=False)
    from waitress import serve # install in dev env
    serve(app, host="0.0.0.0", port=5000)