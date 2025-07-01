"""
Local translation helper
Uses Meta's NLLB to translate
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

_MODEL = "facebook/nllb-200-distilled-1.3B"  # original
#_MODEL = "facebook/nllb-200-3.3B"  # testing only
tokenizer = AutoTokenizer.from_pretrained(_MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(_MODEL, torch_dtype="auto")
translator = pipeline("translation", model=model, tokenizer=tokenizer, device=0)  # GPU

_LANG_CODE = {"en": "eng_Latn", "tw": "aka_Latn"}


def translate(text: str, src: str, tgt: str) -> str:
    return translator(
        text,
        src_lang=_LANG_CODE[src],
        tgt_lang=_LANG_CODE[tgt],
        max_length=200,
    )[0]["translation_text"]
