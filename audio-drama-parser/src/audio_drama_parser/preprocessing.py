
import re


def clean_script_text(text: str) -> str:
	text = text.replace("\r\n", "\n").replace("\r", "\n")
	lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.split("\n")]
	return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def preprocess_text(text: str) -> str:
	return clean_script_text(text)


__all__ = ["clean_script_text", "preprocess_text"]
