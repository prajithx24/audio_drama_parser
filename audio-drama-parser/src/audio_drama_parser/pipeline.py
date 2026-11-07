
from .ingestion import load_script
from .preprocessing import clean_script_text


def process_script(file_path: str) -> str:
	raw_text = load_script(file_path)
	return clean_script_text(raw_text)
