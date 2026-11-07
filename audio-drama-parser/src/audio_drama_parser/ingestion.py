
from pathlib import Path


def load_script(file_path: str) -> str:
	path = Path(file_path)

	if path.suffix.lower() != ".txt":
		raise ValueError(
			f"Unsupported script format for '{file_path}'. Only .txt files are supported."
		)

	if not path.is_file():
		raise FileNotFoundError(f"Script file not found: '{file_path}'")

	try:
		with path.open("r", encoding="utf-8-sig", newline="") as script_file:
			return script_file.read()
	except UnicodeDecodeError as error:
		raise UnicodeDecodeError(
			error.encoding,
			error.object,
			error.start,
			error.end,
			f"Unable to decode script '{file_path}' as UTF-8: {error.reason}",
		) from error