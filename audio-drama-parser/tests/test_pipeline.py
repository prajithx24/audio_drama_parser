import pytest

from audio_drama_parser.pipeline import process_script


def test_process_script_loads_and_cleans_script(tmp_path):
	script_path = tmp_path / "script.txt"
	script_path.write_text(
		"  Alice:   Hello!  \n\n\nBob:  Hi!  ",
		encoding="utf-8",
	)

	assert process_script(str(script_path)) == "Alice: Hello!\n\nBob: Hi!"


def test_process_script_returns_empty_string_for_empty_file(tmp_path):
	script_path = tmp_path / "empty.txt"
	script_path.write_text("", encoding="utf-8")

	assert process_script(str(script_path)) == ""


def test_process_script_raises_for_missing_file(tmp_path):
	script_path = tmp_path / "missing.txt"

	with pytest.raises(FileNotFoundError):
		process_script(str(script_path))


def test_process_script_rejects_unsupported_file_extension(tmp_path):
	script_path = tmp_path / "script.pdf"
	script_path.write_text("Alice: Hello!", encoding="utf-8")

	with pytest.raises(ValueError):
		process_script(str(script_path))