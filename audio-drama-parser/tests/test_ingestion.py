import pytest

from audio_drama_parser.ingestion import load_script


def test_load_script_reads_utf8_text(tmp_path):
	script_path = tmp_path / "script.txt"
	expected = "Narrator: Café\nAlice: Hello"
	script_path.write_bytes(expected.encode("utf-8"))

	assert load_script(str(script_path)) == expected


def test_load_script_reads_utf8_bom(tmp_path):
	script_path = tmp_path / "script.txt"
	script_path.write_bytes(b"\xef\xbb\xbfNarrator: Hello")

	assert load_script(str(script_path)) == "Narrator: Hello"


def test_load_script_preserves_original_text_and_line_breaks(tmp_path):
	script_path = tmp_path / "script.txt"
	expected = "Scene one\r\nAlice: Hello\n\nBob: Goodbye\r"
	script_path.write_bytes(expected.encode("utf-8"))

	assert load_script(str(script_path)) == expected


def test_load_script_raises_for_missing_file(tmp_path):
	script_path = tmp_path / "missing.txt"

	with pytest.raises(FileNotFoundError):
		load_script(str(script_path))


def test_load_script_rejects_unsupported_extension(tmp_path):
	script_path = tmp_path / "script.pdf"
	script_path.write_bytes(b"not a text script")

	with pytest.raises(ValueError):
		load_script(str(script_path))


def test_load_script_rejects_invalid_utf8(tmp_path):
	script_path = tmp_path / "script.txt"
	script_path.write_bytes(b"Valid text\xff")

	with pytest.raises(UnicodeDecodeError):
		load_script(str(script_path))


def test_load_script_accepts_uppercase_txt_extension(tmp_path):
	script_path = tmp_path / "script.TXT"
	expected = "Narrator: Hello"
	script_path.write_text(expected, encoding="utf-8")

	assert load_script(str(script_path)) == expected