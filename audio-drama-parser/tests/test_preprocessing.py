import pytest

from audio_drama_parser.preprocessing import preprocess_text


@pytest.mark.parametrize(
	("text", "expected"),
	[
		("", ""),
		("  Hello   world  ", "Hello world"),
		("Alice: Hello\r\nBob: Hi\r\n", "Alice: Hello\nBob: Hi"),
		("Scene one\n\n\n\nScene two", "Scene one\n\nScene two"),
		("Narrator: The door creaked open.", "Narrator: The door creaked open."),
		(" \t\r\n  \n\t ", ""),
	],
)
def test_preprocess_text(text, expected):
	assert preprocess_text(text) == expected
