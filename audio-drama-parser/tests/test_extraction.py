import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from google.genai.errors import ClientError, ServerError
from google.genai._gaos.errors.createinteraction import (
    CreateInteractionServerError,
    CreateInteractionServerErrorData,
)
from google.genai._gaos.lib.compat_errors import InternalServerError as CompatInternalServerError
import httpx

import audio_drama_parser.extraction as extraction
from audio_drama_parser.extraction import (
    EvidenceValidationError,
    EmptyScriptError,
    ExtractionAPIError,
    ExtractionCompletenessError,
    ExtractionResponseError,
    ExtractionValidationError,
    MissingAPIKeyError,
    extract_characters_and_scenes,
)


SCRIPT = "Alice enters the room.\nAlice: I found the key.\nBob is her brother."


def _payload(**overrides):
    payload = {
        "characters": [
            {
                "name": "Alice",
                "aliases": [],
                "description": None,
                "role_in_story": None,
                "relationships": [],
                "evidence": [
                    {
                        "text": "Alice enters the room.",
                        "fact": "Alice enters the room.",
                        "certainty": "explicit",
                        "source_type": "action",
                    }
                ],
                "confidence": "high",
                "identity_status": "confirmed",
                "narrative_status": "active_character",
            }
        ],
        "scenes": [
            {
                "location": None,
                "time_reference": None,
                "characters_present": ["Alice"],
                "summary": "Alice enters and searches for a key.",
                "events": [],
                "boundary_evidence": [],
                "confidence": "high",
            }
        ],
        "warnings": [],
    }
    payload.update(overrides)
    return payload


def _client(output=None, status="completed", side_effect=None):
    create = Mock(
        return_value=SimpleNamespace(status=status, output_text=output),
        side_effect=side_effect,
    )
    client = Mock()
    client.interactions.create = create
    return client


def test_successful_extraction_assigns_model_name():
    client = _client(json.dumps(_payload()))

    result = extract_characters_and_scenes(SCRIPT, client=client)

    assert result.model_name == "gemini-3.5-flash-lite"
    assert result.characters[0].name == "Alice"


@pytest.mark.parametrize(
    ("characters", "scenes"),
    [
        ([], []),
    ],
)
def test_both_empty_extraction_is_rejected(characters, scenes):
    payload = _payload(characters=characters, scenes=scenes)
    client = _client(json.dumps(payload))

    with pytest.raises(ExtractionCompletenessError):
        extract_characters_and_scenes(SCRIPT, client=client)


def test_characters_empty_but_scenes_populated_is_allowed():
    payload = _payload(characters=[])

    result = extract_characters_and_scenes(
        SCRIPT,
        client=_client(json.dumps(payload)),
    )

    assert result.characters == []
    assert len(result.scenes) == 1


def test_scenes_empty_but_characters_populated_is_allowed():
    payload = _payload(scenes=[])

    result = extract_characters_and_scenes(
        SCRIPT,
        client=_client(json.dumps(payload)),
    )

    assert len(result.characters) == 1
    assert result.scenes == []


def test_both_populated_extraction_is_allowed():
    result = extract_characters_and_scenes(
        SCRIPT,
        client=_client(json.dumps(_payload())),
    )

    assert result.characters
    assert result.scenes


def test_empty_extraction_does_not_retry():
    client = _client(json.dumps(_payload(characters=[], scenes=[])))

    with pytest.raises(ExtractionCompletenessError):
        extract_characters_and_scenes(SCRIPT, client=client)

    client.interactions.create.assert_called_once()


def test_empty_script_is_rejected_before_api_call():
    client = _client()

    with pytest.raises(EmptyScriptError, match="must not be empty"):
        extract_characters_and_scenes("  ", client=client)

    client.interactions.create.assert_not_called()


def test_missing_api_key(monkeypatch):
    monkeypatch.setattr(extraction, "load_dotenv", lambda *_args, **_kwargs: None)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(MissingAPIKeyError):
        extract_characters_and_scenes(SCRIPT)


def test_successful_response_uses_model_and_structured_schema():
    client = _client(json.dumps(_payload()))

    extract_characters_and_scenes(SCRIPT, model_name="custom-model", client=client)

    request = client.interactions.create.call_args.kwargs
    assert request["model"] == "custom-model"
    assert request["response_format"]["type"] == "text"
    assert request["response_format"]["mime_type"] == "application/json"
    assert request["response_format"]["schema"] == extraction.ExtractionResult.model_json_schema()


def test_server_error_retries_and_succeeds(monkeypatch):
    error = ServerError(503, {"error": {"message": "busy"}})
    client = _client(
        side_effect=[error, SimpleNamespace(status="completed", output_text=json.dumps(_payload()))]
    )
    sleep = Mock()
    monkeypatch.setattr(extraction.time, "sleep", sleep)

    result = extract_characters_and_scenes(SCRIPT, client=client)

    assert result.characters[0].name == "Alice"
    assert client.interactions.create.call_count == 2
    sleep.assert_called_once_with(1.0)


def test_interactions_server_error_is_wrapped_after_retries(monkeypatch):
    response = httpx.Response(
        503,
        request=httpx.Request("POST", "https://example.test"),
        json={"error": {"message": "busy"}},
    )
    error = CreateInteractionServerError(
        CreateInteractionServerErrorData.model_validate(
            {"error": {"message": "busy"}}
        ),
        response,
    )
    client = _client(side_effect=error)
    sleep = Mock()
    monkeypatch.setattr(extraction.time, "sleep", sleep)

    with pytest.raises(ExtractionAPIError, match="after 3 attempt"):
        extract_characters_and_scenes(SCRIPT, client=client)

    assert client.interactions.create.call_count == extraction.MAX_ATTEMPTS
    assert sleep.call_count == extraction.MAX_ATTEMPTS - 1


def test_wrapped_interactions_server_error_is_retried_and_wrapped(monkeypatch):
    response = httpx.Response(
        503,
        request=httpx.Request("POST", "https://example.test"),
        json={"error": {"message": "busy"}},
    )
    error = CompatInternalServerError(
        "busy",
        response=response,
        body={"error": {"message": "busy"}},
    )
    client = _client(side_effect=error)
    sleep = Mock()
    monkeypatch.setattr(extraction.time, "sleep", sleep)

    with pytest.raises(ExtractionAPIError, match="after 3 attempt"):
        extract_characters_and_scenes(SCRIPT, client=client)

    assert client.interactions.create.call_count == extraction.MAX_ATTEMPTS
    assert sleep.call_count == extraction.MAX_ATTEMPTS - 1


def test_client_error_is_not_retried(monkeypatch):
    error = ClientError(400, {"error": {"message": "invalid"}})
    client = _client(side_effect=error)
    sleep = Mock()
    monkeypatch.setattr(extraction.time, "sleep", sleep)

    with pytest.raises(ExtractionAPIError):
        extract_characters_and_scenes(SCRIPT, client=client)

    client.interactions.create.assert_called_once()
    sleep.assert_not_called()


def test_rate_limit_error_retries_and_respects_retry_after(monkeypatch):
    error = ClientError(429, {"error": {"message": "rate limited"}})
    error.response = SimpleNamespace(headers={"Retry-After": "7"})
    client = _client(
        side_effect=[
            error,
            SimpleNamespace(status="completed", output_text=json.dumps(_payload())),
        ]
    )
    sleep = Mock()
    monkeypatch.setattr(extraction.time, "sleep", sleep)

    result = extract_characters_and_scenes(SCRIPT, client=client)

    assert result.characters[0].name == "Alice"
    assert client.interactions.create.call_count == 2
    sleep.assert_called_once_with(7.0)


def test_rate_limit_retry_is_bounded(monkeypatch):
    error = ClientError(429, {"error": {"message": "rate limited"}})
    client = _client(side_effect=error)
    sleep = Mock()
    monkeypatch.setattr(extraction.time, "sleep", sleep)

    with pytest.raises(ExtractionAPIError):
        extract_characters_and_scenes(SCRIPT, client=client)

    assert client.interactions.create.call_count == extraction.MAX_ATTEMPTS
    assert sleep.call_count == extraction.MAX_ATTEMPTS - 1


def test_retry_limit_is_bounded(monkeypatch):
    error = ServerError(503, {"error": {"message": "busy"}})
    client = _client(side_effect=error)
    sleep = Mock()
    monkeypatch.setattr(extraction.time, "sleep", sleep)

    with pytest.raises(ExtractionAPIError):
        extract_characters_and_scenes(SCRIPT, client=client)

    assert client.interactions.create.call_count == extraction.MAX_ATTEMPTS
    assert sleep.call_count == extraction.MAX_ATTEMPTS - 1


def test_malformed_json_is_rejected():
    client = _client("{not json")

    with pytest.raises(ExtractionValidationError):
        extract_characters_and_scenes(SCRIPT, client=client)


def test_invalid_pydantic_output_is_rejected():
    payload = _payload()
    payload["characters"][0]["confidence"] = "certain"
    client = _client(json.dumps(payload))

    with pytest.raises(ExtractionValidationError):
        extract_characters_and_scenes(SCRIPT, client=client)


def test_empty_response_is_rejected():
    client = _client("")

    with pytest.raises(ExtractionResponseError):
        extract_characters_and_scenes(SCRIPT, client=client)


def test_incomplete_interaction_is_rejected():
    client = _client(json.dumps(_payload()), status="incomplete")

    with pytest.raises(ExtractionResponseError, match="incomplete"):
        extract_characters_and_scenes(SCRIPT, client=client)


def test_invalid_character_evidence_is_rejected():
    payload = _payload()
    payload["characters"][0]["evidence"][0]["text"] = "Not in script"

    with pytest.raises(EvidenceValidationError, match="characters"):
        extract_characters_and_scenes(SCRIPT, client=_client(json.dumps(payload)))


def test_valid_nested_evidence_is_accepted():
    payload = _payload()
    evidence = {
        "text": "Bob is her brother.",
        "fact": "Bob is Alice's brother.",
        "certainty": "explicit",
        "source_type": "narration",
    }
    payload["characters"][0]["relationships"] = [
        {
            "related_character": "Bob",
            "relationship_type": "brother",
            "evidence": [evidence],
            "certainty": "explicit",
        }
    ]
    payload["scenes"][0]["boundary_evidence"] = [evidence]
    payload["scenes"][0]["events"] = [
        {
            "description": "Alice enters.",
            "characters_involved": ["Alice"],
            "evidence": [
                {
                    "text": "Alice enters the room.",
                    "fact": "Alice enters the room.",
                    "certainty": "explicit",
                    "source_type": "action",
                }
            ],
        }
    ]

    result = extract_characters_and_scenes(
        SCRIPT,
        client=_client(json.dumps(payload)),
    )

    assert result.characters[0].relationships[0].evidence[0].text == "Bob is her brother."


def test_evidence_joining_source_lines_is_accepted():
    payload = _payload()
    payload["characters"][0]["evidence"][0]["text"] = (
        "Alice enters the room. Bob is her brother."
    )

    result = extract_characters_and_scenes(
        "Alice enters the room.\nBob is her brother.",
        client=_client(json.dumps(payload)),
    )

    assert result.characters[0].evidence[0].text == (
        "Alice enters the room. Bob is her brother."
    )


def test_partial_dialogue_quote_is_accepted():
    payload = _payload()
    payload["characters"][0]["evidence"][0]["text"] = '"Alice enters the room."'

    result = extract_characters_and_scenes(
        SCRIPT,
        client=_client(json.dumps(payload)),
    )

    assert result.characters[0].evidence[0].text == '"Alice enters the room."'


@pytest.mark.parametrize(
    "field",
    ["relationships", "boundary_evidence", "events"],
)
def test_nested_evidence_not_found_is_rejected(field):
    payload = _payload()
    invalid_evidence = {
        "text": "Missing quote",
        "fact": "Unsupported fact.",
        "certainty": "uncertain",
        "source_type": "narration",
    }
    if field == "relationships":
        payload["characters"][0]["relationships"] = [
            {
                "related_character": "Bob",
                "relationship_type": "relative",
                "evidence": [invalid_evidence],
                "certainty": "uncertain",
            }
        ]
    elif field == "boundary_evidence":
        payload["scenes"][0]["boundary_evidence"] = [invalid_evidence]
    else:
        payload["scenes"][0]["events"] = [
            {
                "description": "An unsupported event.",
                "characters_involved": [],
                "evidence": [invalid_evidence],
            }
        ]

    with pytest.raises(EvidenceValidationError):
        extract_characters_and_scenes(SCRIPT, client=_client(json.dumps(payload)))
