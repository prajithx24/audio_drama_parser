
from __future__ import annotations

import os
import re
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai.errors import APIError, ClientError, ServerError
from google.genai._gaos.errors.createinteraction import (
    CreateInteractionClientError,
    CreateInteractionServerError,
)
from google.genai._gaos.lib.compat_errors import (
    InternalServerError as CompatInternalServerError,
    RateLimitError as CompatRateLimitError,
)
from pydantic import BaseModel
from pydantic import ValidationError

from .models import Evidence, ExtractionResult

DEFAULT_MODEL_NAME = "gemini-3.5-flash-lite"
MAX_ATTEMPTS = 3
RETRY_BASE_DELAY_SECONDS = 1.0
MODEL_INPUT_TOKEN_LIMIT = 1_048_576
MODEL_OUTPUT_TOKEN_LIMIT = 65_536


class ExtractionError(Exception):
    pass

class MissingAPIKeyError(ExtractionError):
    pass

class EmptyScriptError(ExtractionError):
    pass

class ExtractionCompletenessError(ExtractionError):
    pass

class ExtractionAPIError(ExtractionError):
    pass

class ExtractionResponseError(ExtractionError):
    pass

class ExtractionValidationError(ExtractionError):
    pass

class EvidenceValidationError(ExtractionError):
    pass

SYSTEM_INSTRUCTION = """\
You extract characters and scenes from natural-language audio drama scripts.
Return only one JSON object matching the supplied schema. Do not return Markdown
or explanatory prose.

Extract every distinct character, including speaking characters, non-speaking
characters, people mentioned in narration or dialogue, people addressed, and
unnamed references such as "the guard" or "a voice on the radio". Do not merge
characters because their names are similar unless the script establishes that
they are the same person.

Distinguish spoken dialogue, narration, internal thoughts, and actions. Use
source_type values exactly as dialogue, narration, internal_thought, or action.
Classify characters with narrative_status values active_character,
mentioned_person, or uncertain_reference. Use identity_status confirmed or
ambiguous. Record relationships only when supported and classify their
certainty as explicit, inferred, or uncertain.

Segment scenes using meaningful contextual changes in location, time, action,
character grouping, or dramatic situation, not paragraph boundaries. Record
meaningful events and scene-boundary evidence. Use null for unknown optional
values and empty arrays when no supported items exist.

Never invent descriptions, roles, locations, relationships, events, or
evidence. Every evidence.text value must be copied verbatim from the script.
Populate characters and scenes whenever the script contains identifiable
people, actions, or dramatic situations. Warnings are only for uncertainty or
ambiguity; do not put extracted entities only in warnings. Treat hypothetical
identities, disguises, nicknames, and personas as separate references unless
the script establishes that they are actual characters. Extract meaningful
scenes and events within them rather than collapsing a full episode into one
undifferentiated result. Include warnings for ambiguous identities and
uncertain scene boundaries. Do not generate character IDs.
"""


def _build_prompt(script_text: str) -> str:
    return f"""Extract characters and scenes from this complete audio drama script.
Analyze the entire script before returning the JSON result. Include exact
verbatim evidence for extracted facts and preserve the distinction between
dialogue, narration, internal thoughts, and actions. Extract every person
explicitly mentioned in dialogue or narration who speaks, acts, appears, or is
meaningfully referenced. Populate the characters and scenes arrays when the
script supports them; reserve warnings for uncertainty and ambiguity. Do not
invent unsupported information.

SCRIPT START
{script_text}
SCRIPT END
"""


def _iter_evidence(value: object, location: str = "result"):
    if isinstance(value, Evidence):
        yield location, value
        return
    if isinstance(value, BaseModel):
        for field_name in type(value).model_fields:
            yield from _iter_evidence(
                getattr(value, field_name), f"{location}.{field_name}"
            )
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _iter_evidence(item, f"{location}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _iter_evidence(item, f"{location}[{key!r}]")


def _validate_evidence(result: ExtractionResult, script_text: str) -> None:
    normalized_script = re.sub(r"\s+", " ", script_text).strip()
    for location, evidence in _iter_evidence(result):
        normalized_evidence = re.sub(r"\s+", " ", evidence.text).strip()
        evidence_candidates = [normalized_evidence]
        if (
            len(normalized_evidence) >= 2
            and normalized_evidence[0] == normalized_evidence[-1]
            and normalized_evidence[0] in {'"', "'"}
        ):
            evidence_candidates.append(normalized_evidence[1:-1].strip())
        if not any(candidate in normalized_script for candidate in evidence_candidates):
            raise EvidenceValidationError(
                f"Evidence at {location} was not found in the supplied script."
            )


def _validate_completeness(
    result: ExtractionResult, script_text: str
) -> None:
    if script_text.strip() and not result.characters and not result.scenes:
        raise ExtractionCompletenessError(
            "Gemini returned no characters or scenes for a non-empty script."
        )


def _load_api_key() -> str:
    project_root = Path(__file__).resolve().parents[2]
    load_dotenv(project_root / ".env")
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key or not api_key.strip():
        raise MissingAPIKeyError("GEMINI_API_KEY is not set.")
    return api_key


def _is_retryable_server_error(error: ServerError) -> bool:
    status_code = getattr(error, "code", getattr(error, "status_code", None))
    return isinstance(status_code, int) and status_code >= 500


def _error_status_code(error: object) -> int | None:
    status_code = getattr(error, "code", getattr(error, "status_code", None))
    return status_code if isinstance(status_code, int) else None


def _retry_after_seconds(error: object) -> float | None:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    value = headers.get("Retry-After") if headers is not None else None
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        try:
            retry_at = parsedate_to_datetime(value)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


def _retry_delay(attempt: int, error: object) -> float:
    if _error_status_code(error) == 429:
        retry_after = _retry_after_seconds(error)
        if retry_after is not None:
            return retry_after
    return RETRY_BASE_DELAY_SECONDS * (2 ** (attempt - 1))


def _create_interaction(
    client: genai.Client,
    *,
    model_name: str,
    prompt: str,
) -> object:
    response_format = {
        "type": "text",
        "mime_type": "application/json",
        "schema": ExtractionResult.model_json_schema(),
    }

    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return client.interactions.create(
                model=model_name,
                input=prompt,
                system_instruction=SYSTEM_INSTRUCTION,
                response_format=response_format,
            )
        except CreateInteractionServerError as error:
            if _error_status_code(error) is None or (
                _error_status_code(error) < 500 or attempt == MAX_ATTEMPTS
            ):
                raise ExtractionAPIError(
                    f"Gemini server request failed after {attempt} attempt(s)."
                ) from error
            time.sleep(_retry_delay(attempt, error))
        except CreateInteractionClientError as error:
            if _error_status_code(error) == 429 and attempt < MAX_ATTEMPTS:
                time.sleep(_retry_delay(attempt, error))
                continue
            raise ExtractionAPIError("Gemini client request was rejected.") from error
        except CompatInternalServerError as error:
            if attempt == MAX_ATTEMPTS:
                raise ExtractionAPIError(
                    f"Gemini server request failed after {attempt} attempt(s)."
                ) from error
            time.sleep(_retry_delay(attempt, error))
        except CompatRateLimitError as error:
            if attempt == MAX_ATTEMPTS:
                raise ExtractionAPIError(
                    "Gemini rate limit persisted after the retry limit."
                ) from error
            time.sleep(_retry_delay(attempt, error))
        except ServerError as error:
            if not _is_retryable_server_error(error) or attempt == MAX_ATTEMPTS:
                raise ExtractionAPIError(
                    f"Gemini server request failed after {attempt} attempt(s)."
                ) from error
            time.sleep(_retry_delay(attempt, error))
        except ClientError as error:
            if error.code == 429 and attempt < MAX_ATTEMPTS:
                time.sleep(_retry_delay(attempt, error))
                continue
            raise ExtractionAPIError("Gemini client request was rejected.") from error
        except APIError as error:
            raise ExtractionAPIError("Gemini API request failed.") from error

    raise ExtractionAPIError("Gemini request did not complete.")


def extract_characters_and_scenes(
    script_text: str,
    *,
    model_name: str = DEFAULT_MODEL_NAME,
    client: genai.Client | None = None,
) -> ExtractionResult:
    if not script_text or not script_text.strip():
        raise EmptyScriptError("script_text must not be empty.")

    if client is None:
        client = genai.Client(api_key=_load_api_key())

    interaction = _create_interaction(
        client,
        model_name=model_name,
        prompt=_build_prompt(script_text),
    )
    status = getattr(interaction, "status", None)
    if status != "completed":
        raise ExtractionResponseError(
            f"Gemini interaction did not complete successfully (status={status!r})."
        )

    output_text = getattr(interaction, "output_text", None)
    if not isinstance(output_text, str) or not output_text.strip():
        raise ExtractionResponseError("Gemini returned an empty response.")

    try:
        result = ExtractionResult.model_validate_json(output_text)
    except (ValidationError, ValueError) as error:
        raise ExtractionValidationError(
            "Gemini response did not match the extraction schema."
        ) from error

    _validate_evidence(result, script_text)
    _validate_completeness(result, script_text)
    return result.model_copy(update={"model_name": model_name})


__all__ = [
    "EvidenceValidationError",
    "EmptyScriptError",
    "ExtractionAPIError",
    "ExtractionCompletenessError",
    "ExtractionError",
    "ExtractionResponseError",
    "ExtractionValidationError",
    "MissingAPIKeyError",
    "extract_characters_and_scenes",
]
