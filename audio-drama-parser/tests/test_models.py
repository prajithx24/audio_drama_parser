import pytest
from pydantic import ValidationError

from audio_drama_parser.models import (
    Character,
    Evidence,
    ExtractionResult,
    Scene,
    SceneEvent,
)


def test_valid_character_creation():
    evidence = Evidence(
        text="Alice enters the room.",
        fact="Alice is present in the scene.",
        certainty="explicit",
        source_type="action",
    )

    character = Character(
        name="Alice",
        aliases=["Al"],
        description="A determined investigator.",
        role_in_story="Protagonist",
        evidence=[evidence],
        confidence="high",
        identity_status="confirmed",
        narrative_status="active_character",
    )

    assert character.name == "Alice"
    assert character.aliases == ["Al"]
    assert character.evidence[0] == evidence


def test_valid_scene_creation():
    event = SceneEvent(
        description="Alice opens the locked door.",
        characters_involved=["Alice"],
    )
    scene = Scene(
        location="An abandoned station",
        time_reference="At midnight",
        characters_present=["Alice"],
        summary="Alice investigates an abandoned station.",
        events=[event],
        confidence="medium",
    )

    assert scene.location == "An abandoned station"
    assert scene.events[0].description == "Alice opens the locked door."


def test_valid_extraction_result_creation():
    result = ExtractionResult(
        characters=[
            Character(
                name="Narrator",
                confidence="high",
                identity_status="confirmed",
                narrative_status="active_character",
            )
        ],
        scenes=[
            Scene(
                summary="The narrator introduces the setting.",
                confidence="high",
            )
        ],
        warnings=["The narrator's identity is not specified."],
        model_name="gemini-3.5-flash-lite",
    )

    assert result.characters[0].name == "Narrator"
    assert result.scenes[0].summary == "The narrator introduces the setting."
    assert result.model_name == "gemini-3.5-flash-lite"


def test_invalid_confidence_is_rejected():
    with pytest.raises(ValidationError):
        Character(
            name="Alice",
            confidence="certain",
            identity_status="confirmed",
            narrative_status="active_character",
        )


def test_unknown_optional_descriptions_and_locations_are_none():
    character = Character(
        name="Unknown voice",
        confidence="low",
        identity_status="ambiguous",
        narrative_status="uncertain_reference",
    )
    scene = Scene(summary="The setting is not identified.", confidence="low")

    assert character.description is None
    assert character.role_in_story is None
    assert scene.location is None
    assert scene.time_reference is None


@pytest.mark.parametrize(
    "source_type",
    ["dialogue", "narration", "internal_thought", "action"],
)
def test_valid_evidence_source_types(source_type):
    evidence = Evidence(
        text="Alice speaks.",
        fact="Alice is speaking.",
        certainty="explicit",
        source_type=source_type,
    )

    assert evidence.source_type == source_type


def test_invalid_evidence_source_type_is_rejected():
    with pytest.raises(ValidationError):
        Evidence(
            text="Alice speaks.",
            fact="Alice is speaking.",
            certainty="explicit",
            source_type="description",
        )


@pytest.mark.parametrize(
    "narrative_status",
    ["active_character", "mentioned_person", "uncertain_reference"],
)
def test_valid_character_narrative_statuses(narrative_status):
    character = Character(
        name="Alice",
        confidence="medium",
        identity_status="confirmed",
        narrative_status=narrative_status,
    )

    assert character.narrative_status == narrative_status


def test_invalid_character_narrative_status_is_rejected():
    with pytest.raises(ValidationError):
        Character(
            name="Alice",
            confidence="medium",
            identity_status="confirmed",
            narrative_status="background_only",
        )
