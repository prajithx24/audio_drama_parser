
from typing import Literal, Optional

from pydantic import BaseModel, Field


Certainty = Literal["explicit", "inferred", "uncertain"]
Confidence = Literal["high", "medium", "low"]


class Evidence(BaseModel):

    text: str = Field(..., min_length=1, description="Exact supporting passage from the script.")
    fact: str = Field(..., min_length=1, description="What the passage supports.")
    certainty: Certainty = Field(description="How directly the passage supports the fact.")
    source_type: Literal[
        "dialogue", "narration", "internal_thought", "action"
    ] = Field(description="Origin of the supporting passage.")


class Relationship(BaseModel):

    related_character: str = Field(
        ..., min_length=1, description="Character name or temporary character reference."
    )
    relationship_type: str = Field(..., min_length=1, description="Description of the relationship.")
    evidence: list[Evidence] = Field(
        default_factory=list, description="Evidence supporting this relationship."
    )
    certainty: Certainty = Field(description="How directly the relationship is supported.")


class Character(BaseModel):

    name: str = Field(..., min_length=1, description="Primary character name.")
    aliases: list[str] = Field(
        default_factory=list, description="Alternative names or references."
    )
    description: Optional[str] = Field(
        default=None, description="Description only when supported by the script."
    )
    role_in_story: Optional[str] = Field(
        default=None, description="Character's role in the story when supported."
    )
    relationships: list[Relationship] = Field(
        default_factory=list, description="Relationships with other characters."
    )
    evidence: list[Evidence] = Field(
        default_factory=list, description="Evidence supporting this character extraction."
    )
    confidence: Confidence = Field(description="Confidence in the character extraction.")
    identity_status: Literal["confirmed", "ambiguous"] = Field(
        description="Whether the character identity is confirmed or ambiguous."
    )
    narrative_status: Literal[
        "active_character", "mentioned_person", "uncertain_reference"
    ] = Field(description="How the character participates in the story.")


class SceneEvent(BaseModel):

    description: str = Field(..., min_length=1, description="Event or action in the scene.")
    characters_involved: list[str] = Field(
        default_factory=list, description="Characters involved in the event."
    )
    evidence: list[Evidence] = Field(
        default_factory=list, description="Evidence supporting the event."
    )


class Scene(BaseModel):

    location: Optional[str] = Field(default=None, description="Scene location when known.")
    time_reference: Optional[str] = Field(
        default=None, description="Time reference when stated or inferable."
    )
    characters_present: list[str] = Field(
        default_factory=list, description="Characters present in the scene."
    )
    summary: str = Field(..., min_length=1, description="Short summary of the scene.")
    events: list[SceneEvent] = Field(
        default_factory=list, description="Events occurring in the scene."
    )
    boundary_evidence: list[Evidence] = Field(
        default_factory=list, description="Evidence supporting the scene boundaries."
    )
    confidence: Confidence = Field(description="Confidence in the scene extraction.")


class ExtractionResult(BaseModel):

    characters: list[Character] = Field(
        default_factory=list, description="Characters extracted from the script."
    )
    scenes: list[Scene] = Field(default_factory=list, description="Scenes extracted from the script.")
    warnings: list[str] = Field(
        default_factory=list, description="Warnings produced during extraction."
    )
    model_name: Optional[str] = Field(
        default=None, description="Model used to produce the extraction."
    )
