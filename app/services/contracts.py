from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator


TaskStatus = Literal["SUCCESS", "FAILED"]
Difficulty = Literal["easy", "medium", "hard"]
QuestionType = Literal[
    "multichoice",
    "essay",
    "matching",
    "truefalse",
    "shortanswer",
    "numerical",
]


class RabbitContractModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class QuizGenRequest(RabbitContractModel):
    quiz_id: UUID = Field(alias="quizId")
    user_id: UUID = Field(alias="userId")
    files: list[UUID]
    summary_id: int = Field(alias="summaryId")
    difficulty: Difficulty = "medium"
    question_count: int
    question_types: list[QuestionType]
    additional_requirements: str = ""

    @field_validator("files")
    @classmethod
    def files_must_not_be_empty(cls, value: list[UUID]) -> list[UUID]:
        if not value:
            raise ValueError("files must contain at least one file id")
        return value

    @field_validator("question_count")
    @classmethod
    def question_count_must_be_positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("question_count must be greater than 0")
        return value

    @field_validator("question_types")
    @classmethod
    def question_types_must_not_be_empty(cls, value: list[QuestionType]) -> list[QuestionType]:
        if not value:
            raise ValueError("question_types must contain at least one type")
        return value


class SummaryGenRequest(RabbitContractModel):
    summary_id: UUID = Field(alias="summaryId")
    subject_id: int = Field(alias="subjectId")
    theme_id: int = Field(alias="themeId")
    user_id: UUID = Field(alias="userId")
    files: list[UUID]
    additional_requirements: str = ""

    @field_validator("files")
    @classmethod
    def files_must_not_be_empty(cls, value: list[UUID]) -> list[UUID]:
        if not value:
            raise ValueError("files must contain at least one file id")
        return value


class QuizGenComplete(RabbitContractModel):
    quiz_id: UUID | str = Field(alias="quizId")
    user_id: UUID | str = Field(alias="userId")
    status: TaskStatus
    error: str = ""


class SummaryGenComplete(RabbitContractModel):
    summary_id: UUID | str = Field(alias="summaryId")
    subject_id: int | None = Field(default=None, alias="subjectId")
    theme_id: int | None = Field(default=None, alias="themeId")
    user_id: UUID | str = Field(alias="userId")
    status: TaskStatus
    error: str = ""


class QuizAnswerDialogRequest(RabbitContractModel):
    dialog_id: UUID = Field(alias="dialogId")
    user_id: UUID = Field(alias="userId")
    message_id: UUID = Field(alias="messageId")


class QuizAnswerDialogResponse(RabbitContractModel):
    status: TaskStatus
    dialog_id: UUID | str = Field(alias="dialogId")
    user_id: UUID | str = Field(alias="userId")
    error: str | None = None


def to_payload(model: RabbitContractModel) -> dict:
    return model.model_dump(by_alias=True, mode="json")
