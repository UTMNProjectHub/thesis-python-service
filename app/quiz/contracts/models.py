from __future__ import annotations

from typing import List, Optional, Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field

# === Question type enum ===
QuestionType = Literal[
    "multichoice",
    "single_choice",
    "open",
    "truefalse",
    "shortanswer",
    "matching",
    "essay",
    "numerical",
    "description",
]

# === Request Models ===
class SolveQuestionParams(BaseModel):
    id: UUID = Field(..., description="ID вопроса (uuid)")

class SolveQuestionBody(BaseModel):
    answerIds: Optional[List[UUID]] = None
    answerText: Optional[str] = None
    quizId: UUID

# === Base Models ===
class QuestionModel(BaseModel):
    id: UUID
    type: str  # как в TS: просто строка, не строго QuestionType
    multiAnswer: Optional[bool] = None
    text: str

class VariantModel(BaseModel):
    id: UUID
    text: str
    explainRight: str
    explainWrong: str
    isRight: bool
    questionId: UUID
    variantId: UUID
    questionsVariantsId: UUID

class MatchingLeftItemModel(BaseModel):
    id: UUID
    text: str

class MatchingRightItemModel(BaseModel):
    id: UUID
    text: str

class MatchingCorrectPairModel(BaseModel):
    leftVariantId: UUID
    rightVariantId: UUID
    explainRight: Optional[str] = None
    explainWrong: Optional[str] = None

class MatchingConfigModel(BaseModel):
    leftItems: List[MatchingLeftItemModel]
    rightItems: List[MatchingRightItemModel]
    correctPairs: List[MatchingCorrectPairModel]

class SubmittedVariantResponse(BaseModel):
    variantId: UUID
    variantText: str
    isRight: bool
    explanation: str

class MatchingPairResponse(BaseModel):
    key: str
    value: str
    isRight: bool
    explanation: Optional[str] = None

class ChosenVariantModel(BaseModel):
    id: UUID
    userId: UUID
    quizId: UUID
    questionId: UUID
    chosenId: Optional[UUID] = None
    answer: Optional[Any] = None
    isRight: Optional[bool] = None

# === Response Models ===
class SolveQuestionVariantsResponse(BaseModel):
    question: QuestionModel
    submittedVariants: List[SubmittedVariantResponse]
    allVariants: List[VariantModel]

class SolveQuestionMatchingResponse(BaseModel):
    question: QuestionModel
    submittedAnswer: ChosenVariantModel
    isRight: Optional[bool] = None
    pairs: List[MatchingPairResponse]
    variants: List[VariantModel]
    explanation: Optional[str] = None  # Исправлено: Optional[str], а не Optional[Optional[str]]

class SolveQuestionTextResponse(BaseModel):
    question: QuestionModel
    submittedAnswer: ChosenVariantModel
    isRight: Optional[bool] = None
    explanation: Optional[str] = None
    variants: List[VariantModel]
    pairs: Optional[List[MatchingPairResponse]] = None

SolveQuestionTextResponseUnion = SolveQuestionMatchingResponse | SolveQuestionTextResponse

# === Error Response ===
class ErrorResponse(BaseModel):
    detail: str

# === Update Models ===
class UpdateQuestionBody(BaseModel):
    text: Optional[str] = None
    type: Optional[QuestionType] = None
    multiAnswer: Optional[bool] = None  # Упрощено: Optional[bool], null трактуется как не изменено

class UpdateQuestionVariant(BaseModel):
    text: str
    explainRight: str
    explainWrong: str
    isRight: bool

class UpdateQuestionVariantsBody(BaseModel):
    variants: List[UpdateQuestionVariant]

class UpdateQuestionMatchingConfigBody(BaseModel):
    matchingConfig: MatchingConfigModel

# === Вспомогательная модель для генерации (наш внутренний пакет) ===
class GeneratedQuestionBundle(BaseModel):
    question: QuestionModel
    variants: List[VariantModel] = Field(default_factory=list)
    matchingConfig: Optional[MatchingConfigModel] = None
    questionType: QuestionType

def new_uuid() -> UUID:
    return uuid4()

class ExplainQuizRequest(BaseModel):
    quizId: Optional[UUID] = None
    questions: List[GeneratedQuestionBundle]
    text: str = Field(..., description="Исходный текст лекции/конспекта для RAG")
    difficulty: Literal["easy", "medium", "hard"] = "medium"

class ExplainQuizResponse(BaseModel):
    quizId: UUID
    questions: List[GeneratedQuestionBundle]
    markdown: str
