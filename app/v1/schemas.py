from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from uuid import UUID

from app.quiz.contracts.models import (
    QuestionModel,
    VariantModel,
    MatchingConfigModel,
)


class CompletionRequest(BaseModel):
    text: str = Field(..., description="Исходный текст пользователя")
    user_prompt: str = Field(..., description="User prompt")
    system_prompt: str = Field("", description="System prompt")


class CompletionResponse(BaseModel):
    model: str
    content: str


class TopicText(BaseModel):
    topic: str
    text: str


class SimilarityRequest(BaseModel):
    paths: list[str]


class PairSimilarityRequest(BaseModel):
    a: str
    b: str


class TopicPairRequest(BaseModel):
    topic_a: str
    topic_b: str
    json_path: str = "data/texts.json"


class GeneratedQuestionDTO(BaseModel):
    question: QuestionModel
    variants: List[VariantModel]
    matchingConfig: Optional[MatchingConfigModel] = None


class QuizGenerationRequest(BaseModel):
    topic: str
    difficulty: Literal["easy", "medium", "hard"] = "medium"
    language: str = "ru"
    number_of_questions: int = 10


class QuizGenerationResponse(BaseModel):
    quizId: UUID
    questions: List[GeneratedQuestionDTO]


class FAQGenerationRequest(BaseModel):
    file_path: str = Field(..., description="Путь к файлу (MD, PDF, DOCX)")
    title: Optional[str] = None
    num_questions: int = 10
    detail_level: Literal["low", "medium", "high"] = "medium"
    language: str = "ru"


class FAQItemDTO(BaseModel):
    question: str
    answer: str
    category: Optional[str] = None


class FAQResponse(BaseModel):
    title: str
    items: List[FAQItemDTO]
    markdown: str  # Готовый Markdown

