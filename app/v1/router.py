import logging
from dataclasses import asdict
from typing import List
from uuid import uuid4

from fastapi import APIRouter, HTTPException, Body

from app.faq.config import FAQGenerationConfig
from app.faq.generator import format_faq_as_markdown, generate_faq_from_file
from app.quiz.checker import check_quiz_answers
from app.quiz.contracts.adapters import questions_to_bundles
from app.quiz.contracts.converters import contract_to_internal, internal_to_contract
from app.quiz.contracts.models import ExplainQuizRequest, ExplainQuizResponse
from app.quiz.explainer import generate_all_explanations_async, format_markdown_with_explanations
from app.quiz.generation.config import QuizGenerationConfig
from app.quiz.generation.service import generate_quiz_from_text
from app.quiz.models import GeneratedQuiz as InternalQuiz
from app.quiz.models import UserAnswer, CheckResponse
from app.quiz.rag import SimpleVectorStore
from app.services.proxy_client import proxy_completion
from app.services.similarity import (
    cosine_similarity_pdfs_matrix,
    cosine_similarity_topics_from_json,
    cosine_similarity_two_pdfs,
)
from app.v1.schemas import (
    CompletionRequest,
    CompletionResponse,
    FAQGenerationRequest,
    FAQItemDTO,
    FAQResponse,
    GeneratedQuestionDTO,
    PairSimilarityRequest,
    QuizGenerationRequest,
    QuizGenerationResponse,
    SimilarityRequest,
    TopicPairRequest,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["v1"])


@router.post("/complete", response_model=CompletionResponse)
async def complete(req: CompletionRequest):
    try:
        content, model = await proxy_completion(
            text=req.text,
            user_prompt=req.user_prompt,
            system_prompt=req.system_prompt,
        )
        return CompletionResponse(model=model, content=content)
    except ValueError as ve:
        logger.error(f"Ошибка валидации в completion: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Неожиданная ошибка в completion: {e}")
        raise HTTPException(status_code=502, detail="Внутренняя ошибка сервиса")


@router.post("/similarity/pdfs/matrix")
async def pdfs_matrix(req: SimilarityRequest):
    try:
        mat = cosine_similarity_pdfs_matrix(req.paths)
        return {"matrix": mat}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Ошибка в сходстве PDF (matrix): {e}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка")


@router.post("/similarity/pdfs/pair")
async def pdfs_pair(req: PairSimilarityRequest):
    try:
        score = cosine_similarity_two_pdfs(req.a, req.b)
        return {"score": score}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Ошибка в сходстве PDF (pair): {e}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка")


@router.post("/similarity/topics/pair")
async def topics_pair(req: TopicPairRequest):
    try:
        score = cosine_similarity_topics_from_json(req.topic_a, req.topic_b, req.json_path)
        return {"score": score}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Ошибка в сходстве тем: {e}")
        raise HTTPException(status_code=500, detail="Внутренняя ошибка")


@router.post("/quiz/generate", response_model=QuizGenerationResponse)
async def generate_quiz(req: QuizGenerationRequest) -> QuizGenerationResponse:
    """
    Генерирует квиз на основе темы.
    """
    try:
        question_count = max(1, req.number_of_questions)
        cfg = QuizGenerationConfig(
            language=req.language,
            generate_true_false=False,
            num_true_false=0,
            generate_multiple_choice=True,
            num_multiple_choice=question_count,
            generate_select_all_that_apply=False,
            num_select_all_that_apply=0,
            generate_fill_in_the_blank=False,
            num_fill_in_the_blank=0,
            generate_matching=False,
            num_matching=0,
            generate_short_answer=False,
            num_short_answer=0,
            generate_long_answer=False,
            num_long_answer=0,
        )
        internal_questions = await generate_quiz_from_text(note_text=req.topic, cfg=cfg)

        bundles = questions_to_bundles(internal_questions)

        dto_questions: List[GeneratedQuestionDTO] = [
            GeneratedQuestionDTO(
                question=b.question,
                variants=b.variants,
                matchingConfig=b.matchingConfig,
            )
            for b in bundles
        ]

        return QuizGenerationResponse(
            quizId=uuid4(),
            questions=dto_questions,
        )
    except Exception as e:
        logger.error(f"Ошибка генерации квиза: {e}")
        raise HTTPException(status_code=500, detail="Ошибка генерации квиза")


@router.post("/faq/generate", response_model=FAQResponse)
async def generate_faq(req: FAQGenerationRequest):
    try:
        cfg = FAQGenerationConfig(
            language=req.language,
            num_questions=req.num_questions,
            detail_level=req.detail_level,
        )
        faq = await generate_faq_from_file(
            file_path=req.file_path,
            title=req.title,
            cfg=cfg,
        )
        markdown = format_faq_as_markdown(faq)
        items_dto = [FAQItemDTO(**asdict(item)) for item in faq.items]
        return FAQResponse(title=faq.title, items=items_dto, markdown=markdown)
    except Exception as e:
        logger.error(f"Ошибка генерации FAQ: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/quiz/check", response_model=CheckResponse)
async def check_quiz(
        quiz: InternalQuiz = Body(...),
        answers: List[UserAnswer] = Body(...),
        source_text: str = Body(...),
        document_id: str = "temp"
):
    if not quiz.questions:
        raise HTTPException(400, "Квиз пустой")
    try:
        return await check_quiz_answers(quiz, answers, source_text, document_id)
    except Exception as e:
        logger.error(f"Ошибка проверки квиза: {e}")
        raise HTTPException(status_code=500, detail="Ошибка проверки квиза")


@router.post("/quiz/explain", response_model=ExplainQuizResponse)
async def explain_quiz(request: ExplainQuizRequest):
    rag = SimpleVectorStore()
    rag.add_document_sync(request.text)

    internal = contract_to_internal(request.questions)

    await generate_all_explanations_async(internal, rag, request.difficulty)

    enriched = internal_to_contract(internal, request.questions)

    markdown = format_markdown_with_explanations(internal)

    return ExplainQuizResponse(
        quizId=request.quizId or uuid4(),
        questions=enriched,
        markdown=markdown
    )
