from __future__ import annotations

import math
from dataclasses import dataclass

from app.api.core.config import settings


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def _clamp_int(value: float, min_value: int, max_value: int) -> int:
    return int(round(_clamp(value, min_value, max_value)))


@dataclass(frozen=True)
class LectureContextPolicy:
    target_words: int
    target_sections: int
    section_output_tokens: int
    plan_context_chunks: int
    section_context_chunks: int
    plan_pool_k: int
    section_pool_k: int
    profile_chunks_per_doc: int
    plan_context_token_budget: int
    section_context_token_budget: int
    final_edit_input_token_budget: int


def build_context_policy(
        *,
        doc_count: int,
        total_chunks: int,
        document_profiles_enabled: bool,
        target_words: int | None = None,
) -> LectureContextPolicy:
    target = max(1, int(target_words or settings.lecture_target_words))
    words_per_section = max(1, settings.lecture_words_per_section)
    min_sections = max(1, settings.lecture_min_sections)
    max_sections = max(min_sections, settings.lecture_max_sections)
    docs = max(1, doc_count)
    chunks = max(1, total_chunks)

    target_sections = _clamp_int(round(target / words_per_section), min_sections, max_sections)
    section_words = target / max(1, target_sections)
    section_output_tokens = _clamp_int(round(section_words * 1.8), 900, 2600)

    doc_scale = _clamp(1 + math.log2(docs) * 0.35, 1.0, 2.5)
    corpus_scale = _clamp(1 + math.log2(max(chunks, 80) / 80) * 0.25, 1.0, 2.0)
    lecture_scale = _clamp(target / 5000, 0.75, 2.5)

    plan_context_chunks = _clamp_int(
        settings.lecture_base_plan_context_chunks * lecture_scale * max(doc_scale, corpus_scale),
        8,
        settings.lecture_max_plan_context_chunks,
    )
    section_context_chunks = _clamp_int(
        settings.lecture_base_section_context_chunks * math.sqrt(lecture_scale) * max(1.0, doc_scale * 0.75),
        6,
        settings.lecture_max_section_context_chunks,
    )

    if not document_profiles_enabled:
        plan_context_chunks = min(
            settings.lecture_max_plan_context_chunks,
            max(plan_context_chunks, docs),
        )

    plan_pool_k = _clamp_int(
        max(settings.lecture_retrieval_pool_k, plan_context_chunks * 4, docs * 8),
        settings.lecture_retrieval_pool_k,
        settings.lecture_max_retrieval_pool_k,
    )
    section_pool_k = _clamp_int(
        max(settings.lecture_retrieval_pool_k, section_context_chunks * 4, docs * 4),
        settings.lecture_retrieval_pool_k,
        settings.lecture_max_retrieval_pool_k,
    )

    avg_doc_chunks = chunks / docs
    profile_chunks_per_doc = _clamp_int(
        settings.lecture_doc_profile_chunks + math.ceil(math.log2(avg_doc_chunks + 1)),
        6,
        settings.lecture_doc_profile_max_chunks,
    )

    return LectureContextPolicy(
        target_words=target,
        target_sections=target_sections,
        section_output_tokens=section_output_tokens,
        plan_context_chunks=plan_context_chunks,
        section_context_chunks=section_context_chunks,
        plan_pool_k=plan_pool_k,
        section_pool_k=section_pool_k,
        profile_chunks_per_doc=profile_chunks_per_doc,
        plan_context_token_budget=max(1, settings.lecture_plan_context_token_budget),
        section_context_token_budget=max(1, settings.lecture_section_context_token_budget),
        final_edit_input_token_budget=max(1, settings.lecture_final_edit_input_token_budget),
    )
