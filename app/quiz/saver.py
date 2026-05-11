from __future__ import annotations

from string import ascii_lowercase
from typing import List

from .config import SaveFormat
from .helpers import shuffle_list
from .models import (
    Question,
    TrueFalseQuestion,
    MultipleChoiceQuestion,
    SelectAllThatApplyQuestion,
    FillInTheBlankQuestion,
    MatchingQuestion,
    MatchingPair,
    ShortOrLongAnswerQuestion,
)


def format_question_as_markdown(
        question: Question,
        save_format: SaveFormat,
        inline_separator: str = "::",
        multiline_separator: str = "?",
) -> str:
    """
    Универсальный форматтер одного вопроса в markdown.
    Аналог createCalloutQuestion / createSpacedRepetitionQuestion.
    """
    if save_format == SaveFormat.CALLOUT:
        return _format_callout_question(question)
    elif save_format == SaveFormat.SPACED_REPETITION:
        return _format_spaced_repetition_question(
            question,
            inline_separator=inline_separator,
            multiline_separator=multiline_separator,
        )
    else:
        raise ValueError(f"Unsupported save_format: {save_format}")


def format_quiz_as_markdown(
        questions: List[Question],
        save_format: SaveFormat,
        inline_separator: str = "::",
        multiline_separator: str = "?",
) -> str:
    """
    Форматирование списка вопросов в одну markdown-строку.
    Можно использовать для сохранения всего квиза в файл / БД.
    """
    parts = [
        format_question_as_markdown(
            q,
            save_format=save_format,
            inline_separator=inline_separator,
            multiline_separator=multiline_separator,
        )
        for q in questions
    ]
    return "".join(parts)


def _format_callout_question(question: Question) -> str:
    if isinstance(question, TrueFalseQuestion):
        answer_text = str(question.answer).capitalize()
        return (
            f"> [!question] {question.question}\n"
            f">> [!success]- Answer\n"
            f">> {answer_text}\n\n"
        )

    if isinstance(question, MultipleChoiceQuestion):
        options_lines = _get_callout_options(question.options)
        answer_line = options_lines[question.answer].replace(">", ">>", 1)
        return (
            f"> [!question] {question.question}\n"
            f"{'\n'.join(options_lines)}\n"
            f">> [!success]- Answer\n"
            f"{answer_line}\n\n"
        )

    if isinstance(question, SelectAllThatApplyQuestion):
        options_lines = _get_callout_options(question.options)
        answers_lines = [
            options_lines[idx].replace(">", ">>", 1) for idx in question.answer
        ]
        return (
            f"> [!question] {question.question}\n"
            f"{'\n'.join(options_lines)}\n"
            f">> [!success]- Answer\n"
            f"{'\n'.join(answers_lines)}\n\n"
        )

    if isinstance(question, FillInTheBlankQuestion):
        answer_text = ", ".join(question.answer)
        return (
            f"> [!question] {question.question}\n"
            f">> [!success]- Answer\n"
            f">> {answer_text}\n\n"
        )

    if isinstance(question, MatchingQuestion):
        left_options = shuffle_list([pair.left_option for pair in question.answer])
        right_options = shuffle_list([pair.right_option for pair in question.answer])

        answers_lines = _get_callout_matching_answers(
            left_options=left_options,
            right_options=right_options,
            pairs=question.answer,
        )

        group_a_lines = [
            line.replace(">", ">>", 1) for line in _get_callout_options(left_options)
        ]
        group_b_lines = [
            line.replace(">", ">>", 1)
            for line in _get_callout_options(right_options, start_index=13)
        ]

        return (
            f"> [!question] {question.question}\n"
            f">> [!example] Group A\n"
            f"{'\n'.join(group_a_lines)}\n"
            f">\n"
            f">> [!example] Group B\n"
            f"{'\n'.join(group_b_lines)}\n"
            f">\n"
            f">> [!success]- Answer\n"
            f"{'\n'.join(answers_lines)}\n\n"
        )

    if isinstance(question, ShortOrLongAnswerQuestion):
        return (
            f"> [!question] {question.question}\n"
            f">> [!success]- Answer\n"
            f">> {question.answer}\n\n"
        )

    # На будущее, если появятся новые типы
    return "> [!failure] Error saving question\n\n"


def _get_callout_options(options: List[str], start_index: int = 0) -> List[str]:
    letters = ascii_lowercase[start_index: start_index + len(options)]
    return [f"> {letter}) {opt}" for letter, opt in zip(letters, options)]


def _get_callout_matching_answers(
        left_options: List[str],
        right_options: List[str],
        pairs: List[MatchingPair],
) -> List[str]:
    left_index_map = {opt: idx for idx, opt in enumerate(left_options)}
    sorted_pairs = sorted(
        pairs,
        key=lambda p: left_index_map.get(p.left_option, 0),
    )

    result_lines: List[str] = []
    for pair in sorted_pairs:
        left_idx = left_options.index(pair.left_option)
        right_idx = right_options.index(pair.right_option)
        left_letter = ascii_lowercase[left_idx]  # a-m
        right_letter = ascii_lowercase[13 + right_idx]  # n-z
        result_lines.append(f">> {left_letter}) -> {right_letter})")
    return result_lines


def _format_spaced_repetition_question(
        question: Question,
        inline_separator: str,
        multiline_separator: str,
) -> str:
    if isinstance(question, TrueFalseQuestion):
        answer_text = str(question.answer).capitalize()
        return (
            f"**True or False:** {question.question} "
            f"{inline_separator} {answer_text}\n\n"
        )

    if isinstance(question, MultipleChoiceQuestion):
        options_lines = _get_spaced_rep_options(question.options)
        answer_line = options_lines[question.answer]
        return (
            f"**Multiple Choice:** {question.question}\n"
            f"{'\n'.join(options_lines)}\n"
            f"{multiline_separator}\n"
            f"{answer_line}\n\n"
        )

    if isinstance(question, SelectAllThatApplyQuestion):
        options_lines = _get_spaced_rep_options(question.options)
        answer_lines = [options_lines[idx] for idx in question.answer]
        return (
            f"**Select All That Apply:** {question.question}\n"
            f"{'\n'.join(options_lines)}\n"
            f"{multiline_separator}\n"
            f"{'\n'.join(answer_lines)}\n\n"
        )

    if isinstance(question, FillInTheBlankQuestion):
        answer_text = ", ".join(question.answer)
        return (
            f"**Fill in the Blank:** {question.question} "
            f"{inline_separator} {answer_text}\n\n"
        )

    if isinstance(question, MatchingQuestion):
        left_options = shuffle_list([pair.left_option for pair in question.answer])
        right_options = shuffle_list([pair.right_option for pair in question.answer])
        answers_lines = _get_spaced_rep_matching_answers(
            left_options=left_options,
            right_options=right_options,
            pairs=question.answer,
        )

        group_a_lines = _get_spaced_rep_options(left_options)
        group_b_lines = _get_spaced_rep_options(right_options, start_index=13)

        return (
            f"**Matching:** {question.question}\n"
            f"Group A\n"
            f"{'\n'.join(group_a_lines)}\n"
            f"Group B\n"
            f"{'\n'.join(group_b_lines)}\n"
            f"{multiline_separator}\n"
            f"{'\n'.join(answers_lines)}\n\n"
        )

    if isinstance(question, ShortOrLongAnswerQuestion):
        # Как в плагине: решаем по длине текста
        label = "Short Answer" if len(question.answer) < 250 else "Long Answer"
        return (
            f"**{label}:** {question.question} "
            f"{inline_separator} {question.answer}\n\n"
        )

    return "Error saving question\n\n"


def _get_spaced_rep_options(options: List[str], start_index: int = 0) -> List[str]:
    letters = ascii_lowercase[start_index: start_index + len(options)]
    return [f"{letter}) {opt}" for letter, opt in zip(letters, options)]


def _get_spaced_rep_matching_answers(
        left_options: List[str],
        right_options: List[str],
        pairs: List[MatchingPair],
) -> List[str]:
    left_index_map = {opt: idx for idx, opt in enumerate(left_options)}
    sorted_pairs = sorted(
        pairs,
        key=lambda p: left_index_map.get(p.left_option, 0),
    )

    result_lines: List[str] = []
    for pair in sorted_pairs:
        left_idx = left_options.index(pair.left_option)
        right_idx = right_options.index(pair.right_option)
        left_letter = ascii_lowercase[left_idx]
        right_letter = ascii_lowercase[13 + right_idx]
        result_lines.append(f"{left_letter}) -> {right_letter})")
    return result_lines
