from __future__ import annotations

import re
from string import ascii_lowercase
from typing import List

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


# --- Общие регулярки для callout-парсера --- #

CALL_OUT_QUESTION_RE = re.compile(
    r'^>\s*\[!question][+-]?\s*(.+?)\s*$',
    flags=re.IGNORECASE,
)

SUCCESS_RE = re.compile(r"\[!success]", flags=re.IGNORECASE)
CALL_OUT_OPTION_SINGLE_RE = re.compile(
    r'^\s*>\s*([a-z])\)\s*(.+?)\s*$',
    flags=re.IGNORECASE,
)
CALL_OUT_OPTION_ANY_LEVEL_RE = re.compile(
    r'^\s*>+\s*([a-z])\)\s*(.+?)\s*$',
    flags=re.IGNORECASE,
)
CALL_OUT_ANSWER_TEXT_RE = re.compile(
    r'^\s*>>\s*(.+?)\s*$',
    flags=re.IGNORECASE,
)
CALL_OUT_ANSWER_LETTER_RE = re.compile(
    r'^\s*>>\s*([a-z])\)\s*.*$',
    flags=re.IGNORECASE,
)
GROUP_A_RE = re.compile(r"\[!example].*group\s*a", flags=re.IGNORECASE)
GROUP_B_RE = re.compile(r"\[!example].*group\s*b", flags=re.IGNORECASE)
PAIR_RE = re.compile(
    r'^\s*>>\s*([a-m])\)\s*-+>\s*([n-z])\)\s*$',
    flags=re.IGNORECASE,
)


class QuizParser:
    """
    Универсальный парсер, приближённый по поведению к Obsidian-плагину.

    Можно:
      - отключать парсинг callout / spaced repetition по флагам,
      - менять разделители для spaced repetition (inline / multiline).
    """

    def __init__(
            self,
            inline_separator: str = "::",
            multiline_separator: str = "?",
            parse_callouts: bool = True,
            parse_spaced_repetition: bool = True,
    ) -> None:
        self.inline_separator = inline_separator
        self.multiline_separator = multiline_separator
        self.parse_callouts = parse_callouts
        self.parse_spaced_repetition = parse_spaced_repetition

    def parse(self, markdown: str) -> List[Question]:
        questions: List[Question] = []

        if self.parse_callouts:
            questions.extend(self._parse_callout_questions(markdown))

        if self.parse_spaced_repetition:
            questions.extend(
                self._parse_spaced_repetition_questions(
                    markdown,
                    inline_sep=self.inline_separator,
                    multiline_sep=self.multiline_separator,
                )
            )

        return questions

    # --- CALL OUT PARSER --- #

    def _parse_callout_questions(self, content: str) -> List[Question]:
        questions: List[Question] = []
        lines = content.splitlines()
        i = 0

        while i < len(lines):
            line = lines[i].strip()
            m_q = CALL_OUT_QUESTION_RE.match(line)
            if not m_q:
                i += 1
                continue

            question_text = m_q.group(1).strip()
            i += 1

            # Собираем блок строк, относящийся к этому вопросу
            block_lines: List[str] = []
            while i < len(lines):
                if CALL_OUT_QUESTION_RE.match(lines[i].strip()):
                    break
                block_lines.append(lines[i])
                i += 1

            # 1) Matching (группы A/B + стрелочки)
            matching_q = self._parse_callout_matching_block(question_text, block_lines)
            if matching_q is not None:
                questions.append(matching_q)
                continue

            # 2) Multiple choice / Select all
            mc_q = self._parse_callout_multiple_choice_block(question_text, block_lines)
            if mc_q is not None:
                questions.append(mc_q)
                continue

            # 3) True/False / Fill in the blank / Short/Long answer
            simple_q = self._parse_callout_simple_block(question_text, block_lines)
            if simple_q is not None:
                questions.append(simple_q)

        return questions

    def _parse_callout_multiple_choice_block(
            self,
            question_text: str,
            block_lines: List[str],
    ) -> Question | None:
        # Находим строку [!success]
        idx_success = None
        for idx, line in enumerate(block_lines):
            if SUCCESS_RE.search(line):
                idx_success = idx
                break

        if idx_success is None:
            return None

        # Варианты ответов до [!success]
        options_by_letter: dict[str, str] = {}
        for line in block_lines[:idx_success]:
            m_opt = CALL_OUT_OPTION_SINGLE_RE.match(line.strip())
            if m_opt:
                letter = m_opt.group(1).lower()
                text = m_opt.group(2).strip()
                options_by_letter[letter] = text

        if not options_by_letter:
            return None

        # Ответы после [!success]
        answer_letters: List[str] = []
        for line in block_lines[idx_success + 1 :]:
            stripped = line.strip()
            m_ans = CALL_OUT_ANSWER_LETTER_RE.match(stripped)
            if not m_ans:
                # если строка с ">>", но без буквы — пропускаем,
                # иначе заканчиваем блок ответов
                if stripped.startswith(">>"):
                    continue
                break
            answer_letters.append(m_ans.group(1).lower())

        if not answer_letters:
            return None

        # Сортируем варианты по алфавиту (a..z)
        options: List[str] = [
            options_by_letter[c] for c in ascii_lowercase if c in options_by_letter
        ]
        letter_to_index = {
            c: i for i, c in enumerate(ascii_lowercase) if c in options_by_letter
        }

        answer_indices = [
            letter_to_index[a] for a in answer_letters if a in letter_to_index
        ]
        if not answer_indices:
            return None

        if len(answer_indices) == 1:
            return MultipleChoiceQuestion(
                question=question_text,
                options=options,
                answer=answer_indices[0],
            )

        return SelectAllThatApplyQuestion(
            question=question_text,
            options=options,
            answer=answer_indices,
        )

    def _parse_callout_matching_block(
            self,
            question_text: str,
            block_lines: List[str],
    ) -> MatchingQuestion | None:
        idx_ga = idx_gb = idx_success = None

        for idx, line in enumerate(block_lines):
            if idx_ga is None and GROUP_A_RE.search(line):
                idx_ga = idx
            elif idx_ga is not None and idx_gb is None and GROUP_B_RE.search(line):
                idx_gb = idx

            if idx_success is None and SUCCESS_RE.search(line):
                idx_success = idx

        if idx_ga is None or idx_gb is None or idx_success is None:
            return None

        # Левая группа (A–M)
        left_by_letter: dict[str, str] = {}
        i = idx_ga + 1
        while i < len(block_lines) and i < idx_gb:
            m = CALL_OUT_OPTION_ANY_LEVEL_RE.match(block_lines[i].strip())
            if m:
                letter = m.group(1).lower()
                text = m.group(2).strip()
                left_by_letter[letter] = text
                i += 1
            else:
                # пропускаем пустые и просто ">"
                if block_lines[i].strip().startswith(">"):
                    i += 1
                    continue
                break

        # Правая группа (N–Z)
        right_by_letter: dict[str, str] = {}
        i = idx_gb + 1
        while i < len(block_lines) and i < idx_success:
            m = CALL_OUT_OPTION_ANY_LEVEL_RE.match(block_lines[i].strip())
            if m:
                letter = m.group(1).lower()
                text = m.group(2).strip()
                right_by_letter[letter] = text
                i += 1
            else:
                if block_lines[i].strip().startswith(">"):
                    i += 1
                    continue
                break

        # Пары a) -> n)
        pairs_letters: List[tuple[str, str]] = []
        for line in block_lines[idx_success + 1 :]:
            stripped = line.strip()
            m_pair = PAIR_RE.match(stripped)
            if not m_pair:
                if stripped.startswith(">>"):
                    continue
                break
            left_letter = m_pair.group(1).lower()
            right_letter = m_pair.group(2).lower()
            pairs_letters.append((left_letter, right_letter))

        if not left_by_letter or not right_by_letter or not pairs_letters:
            return None

        # Сопоставляем буквы с текстами
        answer_pairs: List[MatchingPair] = []
        for left_letter, right_letter in pairs_letters:
            left_text = left_by_letter.get(left_letter)
            right_text = right_by_letter.get(right_letter)
            if left_text is None or right_text is None:
                continue
            answer_pairs.append(MatchingPair(left_option=left_text, right_option=right_text))

        if not answer_pairs:
            return None

        return MatchingQuestion(question=question_text, answer=answer_pairs)

    def _parse_callout_simple_block(
            self,
            question_text: str,
            block_lines: List[str],
    ) -> Question | None:
        # Ищем [!success]
        idx_success = None
        for idx, line in enumerate(block_lines):
            if SUCCESS_RE.search(line):
                idx_success = idx
                break

        if idx_success is None:
            return None

        # Первый ответ после [!success]
        answer_line = None
        for line in block_lines[idx_success + 1 :]:
            stripped = line.strip()
            m = CALL_OUT_ANSWER_TEXT_RE.match(stripped)
            if m:
                answer_line = m.group(1).strip()
                break
            if stripped == "":
                continue
            if stripped.startswith(">>"):
                continue
            break

        if answer_line is None:
            return None

        lower_ans = answer_line.lower()

        # True / False
        if lower_ans in {"true", "false"}:
            return TrueFalseQuestion(
                question=question_text,
                answer=(lower_ans == "true"),
            )

        # Fill in the blank (в вопросе есть `____`)
        if re.search(r"`_+`", question_text):
            parts = [p.strip() for p in re.split(r"\s*,\s+", answer_line) if p.strip()]
            return FillInTheBlankQuestion(
                question=question_text,
                answer=parts,
            )

        # Открытый ответ (короткий/длинный — различаем по длине при сохранении)
        return ShortOrLongAnswerQuestion(
            question=question_text,
            answer=answer_line,
        )

    # --- SPACED REPETITION PARSER --- #

    def _parse_spaced_repetition_questions(
            self,
            content: str,
            inline_sep: str,
            multiline_sep: str,
    ) -> List[Question]:
        lines = content.splitlines()
        i = 0
        questions: List[Question] = []

        # Регулярки под заголовки
        tf_re = re.compile(
            r"^\s*\*\*True or False:\*\*\s*(?P<question>.+?)\s*"
            + re.escape(inline_sep)
            + r"\s*(?P<answer>.+?)\s*$",
            flags=re.IGNORECASE,
            )
        fib_re = re.compile(
            r"^\s*\*\*Fill in the Blank:\*\*\s*(?P<question>.+?)\s*"
            + re.escape(inline_sep)
            + r"\s*(?P<answer>.+?)\s*$",
            flags=re.IGNORECASE,
            )
        short_re = re.compile(
            r"^\s*\*\*Short Answer:\*\*\s*(?P<question>.+?)\s*"
            + re.escape(inline_sep)
            + r"\s*(?P<answer>.+?)\s*$",
            flags=re.IGNORECASE,
            )
        long_re = re.compile(
            r"^\s*\*\*Long Answer:\*\*\s*(?P<question>.+?)\s*"
            + re.escape(inline_sep)
            + r"\s*(?P<answer>.+?)\s*$",
            flags=re.IGNORECASE,
            )
        mc_header_re = re.compile(
            r"^\s*\*\*Multiple Choice:\*\*\s*(?P<question>.+?)\s*$",
            flags=re.IGNORECASE,
        )
        sa_header_re = re.compile(
            r"^\s*\*\*Select All That Apply:\*\*\s*(?P<question>.+?)\s*$",
            flags=re.IGNORECASE,
        )
        matching_header_re = re.compile(
            r"^\s*\*\*Matching:\*\*\s*(?P<question>.+?)\s*$",
            flags=re.IGNORECASE,
        )
        opt_re = re.compile(r"^\s*([a-z])\)\s+(.+?)\s*$", flags=re.IGNORECASE)
        ans_from_opt_re = re.compile(r"^\s*([a-z])\)", flags=re.IGNORECASE)
        pair_line_re = re.compile(
            r"^\s*([a-m])\)\s*->\s*([n-z])\)\s*$",
            flags=re.IGNORECASE,
        )

        while i < len(lines):
            line = lines[i].strip()

            # True / False
            m = tf_re.match(line)
            if m:
                q = m.group("question").strip()
                ans = m.group("answer").strip().lower()
                questions.append(
                    TrueFalseQuestion(question=q, answer=(ans == "true"))
                )
                i += 1
                continue

            # Fill in the Blank
            m = fib_re.match(line)
            if m:
                q = m.group("question").strip()
                raw = m.group("answer").strip()
                parts = [p.strip() for p in re.split(r"\s*,\s+", raw) if p.strip()]
                questions.append(FillInTheBlankQuestion(question=q, answer=parts))
                i += 1
                continue

            # Short Answer
            m = short_re.match(line)
            if m:
                q = m.group("question").strip()
                ans = m.group("answer").strip()
                questions.append(ShortOrLongAnswerQuestion(question=q, answer=ans))
                i += 1
                continue

            # Long Answer
            m = long_re.match(line)
            if m:
                q = m.group("question").strip()
                ans = m.group("answer").strip()
                questions.append(ShortOrLongAnswerQuestion(question=q, answer=ans))
                i += 1
                continue

            # Multiple Choice
            m = mc_header_re.match(line)
            if m:
                q = m.group("question").strip()
                i += 1

                options_by_letter: dict[str, str] = {}

                # читаем варианты до multiline_sep
                while i < len(lines):
                    l = lines[i].strip()
                    if l == "":
                        i += 1
                        continue
                    if l == multiline_sep:
                        i += 1
                        break
                    m_opt = opt_re.match(l)
                    if not m_opt:
                        break
                    letter = m_opt.group(1).lower()
                    text = m_opt.group(2).strip()
                    options_by_letter[letter] = text
                    i += 1

                options: List[str] = [
                    options_by_letter[c] for c in ascii_lowercase if c in options_by_letter
                ]
                answer_indices: List[int] = []

                # читаем ответ(ы)
                while i < len(lines):
                    l = lines[i].strip()
                    if l == "":
                        i += 1
                        break
                    m_ans = ans_from_opt_re.match(l)
                    if not m_ans:
                        break
                    letter = m_ans.group(1).lower()
                    idx = ord(letter) - ord("a")
                    answer_indices.append(idx)
                    i += 1

                if options and answer_indices:
                    if len(answer_indices) == 1:
                        questions.append(
                            MultipleChoiceQuestion(
                                question=q,
                                options=options,
                                answer=answer_indices[0],
                            )
                        )
                    else:
                        questions.append(
                            SelectAllThatApplyQuestion(
                                question=q,
                                options=options,
                                answer=answer_indices,
                            )
                        )
                continue

            # Select All That Apply
            m = sa_header_re.match(line)
            if m:
                q = m.group("question").strip()
                i += 1

                options_by_letter: dict[str, str] = {}
                while i < len(lines):
                    l = lines[i].strip()
                    if l == "":
                        i += 1
                        continue
                    if l == multiline_sep:
                        i += 1
                        break
                    m_opt = opt_re.match(l)
                    if not m_opt:
                        break
                    letter = m_opt.group(1).lower()
                    text = m_opt.group(2).strip()
                    options_by_letter[letter] = text
                    i += 1

                options = [
                    options_by_letter[c] for c in ascii_lowercase if c in options_by_letter
                ]
                answer_indices: List[int] = []
                while i < len(lines):
                    l = lines[i].strip()
                    if l == "":
                        i += 1
                        break
                    m_ans = ans_from_opt_re.match(l)
                    if not m_ans:
                        break
                    letter = m_ans.group(1).lower()
                    idx = ord(letter) - ord("a")
                    answer_indices.append(idx)
                    i += 1

                if options and answer_indices:
                    questions.append(
                        SelectAllThatApplyQuestion(
                            question=q,
                            options=options,
                            answer=answer_indices,
                        )
                    )
                continue

            # Matching
            m = matching_header_re.match(line)
            if m:
                q = m.group("question").strip()
                i += 1

                # Пропускаем пустые строки
                while i < len(lines) and lines[i].strip() == "":
                    i += 1

                # Group A
                if i < len(lines) and lines[i].strip().lower().startswith("group a"):
                    i += 1

                left_by_letter: dict[str, str] = {}
                while i < len(lines):
                    l = lines[i].strip()
                    if l.lower().startswith("group b"):
                        i += 1
                        break
                    m_opt = opt_re.match(l)
                    if m_opt:
                        letter = m_opt.group(1).lower()
                        text = m_opt.group(2).strip()
                        left_by_letter[letter] = text
                        i += 1
                    else:
                        i += 1

                right_by_letter: dict[str, str] = {}
                while i < len(lines):
                    l = lines[i].strip()
                    if l == multiline_sep:
                        i += 1
                        break
                    m_opt = opt_re.match(l)
                    if m_opt:
                        letter = m_opt.group(1).lower()
                        text = m_opt.group(2).strip()
                        right_by_letter[letter] = text
                        i += 1
                    else:
                        i += 1

                pairs: List[MatchingPair] = []
                while i < len(lines):
                    l = lines[i].strip()
                    if l == "":
                        i += 1
                        break
                    m_pair = pair_line_re.match(l)
                    if not m_pair:
                        break
                    left_letter = m_pair.group(1).lower()
                    right_letter = m_pair.group(2).lower()
                    left_text = left_by_letter.get(left_letter)
                    right_text = right_by_letter.get(right_letter)
                    if left_text and right_text:
                        pairs.append(
                            MatchingPair(
                                left_option=left_text,
                                right_option=right_text,
                            )
                        )
                    i += 1

                if left_by_letter and right_by_letter and pairs:
                    questions.append(MatchingQuestion(question=q, answer=pairs))
                continue

            i += 1

        return questions
