from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class SaveFormat(str, Enum):
    CALLOUT = "callout"
    SPACED_REPETITION = "spaced_repetition"


@dataclass
class GeneralConfig:
    show_note_path: bool = False
    show_folder_path: bool = False
    include_subfolder_notes: bool = True
    randomize_questions: bool = True
    language: str = "ru"  # только русский по умолчанию


@dataclass
class GenerationConfig:
    generate_true_false: bool = True
    number_true_false: int = 1

    generate_multiple_choice: bool = True
    number_multiple_choice: int = 1

    generate_select_all_that_apply: bool = True
    number_select_all_that_apply: int = 1

    generate_fill_in_the_blank: bool = True
    number_fill_in_the_blank: int = 1

    generate_matching: bool = True
    number_matching: int = 1

    generate_short_answer: bool = True
    number_short_answer: int = 1

    generate_long_answer: bool = True
    number_long_answer: int = 1


@dataclass
class SavingConfig:
    auto_save: bool = False
    save_path: str = "./quizzes"
    save_format: SaveFormat = SaveFormat.CALLOUT
    quiz_material_property: str = "sources"
    inline_separator: str = "::"
    multiline_separator: str = "?"


@dataclass
class QuizSettings:
    general: GeneralConfig = field(default_factory=GeneralConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    saving: SavingConfig = field(default_factory=SavingConfig)
