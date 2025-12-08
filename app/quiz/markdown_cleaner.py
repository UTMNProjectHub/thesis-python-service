from __future__ import annotations

import re


def clean_up_note_contents(note_contents: str, has_front_matter: bool) -> str:
    """
    Убираем фронтматтер, ссылки, markdown-разметку и лишние пробелы.
    Это пригодится для подготовки текста к генерации вопросов (LLM).
    """
    text = note_contents
    if has_front_matter:
        text = _remove_front_matter(text)
    text = _clean_up_links(text)
    text = _remove_markdown_headings(text)
    text = _remove_markdown_formatting(text)
    text = _clean_up_whitespace(text)
    return text


def _remove_front_matter(input_text: str) -> str:
    # --- ... --- в начале файла
    yaml_front_matter_regex = re.compile(r"^---[\s\S]+?---\n", flags=re.MULTILINE)
    return yaml_front_matter_regex.sub("", input_text)


def _clean_up_links(input_text: str) -> str:
    # [[link|display]] или [[link]] или [text](url)
    wiki_link_pattern = r"\[\[([^\]|]+)(?:\|([^\]]+))??]]"
    markdown_link_pattern = r"\[([^\]]+)]\([^)]+\)"

    combined_regex = re.compile(f"{wiki_link_pattern}|{markdown_link_pattern}")

    def _replace(match: re.Match) -> str:
        wiki_link = match.group(1)
        wiki_display_text = match.group(2)
        markdown_text = match.group(3)
        return wiki_display_text or wiki_link or markdown_text or ""

    return combined_regex.sub(_replace, input_text)


def _remove_markdown_headings(input_text: str) -> str:
    heading_regex = re.compile(r"^(#+.*)$", flags=re.MULTILINE)
    return heading_regex.sub("", input_text)


def _remove_markdown_formatting(input_text: str) -> str:
    # **bold**, *italic*, ~~strike~~, ==highlight==, %%comment%%
    markdown_formatting_regex = re.compile(r"([*_]{1,3}|~~|==|%%)(.*?)\1", flags=re.DOTALL)
    return markdown_formatting_regex.sub(r"\2", input_text)


def _clean_up_whitespace(input_text: str) -> str:
    consecutive_spaces_regex = re.compile(r"\s+")
    return consecutive_spaces_regex.sub(" ", input_text).strip()
