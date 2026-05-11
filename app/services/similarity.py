from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from app.utils.json_store import read_texts_by_topics
from app.utils.pdf_utils import extract_text_from_pdf


def _tfidf_cosine(texts: List[str]):
    vec = TfidfVectorizer(stop_words=None)
    X = vec.fit_transform(texts)
    sim = cosine_similarity(X)
    return sim


def cosine_similarity_pdfs_matrix(paths: List[str]):
    if not paths or len(paths) < 2:
        raise ValueError("Передайте >=2 путей к PDF")
    texts = [extract_text_from_pdf(p) for p in paths]
    sim = _tfidf_cosine(texts)
    return sim.tolist()


def cosine_similarity_two_pdfs(a: str, b: str) -> float:
    texts = [extract_text_from_pdf(a), extract_text_from_pdf(b)]
    sim = _tfidf_cosine(texts)
    return float(sim[0, 1])


def cosine_similarity_topics_from_json(topic_a: str, topic_b: str, json_path: str) -> float:
    texts_map = read_texts_by_topics(json_path)
    if topic_a not in texts_map or topic_b not in texts_map:
        raise ValueError("Обе темы должны присутствовать в JSON")
    texts = [texts_map[topic_a], texts_map[topic_b]]
    sim = _tfidf_cosine(texts)
    return float(sim[0, 1])
