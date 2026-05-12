from PyPDF2 import PdfReader


def extract_text_from_pdf(path: str) -> str:
    try:
        reader = PdfReader(path)
        chunks = []
        for page in reader.pages:
            text = page.extract_text() or ""
            chunks.append(text)
        full_text = "\n".join(chunks).strip()
        if not full_text:
            full_text = ""
        return full_text
    except FileNotFoundError:
        raise
    except Exception as e:
        raise RuntimeError(f"Ошибка при чтении PDF '{path}': {e}")
