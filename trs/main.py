import os
from loguru import logger
from .schema import Document, Indicators
from .llm import LLM

class TRS:
    def __init__(self):
        self.llm = LLM()  # Only Ollama
        self.previous_files = set()

    def txt_to_doc(self, file_path: str) -> Document:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        return Document(text=text, source=file_path)

    # Detections
    def detections(self, file: str) -> str:
        doc = self.txt_to_doc(file)
        return self.llm.detect(doc=doc)

    # Summary + Mindmap + IOC
    def summarize(self, file: str):
        doc = self.txt_to_doc(file)
        summary = self.llm.summarize(doc=doc)
        mindmap = self.llm.mindmap(doc=doc)
        # IOC extraction placeholder
        iocs = Indicators()
        return summary.summary if summary else "", mindmap, iocs

    # QnA
    def qna(self, prompt: str) -> str:
        # For simplicity, send prompt as document text
        doc = Document(text=prompt, source="qna")
        return self.llm.qna(prompt, doc.text)

    # Custom prompt
    def custom(self, file: str, prompt_name: str) -> str:
        doc = self.txt_to_doc(file)
        return self.llm.custom(prompt_name, doc)
