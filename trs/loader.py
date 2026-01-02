import os
import json
from typing import List, Optional
from loguru import logger
from unstructured.partition.html import partition_html
import PyPDF2

from .schema import Document


class Loader:

    # ---------------------------
    # Load URL (HTML)
    # ---------------------------
    def url(self, source: str) -> Optional[Document]:
        """Retrieve a URL and return a Document containing the text"""
        logger.info(f"Loading URL: {source}")

        try:
            elements = partition_html(url=source)
        except Exception as err:
            logger.error(f"Error retrieving HTML: {source} - {err}")
            return None

        content = "\n".join([elem.text for elem in elements])

        return Document(
            source=source,
            text=content,
            metadata={"type": "url"}
        )

    # ---------------------------
    # Load PDF
    # ---------------------------
    def pdf(self, source: str) -> Optional[Document]:
        """Parse a PDF file to text and return a Document"""

        if not os.path.exists(source):
            logger.error(f"File {source} does not exist")
            return None

        logger.info(f"Loading PDF: {source}")

        texts = []
        with open(source, "rb") as fp:
            pdf_reader = PyPDF2.PdfReader(fp)

            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    texts.append(page.extract_text() or "")
                except Exception as err:
                    logger.error(f"Error reading page {page_num}: {err}")

        return Document(
            source=source,
            text="\n".join(texts),
            metadata={"type": "pdf"}
        )

    # ---------------------------
    # Load TXT
    # ---------------------------
    def txt(self, source: str) -> Optional[Document]:
        if not os.path.exists(source):
            logger.error(f"File {source} does not exist")
            return None

        logger.info(f"Loading TXT: {source}")

        with open(source, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()

        return Document(
            source=source,
            text=content,
            metadata={"type": "txt"}
        )

    # ---------------------------
    # Load JSON
    # ---------------------------
    def json(self, source: str) -> Optional[Document]:
        if not os.path.exists(source):
            logger.error(f"File {source} does not exist")
            return None

        logger.info(f"Loading JSON: {source}")

        with open(source, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except Exception as err:
                logger.error(f"Error parsing JSON: {err}")
                return None

        return Document(
            source=source,
            text=json.dumps(data, indent=2),
            metadata={"type": "json"}
        )

    # ---------------------------
    # Load Folder
    # ---------------------------
    def folder(self, path: str) -> List[Document]:
        """Load all files in a folder"""
        docs = []

        logger.info(f"Loading folder: {path}")

        for filename in os.listdir(path):
            file_path = os.path.join(path, filename)

            if filename.endswith(".pdf"):
                doc = self.pdf(file_path)

            elif filename.endswith(".txt"):
                doc = self.txt(file_path)

            elif filename.endswith(".json"):
                doc = self.json(file_path)

            elif filename.endswith(".html"):
                doc = self.url("file://" + file_path)

            else:
                logger.warning(f"Skipping unsupported file type: {filename}")
                continue

            if doc:
                docs.append(doc)

        logger.info(f"Loaded {len(docs)} documents from {path}")
        return docs
