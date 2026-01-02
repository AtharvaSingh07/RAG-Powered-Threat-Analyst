from llama_index.core.node_parser import SentenceSplitter, TokenTextSplitter
from loguru import logger
from typing import List


class TextSplitter:
    def __init__(self, chunk_size: int = 512, overlap: int = 20, method: str = "sentence"):
        """
        method: "sentence" or "token"
        """
        if method == "sentence":
            self.splitter = SentenceSplitter(
                chunk_size=chunk_size,
                chunk_overlap=overlap
            )
        elif method == "token":
            self.splitter = TokenTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=overlap
            )
        else:
            raise ValueError("Invalid method: choose 'sentence' or 'token'")

    def split(self, text: str) -> List[str]:
        logger.info(f"Splitting text (len: {len(text)})")
        chunks = self.splitter.get_nodes_from_documents([text])[0].get_content()
        logger.info(f"Chunks: {len(chunks)}")
        return chunks
