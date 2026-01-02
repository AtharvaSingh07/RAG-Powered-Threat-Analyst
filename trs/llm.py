import os
import requests
from loguru import logger
from .schema import Document, Summary
from typing import Optional


PROMPT_DIR = os.path.abspath(os.path.join(os.path.abspath('.'), 'prompts'))


class LLM:
    def __init__(self, model: str = "llama3") -> None:
        """
        Local LLM wrapper using Ollama.
        No API keys needed.
        """
        self.model = model
        self.token_limit = 128000   # still useful for safety
        
        # Check if Ollama is running
        try:
            r = requests.get("http://localhost:11434/api/tags")
            if r.status_code != 200:
                raise Exception("Ollama not responding")
        except Exception as err:
            logger.error(f"Error connecting to local LLM (Ollama): {err}")
            raise

    def num_tokens(self, text: str) -> int:
        """
        Naive token approximation (we don't need exact tiktoken anymore).
        """
        if not text:
            return 0
        return len(text.split())

    def _read_prompt(self, prompt_name: str) -> Optional[str]:
        path = os.path.join(PROMPT_DIR, f"{prompt_name}.txt")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
        logger.error(f"Prompt not found: {path}")
        return None

    def _call_llama(self, user_prompt: str, system_prompt: Optional[str] = None) -> Optional[str]:
        system_prompt = system_prompt or "You are a helpful AI cybersecurity assistant."

        # Build combined prompt
        full_prompt = f"<|system|>\n{system_prompt}\n\n<|user|>\n{user_prompt}"

        # Token safety (approx only)
        if self.num_tokens(full_prompt) > self.token_limit:
            logger.error("Token limit exceeded.")
            return None

        # Call local Llama3
        try:
            payload = {
                "model": self.model,
                "prompt": full_prompt,
                "stream": False
            }

            r = requests.post("http://localhost:11434/api/generate", json=payload)

            if r.status_code != 200:
                logger.error(f"LLM error: {r.text}")
                return None

            return r.json().get("response", None)

        except Exception as err:
            logger.error(f"Error running local Llama: {err}")
            return None

    def _generic_prompt(self, prompt_name: str, doc: Document) -> Optional[str]:
        template = self._read_prompt(prompt_name)
        if template:
            user_prompt = template.format(document=doc.text)
            return self._call_llama(user_prompt=user_prompt)
        return None

    # ---------- Public Methods ---------- #

    def mindmap(self, doc: Document) -> Optional[str]:
        return self._generic_prompt("mindmap", doc)

    def summarize(self, doc: Document) -> Optional[Summary]:
        summary = self._generic_prompt("summary", doc)
        if summary:
            return Summary(source=doc.source, summary=summary)
        return None

    def detect(self, doc: Document) -> Optional[str]:
        return self._generic_prompt("detect", doc)

    def qna(self, question: str, docs: str) -> Optional[str]:
        template = self._read_prompt("qna")
        if template:
            user_prompt = template.format(question=question, documents=docs)
            return self._call_llama(user_prompt=user_prompt)
        return None

    def custom(self, prompt_name: str, doc: Document) -> Optional[str]:
        return self._generic_prompt(prompt_name, doc)
