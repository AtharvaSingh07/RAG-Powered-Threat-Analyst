from typing import List, Tuple, Union, Dict
from chromadb import PersistentClient, Settings
from loguru import logger
from uuid import uuid4

from sentence_transformers import SentenceTransformer


class VectorDB:
    def __init__(self, collection_name: str, db_dir: str, n_results: int = 5):
        """
        Offline Vector DB using:
        - ChromaDB (persistent)
        - SentenceTransformers (local embedding)
        """
        self.collection_name = collection_name
        self.db_dir = db_dir
        self.n_results = n_results

        # --------------------------
        # Local Embedding Model
        # --------------------------
        logger.info("Loading local embedding model (MiniLM-L6)...")
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")  # CPU friendly
        logger.success("Loaded local embedding model")

        # --------------------------
        # Initialize Vector DB
        # --------------------------
        self.client = PersistentClient(
            path=self.db_dir,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            ),
        )

        self.collection = self.get_or_create_collection(self.collection_name)
        logger.success("Loaded local ChromaDB collection")

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Compute embeddings locally using SentenceTransformers"""
        return self.embed_model.encode(texts, show_progress_bar=False).tolist()

    def get_or_create_collection(self, name: str):
        logger.info(f"Using collection: {name}")

        try:
            return self.client.get_collection(name)
        except Exception:
            logger.info("Collection not found. Creating new one.")

        return self.client.create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"}
        )

    def count(self) -> int:
        return self.collection.count()

    def get(self) -> Dict[str, List[Union[str, List[float], dict]]]:
        logger.info("Getting all documents")
        return self.collection.get()

    # --------------------------------------------
    # ADD DOCUMENTS (Auto-Embeds Locally)
    # --------------------------------------------
    def add_texts(self, texts: List[str], metadatas: List[dict]) -> Tuple[bool, List[str]]:
        logger.info(f"Adding {len(texts)} texts to vector DB")
        ids = [str(uuid4()) for _ in range(len(texts))]
        success = False

        try:
            embeddings = self.embed(texts)

            self.collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            success = True

        except Exception as err:
            logger.error(f"Failed to add texts: {err}")

        return success, ids

    # --------------------------------------------
    # ADD PRECOMPUTED EMBEDDINGS
    # --------------------------------------------
    def add_embeddings(self, texts: List[str], embeddings: List[List[float]], metadatas: List[dict]) -> Tuple[bool, List[str]]:
        logger.info(f"Adding {len(texts)} precomputed embeddings")
        ids = [str(uuid4()) for _ in range(len(texts))]
        success = False

        try:
            self.collection.add(
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            success = True
        except Exception as err:
            logger.error(f"Failed to add embeddings: {err}")

        return success, ids

    # --------------------------------------------
    # QUERY VECTOR DB (Local)
    # --------------------------------------------
    def query(self, text: str) -> List[dict]:
        logger.info(f"Querying DB for: {text}")

        try:
            query_embedding = self.embed([text])[0]

            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=self.n_results
            )

            flattened = []
            for node_id, txt, metadata, distance in zip(
                    results["ids"][0],
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
            ):
                flattened.append({
                    "id": node_id,
                    "text": txt,
                    "metadata": metadata,
                    "distance": distance
                })

            logger.info(f"Found {len(flattened)} results")
            return flattened

        except Exception as err:
            logger.error(f"DB query failed: {err}")
            return []
