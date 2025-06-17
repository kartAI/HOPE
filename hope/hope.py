import logging
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import BaseLLM

from hope.common.dataclasses import ChunkData, DocumentMeta, HopeScore
from hope.common.exceptions import ChunkingException
from hope.scoring import (
    ConceptUnityScorer,
    InformationPreservationScorer,
    SemanticIndependenceScorer,
)


logger = logging.getLogger()


class Hope:
    def __init__(
        self,
        llm: BaseChatModel | BaseLLM,
        embedding_model: Embeddings,
        multi_process: bool = False,
    ):
        """The main class for calculating the HOPE score.

        Args:
            llm (BaseChatModel | BaseLLM): An LLM to be used for calculating the HOPE score. Currently only OpenAI GPT models and Llama 3.1 models are tested.
            embedding_model (Embeddings): An embedding model to be used for calculating HOPE scores.
            multi_process (bool, optional): Process sub-scores in parallel. Defaults to False.
        """
        self._MULTI_PROCESS = multi_process

        self._concept_unity_scorer = ConceptUnityScorer(
            llm=llm, embedding_model=embedding_model
        )
        self._information_preservation_scorer = InformationPreservationScorer(
            llm=llm, embedding_model=embedding_model
        )
        self._semantic_dependence_scorer = SemanticIndependenceScorer(
            llm=llm, embedding_model=embedding_model
        )

        self.calculate_concept_unity = self._concept_unity_scorer.evaluate
        self.calculate_information_preservation = (
            self._information_preservation_scorer.evaluate
        )
        self.calculate_semantic_independence = self._semantic_dependence_scorer.evaluate

        self.scorers: dict[str, callable] = {
            self._concept_unity_scorer.score_name: self.calculate_concept_unity,
            self._information_preservation_scorer.score_name: self.calculate_information_preservation,
            self._semantic_dependence_scorer.score_name: self.calculate_semantic_independence,
        }

        self.score_storage: dict[str, HopeScore] = dict()

    def reset_scores(self):
        self.score_storage = dict()

    def get_scores(
        self, as_pandas: bool = False
    ) -> pd.DataFrame | dict[str, HopeScore]:
        if as_pandas:
            data = [
                {
                    "Document": k,
                    "Hope score": v.hope,
                    "Conceptualism score": v.conceptualism,
                    "Information preservation score": v.information_preservation,
                    "Semantic independence score": v.semantic_independence,
                    "Number of chunks": v.document_meta.num_chunks,
                    "Number of words": v.document_meta.num_words,
                    "Number of characters": v.document_meta.num_characters,
                    "Mean chunk length": v.document_meta.chunk_length_mean,
                    "Chunk length std": v.document_meta.chunk_length_std,
                    "Chunks MD5 IDs": v.document_meta.chunks_md5_ids,
                }
                for k, v in self.score_storage.items()
            ]

            return pd.DataFrame(data)

        return self.score_storage

    def evaluate(
        self,
        document: Document,
        chunks: list[ChunkData],
        document_name: str,
        skip_concept_unity: bool = False,
        skip_information_preservation: bool = False,
        skip_semantic_independence: bool = False,
    ) -> HopeScore:
        """Evaluate the HOPE score for a provided document and chunks.

        Args:
            document (Document): The original document before chunking.
            chunks (list[ChunkData]): The chunks produced by chunking the "document".
            document_name (str): The name of the document.
            skip_concept_unity (bool, optional): Skip calculating concept unity. Defaults to False.
            skip_information_preservation (bool, optional): Skip calculating information preservation . Defaults to False.
            skip_semantic_independence (bool, optional): Skip calculating semantic independence. Defaults to False.
        """

        assert isinstance(document, Document), "document must be of type Document"
        assert all(
            isinstance(chunk, ChunkData) for chunk in chunks
        ), "all chunks must be of type ChunkData"

        if len(chunks) < 2:
            raise ChunkingException(
                f"Document must be chunked into at least 2 sections to evaluate Hope score. Got {len(chunks)} sections."
            )

        logger.info(
            f'Evaluating Hope score for document "{document_name}" with {len(chunks)} chunks'
        )

        if not self._MULTI_PROCESS:
            logger.debug("Evaluating Hope scores sequentially")
            if not skip_concept_unity:
                logger.debug("Evaluating Conceptualism score")
                conceptualism = self.calculate_concept_unity(
                    document=document, chunks=chunks
                )
                logger.debug("Conceptualism score: %s", conceptualism)
            else:
                conceptualism = None

            if not skip_information_preservation:
                logger.debug("Evaluating Information Preservation score")
                information_preservation = self.calculate_information_preservation(
                    document=document, chunks=chunks
                )
                logger.debug(
                    "Information Preservation score: %s", information_preservation
                )
            else:
                information_preservation = None

            if not skip_semantic_independence:
                logger.debug("Evaluating Semantic Independence score")
                semantic_independence = self.calculate_semantic_independence(
                    document=document, chunks=chunks
                )
                logger.debug("Semantic Independence score: %s", semantic_independence)
            else:
                semantic_independence = None

            hope_score = HopeScore(
                conceptualism=conceptualism,
                information_preservation=information_preservation,
                semantic_independence=semantic_independence,
            )
        else:
            with ThreadPoolExecutor() as pool:
                hope_score = HopeScore(
                    **{
                        score_name: score
                        for score_name, score in zip(
                            self.scorers.keys(),
                            pool.map(
                                lambda scorer: scorer.evaluate(
                                    document=document, chunks=chunks
                                ),
                                self.scorers.values(),
                            ),
                        )
                    }
                )

        hope_score.document_meta = DocumentMeta.from_chunks_and_document(
            chunks=[chunk.text for chunk in chunks], document=document.page_content
        )

        self.score_storage[document_name] = hope_score
        return hope_score

    __call__ = evaluate
