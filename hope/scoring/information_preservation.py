import logging
from concurrent.futures import ThreadPoolExecutor
from copy import copy
from itertools import count

from librosa import ex
import numpy as np
from hope.common.dataclasses import ChunkData
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import BaseLLM
from nltk.tokenize import sent_tokenize

from hope.common import hope_config
from hope.common.exceptions import UnableToGenerateQuestionsException
from hope.generators import MultipleChoiceGenerator
from hope.predictors import LlmAnswererRagMc
from hope.prompt.json_schema.schema_mc import SchemaMultipleChoice
from hope.scoring.base import BaseScorer
from hope.utils.vectorstore import ChromaVectorStore

logger = logging.getLogger(__name__)


class InformationPreservationScorer(BaseScorer):
    def __init__(
        self,
        llm: BaseChatModel | BaseLLM,
        embedding_model: Embeddings,
    ):
        super().__init__(
            score_name="information_preservation",
            llm=llm,
            embedding_model=embedding_model,
        )
        self._predictor = LlmAnswererRagMc(llm)
        self._statement_generator = MultipleChoiceGenerator(llm)
        self._vector_store = ChromaVectorStore(
            embedding_model=embedding_model,
            db_name=self.score_name + "_vector_store",
        )

        self._num_segments_generated = count()
        self._num_evaluations = count()

        self._cfg = hope_config.scoring.information_preservation

    def _sample_segments(self, text: str) -> list[str]:

        sentences = sent_tokenize(text)

        segment_size = self._cfg.segment_size
        num_segments = min(self._cfg.num_segments, len(sentences) - segment_size)

        if len(sentences) < segment_size:
            return [text for _ in range(num_segments)]

        indexes = np.random.choice(
            len(sentences) - segment_size, num_segments, replace=False
        )

        segments = [". ".join(sentences[i : i + segment_size]) + "." for i in indexes]
        return segments

    def _generate_statements(self, text: str) -> list[SchemaMultipleChoice]:

        def generate_statement(sample: str) -> SchemaMultipleChoice | None:
            try:
                statements = self._statement_generator.generate(sample)

                next(self._num_segments_generated)
                return statements
            except UnableToGenerateQuestionsException:
                return None

        with ThreadPoolExecutor() as executor:
            statements = list(
                executor.map(generate_statement, self._sample_segments(text))
            )

        return [s for s in statements if s is not None]

    @property
    def num_segments_generated(self):
        return next(copy(self._num_segments_generated)) - 1

    @property
    def question_generated_ratio(self):
        return max(
            self.num_segments_generated
            / (self._cfg.num_questions * self._cfg.num_segments),
            0.0,
        )

    def _evaluate_statements(self, statement: SchemaMultipleChoice) -> float:

        relevant_data = self._vector_store.query(
            statement.derived_statement, top_k=self._cfg.num_chunks_to_retrieve
        )
        reference = [d.page_content for d in relevant_data]
        ans_is_correct = self._predictor.select_statement(statement, reference)

        if ans_is_correct is None:
            return None

        return 1.0 if ans_is_correct else 0.0

    def evaluate(self, document: Document, chunks: list[ChunkData]) -> float:
        self._vector_store.add_data(chunks)
        scores = []
        statements: list[SchemaMultipleChoice] = []

        for n in range(5):
            statements.extend(self._generate_statements(document.page_content))

            if len(statements) >= self._cfg.num_segments:
                break

        else:
            logger.warning(
                f"Unable to generate {self._cfg.num_segments} statements after {n} attempts ({len(statements)} generated)."
            )
            if len(statements) == 0:
                return 0.0

        with ThreadPoolExecutor() as executor:
            scores = list(executor.map(self._evaluate_statements, statements))

        scores = [s for s in scores if s is not None]

        if len(scores) == 0:
            logger.warning("No scores were generated. All answers were ´None´.")
            return 0.0

        return np.mean(scores)


if __name__ == "__main__":
    from langchain_community.document_loaders import DirectoryLoader
    from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings

    generator = AzureChatOpenAI(
        model=hope_config.openai_llm.deployment,
        api_key=hope_config.openai_llm.api_key,
        azure_endpoint=hope_config.openai_llm.endpoint,
        api_version=hope_config.openai_llm.api_version,
    )
    embedding_model = AzureOpenAIEmbeddings(
        model=hope_config.openai_embedding.deployment,
        api_key=hope_config.openai_embedding.api_key,
        azure_endpoint=hope_config.openai_embedding.endpoint,
        api_version=hope_config.openai_embedding.api_version,
    )

    scorer = InformationPreservationScorer(
        llm=generator, embedding_model=embedding_model
    )

    document = next(DirectoryLoader(r"/workspaces/phd-HOPE/dummy_files").lazy_load())

    chunks = [
        ChunkData(
            text="Henrik, Nicole, and Aurora were three friends who lived in a small town nestled between the mountains and the sea. Henrik was an adventurous soul, always seeking new challenges and experiences. He loved hiking up the steep mountain trails and exploring the dense forests that surrounded their town. Nicole, on the other hand, was a brilliant scientist with a passion for marine biology. She spent most of her days at the local marine research center, studying the diverse marine life that thrived in the nearby ocean. Aurora was an artist, known for her beautiful paintings that captured the essence of their picturesque town. She found inspiration in the natural beauty that surrounded them and often accompanied Henrik on his hikes to sketch the breathtaking landscapes. "
        ),
        ChunkData(
            text="One summer, the trio decided to embark on an ambitious project that combined their unique talents. They planned to create a comprehensive documentary about the natural wonders of their town, showcasing the stunning mountain vistas, the vibrant marine ecosystem, and the artistic interpretations of these landscapes. Henrik took charge of the adventure segments, leading the team on thrilling expeditions to capture the most awe-inspiring views. Nicole provided scientific insights, explaining the intricate details of the marine life they encountered. Aurora documented their journey through her art, creating a visual narrative that tied everything together."
        ),
        ChunkData(
            text="Their project quickly gained attention, and soon they were joined by a team of filmmakers and researchers who were eager to contribute. Together, they spent months exploring every corner of their town, from the highest peaks to the deepest ocean trenches. They encountered rare species, discovered hidden waterfalls, and even unearthed ancient artifacts that hinted at the town's rich history. As they pieced together their documentary, they realized that their project was not just about showcasing the natural beauty of their town, but also about highlighting the importance of preserving these wonders for future generations."
        ),
        ChunkData(
            text="The documentary premiered at a local film festival and received widespread acclaim. It was praised for its stunning visuals, informative content, and the heartfelt passion that Henrik, Nicole, and Aurora had poured into it. The success of their project brought the community closer together and sparked a renewed interest in environmental conservation. Henrik, Nicole, and Aurora continued to work on similar projects, using their talents to inspire others and make a positive impact on the world around them.        "
        ),
    ]

    res_full = scorer.evaluate(document, chunks=chunks)
    res_partial = scorer.evaluate(document, chunks=chunks[:2])

    print("Full score:", res_full)
    print("Partial score:", res_partial)
