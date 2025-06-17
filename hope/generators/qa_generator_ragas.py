import nest_asyncio

nest_asyncio.apply()

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import BaseLLM
from langchain_core.embeddings import Embeddings

from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.testset import TestsetGenerator, Testset

from hope.common import hope_config


class QAGeneratorRagas(object):
    def __init__(
        self,
        generator_llm: BaseChatModel | BaseLLM,
        critic_llm: BaseChatModel | BaseLLM,
        embedding_model: Embeddings,
    ):
        self._ragas_llm = LangchainLLMWrapper(generator_llm)
        self._ragas_embedding = LangchainEmbeddingsWrapper(embedding_model)

        self._generator = TestsetGenerator(
            llm=self._ragas_llm,
        )

    def generate_qa(
        self,
        document: str | Document | list[str] | list[Document],
        num_questions: int = 5,
    ) -> Testset:

        if isinstance(document, Document):
            document = [document]

        elif isinstance(document, str):
            document = [Document(page_content=document)]

        if isinstance(document[0], str):
            document = [Document(page_content=doc) for doc in document]

        dataset = self._generator.generate_with_langchain_docs(
            documents=document,
            testset_size=num_questions,
            transforms_llm=self._ragas_llm,
            transforms_embedding_model=self._ragas_embedding,
            # # is_async=False,
            # raise_exceptions=False,
            # run_config=RunConfig(
            #     max_wait=60,
            #     max_retries=2,
            #     max_workers=100,  # Adjust the number of workers
            #     timeout=180,
            #     # timeouts={"connect_timeout": 10, "read_timeout": 30},  # Set timeouts
            #     # rate_limits={"requests_per_minute": 60},  # Set rate limits
            # ),
        )

        return dataset

    __call__ = generate_qa


if __name__ == "__main__":
    from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
    from langchain_community.document_loaders import DirectoryLoader

    generator = AzureChatOpenAI(
        model=hope_config.openai_llm.deployment,
        api_key=hope_config.openai_llm.api_key,
        azure_endpoint=hope_config.openai_llm.endpoint,
        api_version=hope_config.openai_llm.api_version,
    )
    critic = AzureChatOpenAI(
        model=hope_config.openai_llm.deployment,
        api_key=hope_config.openai_llm.api_key,
        azure_endpoint=hope_config.openai_llm.endpoint,
        api_version=hope_config.openai_llm.api_version,
    )
    embeddings = AzureOpenAIEmbeddings(
        model=hope_config.openai_embedding.deployment,
        api_key=hope_config.openai_embedding.api_key,
        azure_endpoint=hope_config.openai_embedding.endpoint,
        api_version=hope_config.openai_embedding.api_version,
    )

    qa_generator = QAGeneratorRagas(generator, critic, embeddings)

    document_loader = DirectoryLoader(r"/workspaces/phd-HOPE/dummy_files")

    documents = document_loader.load()

    print(f"Num documents loaded: {len(documents)}")
    print(f'Documents loaded: {" - ".join([str(d.metadata) for d in documents])}')

    qa_dataset = qa_generator.generate_qa(documents, num_questions=10)

    qa_dataset.to_pandas().to_csv("qa_dataset.csv", index=False)
