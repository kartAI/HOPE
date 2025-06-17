import logging
from concurrent.futures import ThreadPoolExecutor
from textwrap import dedent

from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import BaseLLM
from langchain_core.output_parsers import JsonOutputParser

from hope.common import hope_config
from hope.common.dataclasses import Question
from hope.common.exceptions import UnableToGenerateQuestionsException
from hope.utils.llm import LlmWrapper
from hope.prompt import (
    prompt_template_qna_generator,
    prompt_template_qna_critic,
    SchemaQnaCritic,
    SchemaQnaGenerator,
)

logger = logging.getLogger(__name__)


class QuestionsGenerator:
    def __init__(
        self,
        generator_llm: BaseChatModel | BaseLLM,
        critic_llm: BaseChatModel | BaseLLM | None = None,
    ):
        self._generator_llm = LlmWrapper(
            generator_llm  # .bind(response_format={"type": "json_object"})
        )
        self._critic_llm = LlmWrapper(critic_llm or generator_llm)
        self._parser = JsonOutputParser()

        self._generator_chain = prompt_template_qna_generator | self._generator_llm
        self._critic_chain = prompt_template_qna_critic | self._critic_llm

    @property
    def MAX_RETRIES(self):
        return 5

    def generate(
        self,
        document: Document | str,
        num_questions_to_generate: int = 5,
    ) -> list[Question]:

        text = document.page_content if isinstance(document, Document) else document
        assert isinstance(text, str), "Document should be a string or Document object"

        questions = []

        for _ in range(self.MAX_RETRIES):
            response_raw = self._generator_chain.invoke(
                {
                    "number_of_questions": str(num_questions_to_generate),
                    "reference_text": text,
                },
                extra_body={"guide_json": SchemaQnaGenerator.model_json_schema()},
            )

            if not response_raw:
                continue
            try:
                response_raw = self._response_post_processing(response_raw)
                response: dict = self._parser.parse(response_raw)
                questions_data = response.get("questions")
                new_questions = [Question(**q) for q in questions_data]
            except Exception:
                # logger.error(f"Failed to parse response: {response_raw}")
                # logger.error(e)
                continue

            new_questions = [q for q in new_questions if q.question and q.answer]

            questions.extend(new_questions)

            if len(questions) >= num_questions_to_generate:
                break

        else:
            raise UnableToGenerateQuestionsException(
                f"Failed to generate enough questions ({len(questions)} < {num_questions_to_generate}) . Max number of retries reached ({self.MAX_RETRIES})."
            )

        with ThreadPoolExecutor() as executor:
            results = executor.map(self._critic, ((q, text) for q in questions))

        processed_questions = [q for q, res in zip(questions, results) if res]

        if len(processed_questions) == 0:
            raise UnableToGenerateQuestionsException(
                f"Failed to generate eny questions"
            )

        return processed_questions

    def _critic(self, *args) -> bool:
        if len(args) > 0 and isinstance(args[0], tuple):
            args = args[0]

        assert (
            len(args) == 2
        ), f"Critic function should have 2 arguments, got {len(args)}"
        return self.critic(*args)

    @staticmethod
    def _response_post_processing(response: str) -> str:
        out = response.replace("True", "true").replace("False", "false")
        out = "{" + out.split("{", maxsplit=1)[1]
        out = out.rsplit("}", maxsplit=1)[0] + "}"
        out = out.replace("'", '"')

        lines = out.split("\n")
        lines_out = []
        for i, line in enumerate(lines):
            if line.count('"') > 4:
                line = line.replace('"', "'").replace("'", '"', 3)
                line = line.rsplit("'", maxsplit=1)[0] + '"'
            if i != len(lines) - 1:
                if not lines[i + 1].strip().startswith(("}", "]")) and not (
                    line.strip().startswith(("{", "["))
                    or line.strip().endswith(("{", "["))
                ):
                    if not line.strip().endswith(","):
                        line += ","
                else:
                    if line.strip().endswith(","):
                        line = line.rsplit(",", maxsplit=1)[0]

            lines_out.append(line)
        out = "".join(lines_out)
        out = out.replace("\t", "").strip()
        out = out.replace(r"{{", "{").replace(r"}}", "}")
        out = out.replace("*", "")

        return out

    def critic(self, question: Question, reference_text: str) -> bool:
        is_relevant = False

        for _ in range(self.MAX_RETRIES):
            response_raw = self._critic_chain.invoke(
                {"question": question.question, "reference_text": reference_text},
                extra_body={"guide_json": SchemaQnaCritic.model_json_schema()},
            )
            if not response_raw:
                continue

            try:
                response_raw = self._response_post_processing(response_raw)
                response: dict = self._parser.parse(response_raw)
                is_relevant = response.get("is_relevant", False)
                break
            except Exception:
                # logger.error(f"Failed to parse response: {response_raw}")
                # logger.error(e)
                continue

        return is_relevant


# text = """
# ,
# ,
# ```json,
# {
#     "is_relevant": true,
#     "reason": "The reference text explicitly mentions 'Atlas' as a retrieval-augmented language model that trains both the retriever and the language model, directly answering the question."
# }
# """

# out = QuestionsGenerator._response_post_processing(text)
# out = JsonOutputParser().parse(out)
# print(out)


if __name__ == "__main__":

    from textwrap import dedent

    from langchain_community.document_loaders import DirectoryLoader
    from langchain_openai import AzureChatOpenAI

    generator = AzureChatOpenAI(
        model=hope_config.openai_llm.deployment,
        api_key=hope_config.openai_llm.api_key,
        azure_endpoint=hope_config.openai_llm.endpoint,
        api_version=hope_config.openai_llm.api_version,
    )

    qa_generator = QuestionsGenerator(generator, generator)

    data = dedent(
        """
    In the realm of artificial intelligence, machine learning has emerged as a pivotal technology, driving advancements across various sectors.
    Machine learning algorithms enable computers to learn from data and make predictions or decisions without being explicitly programmed.
    This transformative capability is harnessed in applications ranging from healthcare and finance to autonomous vehicles and natural language processing.
    As data continues to proliferate, the importance of robust and scalable machine learning models becomes increasingly evident.
    Researchers and practitioners are continually exploring innovative techniques to enhance the accuracy, efficiency, and interpretability of these models, paving the way for a future where intelligent systems seamlessly integrate into everyday life.
    """
    )

    questions = qa_generator.generate(data)
    for q in questions:
        print(q.question)
