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


from hope.prompt import prompt_template_mc_generator, SchemaMultipleChoice
from hope.prompt.json_schema.schema_mc import SchemaMultipleChoice

logger = logging.getLogger(__name__)


class MultipleChoiceGenerator:
    def __init__(
        self,
        generator_llm: BaseChatModel | BaseLLM,
    ):
        self._generator_llm = LlmWrapper(generator_llm)
        self._parser = JsonOutputParser()

        self._generator_chain = prompt_template_mc_generator | self._generator_llm

    @property
    def MAX_RETRIES(self):
        return 5

    def generate(self, document: Document | str) -> SchemaMultipleChoice | None:

        text = document.page_content if isinstance(document, Document) else document
        assert isinstance(text, str), "Document should be a string or Document object"

        for _ in range(self.MAX_RETRIES):
            try:
                response_raw = self._generator_chain.invoke(
                    {"reference_text": text},
                    extra_body={"guide_json": SchemaMultipleChoice.model_json_schema()},
                )

                if not response_raw:
                    continue

                # try:
                response_raw = self._response_post_processing_2(response_raw)
                response: dict = self._parser.parse(response_raw)
                statements = SchemaMultipleChoice(**response)

                if len(statements.false_statements) != 3:
                    raise ValueError(
                        "Incorrect number of false statements: expected 3 \n"
                        + response_raw
                    )

                if not statements.derived_statement:
                    raise ValueError("No correct statement found \n" + response_raw)

                break

            except Exception as e:
                logger.error(f"Failed to parse response: {response_raw}")
                logger.error(e)
                continue
        else:
            return None

        return statements

    @staticmethod
    def _response_post_processing_2(response: str) -> str:
        out = response
        if "´´´" in out:
            out = out.replace("´´´", "")
            out = out.replace("True", "true").replace("False", "false")
            out = out.replace("json", "")
            out = out.strip()
        out = out.encode("ascii", "replace").decode("ascii")
        return out

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


if __name__ == "__main__":
    from langchain_openai.chat_models import AzureChatOpenAI

    ragas_llm = AzureChatOpenAI(
        model=hope_config.openai_llm.deployment,
        api_key=hope_config.openai_llm.api_key,
        azure_endpoint=hope_config.openai_llm.endpoint,
        api_version=hope_config.openai_llm.api_version,
    )

    data = dedent(
        """
    In the realm of artificial intelligence, machine learning has emerged as a pivotal technology, driving advancements across various sectors.
    Machine learning algorithms enable computers to learn from data and make predictions or decisions without being explicitly programmed.
    This transformative capability is harnessed in applications ranging from healthcare and finance to autonomous vehicles and natural language processing.
    As data continues to proliferate, the importance of robust and scalable machine learning models becomes increasingly evident.
    Researchers and practitioners are continually exploring innovative techniques to enhance the accuracy, efficiency, and interpretability of these models, paving the way for a future where intelligent systems seamlessly integrate into everyday life.
    """
    )
    res = MultipleChoiceGenerator(ragas_llm).generate(data)

    print(res)
    print(type(res))
