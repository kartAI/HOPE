import logging
from random import choice
from textwrap import dedent
from pathlib import Path

from langchain.prompts import PromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import BaseLLM
from langchain_core.output_parsers import JsonOutputParser

from hope.prompt import prompt_template_rag, prompt_template_rag_mc
from hope.prompt.json_schema.schema_mc import SchemaMultipleChoice
from hope.prompt.json_schema.schema_rag_mc import SchemaMultipleChoiceRag
from hope.utils.llm import LlmWrapper

logger = logging.getLogger(__name__)

error_file_path: Path | None = None
for handler in logger.handlers:
    if isinstance(handler, logging.FileHandler):
        error_file_path = Path(handler.baseFilename).parent / "llm.log"
        break


class LlmAnswererRag:
    def __init__(
        self, llm: BaseChatModel | BaseLLM, prompt_template: PromptTemplate = None
    ):
        llm = LlmWrapper(llm)
        prompt_template = prompt_template or prompt_template_rag
        self._chain = prompt_template | llm

    def answer_question(self, question: str, reference: list[str]) -> str:
        res = self._chain.invoke(
            {"question": question, "reference": "\n\n".join(reference)}
        )
        if res:
            return res if isinstance(res, str) else res.content
        return ""


class LlmAnswererRagMc:
    def __init__(
        self, llm: BaseChatModel | BaseLLM, prompt_template: PromptTemplate = None
    ):
        llm = LlmWrapper(llm)
        prompt_template = prompt_template or prompt_template_rag_mc
        self._chain = prompt_template | llm
        self._parser = JsonOutputParser()

    def select_statement(
        self, statements: SchemaMultipleChoice, reference: list[str]
    ) -> bool | None:

        all_indexes = [1, 2, 3, 4]
        correct_index = choice(all_indexes)

        all_statements = [
            *statements.false_statements,
        ]
        all_statements.insert(correct_index - 1, statements.derived_statement)

        input_data = {
            "reference": dedent("\n\n".join(reference)),
            "statements": dedent(
                "\n".join(
                    f'{ind}. "{stat}"' for ind, stat in zip(all_indexes, all_statements)
                )
            ),
        }

        for temp in range(5):
            model_response_raw = self._chain.invoke(
                input_data,
                extra_body={"guide_json": SchemaMultipleChoiceRag.model_json_schema()},
                temperature=temp / 10,
            )

            if not model_response_raw:
                continue

            try:
                response_raw = self._response_post_processing(model_response_raw)
                response: dict = self._parser.parse(response_raw)
                res = response.get("correct_statement")
            except Exception as e:
                continue

            if res is None:
                continue

            if isinstance(res, str):
                if not res.isnumeric():
                    continue

                res = int(res.strip())

            return int(res) == correct_index

        else:
            if error_file_path:
                error_file_path.write_text(
                    f"Failed to parse response: {model_response_raw}\n"
                )

            # logger.error(f"Error while parsing response: {e}")
            # logger.error(f"Response model: {model_response_raw}")
            # logger.error(f"Response processed: {response_raw}")
            # logger.error(f"Input data: {input_data}")
            # logger.error(
            # f"===========================================================================000"
            # )
            return None

    @staticmethod
    def _response_post_processing(response: str) -> str:
        out = response.replace("True", "true").replace("False", "false")
        out = "{" + out.split("{")[-1]
        out = out.rsplit("}")[0] + "}"
        out = out.replace("'", '"')
        out = out.replace("\t", "").strip()
        out = out.replace(r"{{", "{").replace(r"}}", "}")
        out = out.replace("*", "")
        lines = out.split("\n")
        lines_out = []
        for line in lines:
            if line.count('"') > 4:
                line = line.replace('"', "'").replace("'", '"', 3)
                line = line.rsplit("'", maxsplit=1)[0] + '"'

            lines_out.append(line)
        out = "\n".join(lines_out)
        return out
