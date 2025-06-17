from langchain.prompts import PromptTemplate
from hope.common.config import hope_config
from hope.prompt.json_schema.schema_qna import SchemaQnaGenerator, SchemaQnaCritic
from hope.prompt.json_schema.schema_mc import SchemaMultipleChoice

__all__ = [
    "prompt_template_rag",
    "prompt_template_qna_generator",
    "prompt_template_qna_critic",
    "SchemaQnaGenerator",
    "SchemaQnaCritic",
    "SchemaMultipleChoice",
]

prompt_template_rag: PromptTemplate
prompt_template_qna_generator: PromptTemplate
prompt_template_qna_critic: PromptTemplate
prompt_template_mc_generator: PromptTemplate
prompt_template_rag_mc: PromptTemplate


match hope_config.prompt.prompt_style:

    case "llama-3.1":
        from hope.prompt._prompt_templates.llama_rag import (
            prompt_template_rag as llama_prompt_template_rag,
        )
        from hope.prompt._prompt_templates.llama_qna import (
            prompt_template_qna_generator as llama_prompt_template_qna_generator,
            prompt_template_qna_critic as llama_prompt_template_qna_critic,
        )
        from hope.prompt._prompt_templates.llama_mc import (
            prompt_template_mc_generator as llama_prompt_template_mc_generator,
        )
        from hope.prompt._prompt_templates.llama_rag_mc import (
            prompt_template_rag_mc as llama_prompt_template_rag_mc,
        )

        prompt_template_rag = llama_prompt_template_rag
        prompt_template_qna_generator = llama_prompt_template_qna_generator
        prompt_template_qna_critic = llama_prompt_template_qna_critic
        prompt_template_mc_generator = llama_prompt_template_mc_generator
        prompt_template_rag_mc = llama_prompt_template_rag_mc

    case "openai":
        from hope.prompt._prompt_templates.openai_rag import (
            prompt_template_rag as openai_prompt_template_rag,
        )
        from hope.prompt._prompt_templates.openai_qna import (
            prompt_template_qna_generator as openai_prompt_template_qna_generator,
            prompt_template_qna_critic as openai_prompt_template_qna_critic,
        )
        from hope.prompt._prompt_templates.openai_mc import (
            prompt_template_mc_generator as openai_prompt_template_mc_generator,
        )

        prompt_template_rag = openai_prompt_template_rag
        prompt_template_qna_generator = openai_prompt_template_qna_generator
        prompt_template_qna_critic = openai_prompt_template_qna_critic
        prompt_template_mc_generator = openai_prompt_template_mc_generator

    case _:
        from typing import get_args

        f = lambda x: f"'{x}'"

        raise ValueError(
            f"Invalid prompt style: '{hope_config.prompt.prompt_style}'. Valid options are: {', '.join(map(f, get_args(hope_config.prompt._allowed_prompt_styles)))}."
        )
