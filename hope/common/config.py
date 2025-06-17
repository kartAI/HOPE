import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

from hope.common import Singleton

p = Path(__file__).parent.parent.parent.resolve()
load_dotenv(p / ".env")


class LoggingConfig(metaclass=Singleton):
    def __init__(self):
        self.logging_level = os.getenv("LOGGING_LEVEL", "INFO")
        self.logger_base_directory = Path(
            os.getenv("LOGGER_BASE_DIRECTORY", "/workspaces/phd-HOPE/logs")
        )


class HopeConfig(metaclass=Singleton):
    def __init__(self):
        self.prompt = PromptConfig()
        self.scoring = ScoringConfig()


class PromptConfig:
    _allowed_prompt_styles = Literal["openai", "llama-3.1"]

    def __init__(self):
        self.prompt_style: self._allowed_prompt_styles = os.getenv(
            "PROMPT_STYLE", "openai"
        )


class ScoringConfig:
    def __init__(self):
        self.conceptualism = ConceptUnitConfig()
        self.semantic_dependence = SemanticDependenceConfig()
        self.information_preservation = InformationPreservationConfig()
        self.hope = HopeScoreConfig()


class HopeScoreConfig:
    def __init__(self):
        self.conceptualism_weight = float(
            os.getenv("HOPE_SCORE_CONCEPTUALISM_WEIGHT", 1.0)
        )
        self.information_preservation_weight = float(
            os.getenv("HOPE_SCORE_INFORMATION_PRESERVATION_WEIGHT", 1.0)
        )
        self.semantic_independence_weight = float(
            os.getenv("HOPE_SCORE_SEMANTIC_INDEPENDENCE_WEIGHT", 1.0)
        )


class ConceptUnitConfig:
    def __init__(self):
        self.num_questions = int(
            os.getenv("CONCEPT_UNITY_NUM_QUESTIONS_TO_GENERATE", 5)
        )


class SemanticDependenceConfig:
    def __init__(self):
        self.num_questions = int(
            os.getenv("SEMANTIC_DEPENDENCE_NUM_QUESTIONS_TO_GENERATE", 5)
        )


class InformationPreservationConfig:
    def __init__(self):
        self.num_segments = int(os.getenv("INFORMATION_PRESERVATION_NUM_SEGMENTS", 3))
        self.num_questions = int(
            os.getenv(
                "INFORMATION_PRESERVATION_NUM_QUESTIONS_TO_GENERATE_PER_SEGMENT", 5
            )
        )
        self.segment_size = int(os.getenv("INFORMATION_PRESERVATION_SEGMENT_SIZE", 2))
        self.num_chunks_to_retrieve = int(
            os.getenv("INFORMATION_PRESERVATION_NUM_CHUNKS_TO_RETRIEVE", 5)
        )


logging_config = LoggingConfig()
hope_config = HopeConfig()
