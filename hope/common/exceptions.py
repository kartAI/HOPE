class InvalidInputException(Exception):
    pass


class ChunkingException(Exception):
    pass


class UnableToGenerateQuestionsException(Exception):
    pass


class TooManyInvokeRetriesException(Exception):
    pass


class ExternalLlmProviderException(Exception):
    pass
