import logging
from logging.handlers import RotatingFileHandler
from logging import _nameToLevel, _levelToName
from datetime import datetime

from hope.common.config import logging_config


def logger_setup(
    application_name: str,
    log_level: str | int | None = None,
    console_print: bool = True,
    console_print_debug: bool = False,
    *loggers_to_set_to_warning: str,
) -> None:

    log_level = log_level or logging_config.logging_level
    if isinstance(log_level, str):
        try:
            log_level = _nameToLevel[log_level.upper()]
        except KeyError:
            raise ValueError(f"Invalid log level: {log_level}")

    logfile_folder_path = (
        logging_config.logger_base_directory
        / application_name
        / datetime.now().strftime("%Y%m%d-%H%M%S")
    )

    if not logfile_folder_path.exists():
        logfile_folder_path.mkdir(parents=True)

    handlers = [
        RotatingFileHandler(
            str(logfile_folder_path / f"{application_name}.log"),
            maxBytes=1_000_000,
            backupCount=5,
        )
    ]
    if console_print:
        console_handler = logging.StreamHandler()
        console_log_level = logging.DEBUG if console_print_debug else logging.INFO
        console_handler.setLevel(console_log_level)
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        handlers.append(console_handler)

    # Configure logging
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers,
    )

    logging.getLogger("Logger setup").info(
        f"Logger setup complete. Level: {_levelToName[log_level]} ({log_level}), application: {application_name}."
    )

    for logger_name in loggers_to_set_to_warning:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
