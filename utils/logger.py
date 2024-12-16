import logging
from rich.console import Console
from rich.logging import RichHandler


def setup_logging(level=logging.INFO):
    console = Console()

    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )
