import colorlog
import logging
colorlog.basicConfig(
    format="%(log_color)s[%(levelname)s] %(asctime)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.DEBUG,
    log_colors={
        "DEBUG": "blue",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "bold_red",
    },
)

print("hello")
logging.info("hello")