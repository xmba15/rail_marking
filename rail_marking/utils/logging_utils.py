#!/usr/bin/env python
import logging


__all__ = ["config_logging", "CustomFormatter"]


class CustomFormatter(logging.Formatter):
    def format(self, record, *args, **kwargs):
        import copy

        LOG_COLORS = {
            logging.INFO: "\x1b[33m",
            logging.DEBUG: "\x1b[36m",
            logging.WARNING: "\x1b[31m",
            logging.ERROR: "\x1b[31;1m",
            logging.CRITICAL: "\x1b[35m",
        }

        new_record = copy.copy(record)
        if new_record.levelno in LOG_COLORS:
            new_record.levelname = "{color_begin}{level}{color_end}".format(
                level=new_record.levelname,
                color_begin=LOG_COLORS[new_record.levelno],
                color_end="\x1b[0m",
            )
        return super(CustomFormatter, self).format(new_record, *args, **kwargs)


def config_logging(log_file, log_level=logging.DEBUG):
    format_line = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    custom_formatter = CustomFormatter(format_line)
    stream_handler = logging.StreamHandler()
    file_handler = logging.FileHandler(log_file)

    stream_handler.setFormatter(custom_formatter)

    logging.basicConfig(handlers=[file_handler, stream_handler], level=log_level, format=format_line)
