from langchain_classic.output_parsers.regex import RegexParser


def load_output_parser(config: dict) -> dict:
    """加载一个输出解析器。

    Args:
        config: 配置字典

    Returns:
        包含已加载输出解析器的配置字典
    """
    if "output_parsers" in config and config["output_parsers"] is not None:
        _config = config["output_parsers"]
        output_parser_type = _config["_type"]
        if output_parser_type == "regex_parser":
            output_parser = RegexParser(**_config)
        else:
            msg = f"Unsupported output parser {output_parser_type}"
            raise ValueError(msg)
        config["output_parsers"] = output_parser
    return config
