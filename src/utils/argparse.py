import argparse


def parse_config(default_config: str, description: str = "Run script with config"):
    """
    Parse command-line arguments to get the configuration file path.

    Args:
        default_config (str): Default path to the configuration file.
        description (str): Description for the argument parser.

    Returns:
        str: Path to the configuration file.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--config_path",
        type=str,
        default=default_config,
        help=f"Path to the configuration file (default: {default_config})",
    )
    args = parser.parse_args()
    return args.config_path
