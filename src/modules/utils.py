"""
Utility functions for the UCI Online Retail project.
"""

from pathlib import Path
from typing import Any, Dict, Union
import yaml


def read_yaml(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Read a YAML file and return its contents as a dictionary.
    Args:
        file_path: Path to the YAML file (string or Path object)
    Returns:
        Dictionary containing the parsed YAML content
    Raises:
        FileNotFoundError: If the YAML file does not exist
        yaml.YAMLError: If the YAML file is malformed
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"YAML file not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config
