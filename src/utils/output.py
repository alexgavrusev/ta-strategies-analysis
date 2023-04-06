import inspect
import os
from pathlib import Path
from typing import Any, Dict
import backtrader as bt


def get_strategy_name(strategy: bt.Strategy) -> str:
    """
    Get a string representation
    for a strategy with its parameters
    """
    strategy_name = strategy.__class__.__name__.replace("Strategy", "")

    def params_to_string(strategy_params: Dict[str, Any]) -> str:
        def rec(dictionary: Dict[str, Any], acc: str) -> str:
            result = []

            for key, value in dictionary.items():
                if isinstance(value, dict):
                    result.append(rec(value, acc))
                elif inspect.isclass(value):
                    result.append(f"{key}={value.__name__}")
                else:
                    result.append(f"{key}={value}")

            return f'{acc}{",".join(result)}'

        return rec(strategy_params.__dict__, "")

    string_params = params_to_string(strategy.params)

    return f"{strategy_name}({string_params})@{strategy.data._name}"


def rm_rec(dir_path: Path):
    """
    Remove all files and directories
    in a given directory
    """

    for item in os.listdir(dir_path):
        item_path = os.path.join(dir_path, item)

        if os.path.isfile(item_path):
            os.remove(item_path)
        else:
            os.rmdir(item_path)


def ensure_empty_dir(dir_path: Path):
    """
    Ensure that the directory under the specified path
    exists and is empty
    """
    dir_path.mkdir(parents=True, exist_ok=True)
    rm_rec(dir_path)
