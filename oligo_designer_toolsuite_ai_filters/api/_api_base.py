import abc
from typing import Any


class APIBase:

    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    def predict(self, *args: Any, **kwargs: Any) -> Any:
        pass
