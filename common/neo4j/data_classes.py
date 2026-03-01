from abc import ABC, abstractmethod
from typing import List


class Executor(ABC):
    @abstractmethod
    def get_installation_schema(self) -> List[str]:
        pass