from typing import Sequence, Generic, TypeVar
import abc
import dataclasses

Element = TypeVar("Element")

class Distribution(Generic[Element]):
    support: Sequence[Element]
    def log_probability(self, element : Element) -> float:
        pass
    def sample(self) -> Element:
        pass


@dataclasses.dataclass
class ProgramMessage:
    pass

class ObserveMessage(ProgramMessage):
    distribution: Distribution
    value: any

class SampleMessage(ProgramMessage):
    distribution: Distribution

@abc.ABC
class ProgramState(object):
    message: ProgramMessage
    def step(value):
        pass
    def is_returned():
        pass
    def return_value():
        pass
    def __hash__(self):
        pass
