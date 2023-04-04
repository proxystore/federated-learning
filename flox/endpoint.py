from enum import auto, Enum

class EndpointKind(Enum):
    local = auto()
    remote = auto()