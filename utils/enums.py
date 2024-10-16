from enum import Enum


class DisplayLevel(Enum):
    """Enum class for verbose mode"""

    NONE = 0
    ONLY_ERRORS = 1
    LOW = 2
    MEDIUM = 3
    HIGH = 4


class LogCode(Enum):
    SUCCESS = "SUCCESS"
    WARNING = "WARNING"
    ERROR = "ERROR"


class SensorMode(Enum):
    OFFLINE_DUMP = 0
    OFFLINE = 1
    ONLINE = 2


class SourceMode(Enum):
    DUMP = "DUMP"
    DUMP_REF = "DUMP_REF"
    SUPPORT_DUMP = "SUPPORT_DUMP"
    VIDEO = "VIDEO"
    VIDEO_REF = "VIDEO_REF"
    MODEL = "MODEL"


class PathType(Enum):
    FILE = "FILE"
    DIR = "DIRECTORY"
