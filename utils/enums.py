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
    SUPPORT_DUMP = "SUPPORT_DUMP"
    VIDEO = "VIDEO"
    MODEL = "MODEL"
    SPLIT = "SPLIT"


class SimpleSplits(Enum):
    TRAIN = "train"
    VAL = "validation"
    TEST = "test"


class ComposedSplit(Enum):
    TRAINVAL = "trainval"


class ChallengeSplits(Enum):
    VAL_CHALLENGE = "validation_challenge"
    TEST_CHALLENGE = "test_challenge"
