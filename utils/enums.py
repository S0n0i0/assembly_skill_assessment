from enum import Enum


class LogCode(Enum):
    SUCCESS = "SUCCESS"
    WARNING = "WARNING"
    ERROR = "ERROR"


class SourceMode(Enum):
    DUMP = "DUMP"


class SensorMode(Enum):
    OFFLINE_DUMP = "OFFLINE_DUMP"
    OFFLINE = "OFFLINE"
    ONLINE = "ONLINE"