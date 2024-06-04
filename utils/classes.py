from utils.enums import LogCode, SourceMode
from datetime import datetime


class LogManager:
    """Class responsible for managing operation logs

    Attributes:
            show (bool): if True, logs are printed during operations.
            log_file (TextIOWrapper): if not None, logs are saved here.
            verbose (bool): if True, logs are detailed.
    """

    def __init__(self, show=True, path: str = None, new_file=False, verbose=True):
        """Initialize the LogManager object.

        Args:
            show (bool): if True, logs are printed during operations.
            path (str): if not None, represent log_file path
            new_file (str): if path is not None, specify if an eventual old log_file is cleared or not
        """

        self.verbose = verbose
        self.show = show

        if path is None:
            self.log_file = None
            return
        try:
            self.log_file = open(path, "w" if new_file else "a")
        except:
            self.log_file = None
            self.path = None

    def log(
        self, source: str, code: LogCode, sub_code: int = None, message: str = None
    ):
        """Register the log message

        Args:
            source (str): log source
            code (LogCode): log code
            sub_code (int): log subcode
            message (str): log specification message
        """

        now = datetime.now()
        sub_code_str = "_" + str(sub_code) if sub_code is not None else ""
        log_message = f"{source} - {code.value}{sub_code_str}"

        if self.verbose:
            log_message = f"{now} - {log_message}"
            log_message += " - " + message if message is not None else ""
        else:
            now = now.strftime("%d/%m/%Y %H:%M:%S")
            log_message = f"{now} - {log_message}"

        if self.show:
            print(log_message)

        if self.log_file is not None:
            self.log_file.write(f"{log_message}\n")


class Source:
    """Class responsible for storing data source information.

    Attributes:
            path (str): file path where the data source is stored
    """

    def __init__(self, mode: SourceMode, path: str = None):
        """Initialize the Source object.

        Args:
            extrinsic_source (utils.classes.Source): Object containing information about the extrinsic parameters source.
            position_source (utils.classes.Source): Path to the camera positions JSON file.
        """

        self.mode = mode
        self.path = path
