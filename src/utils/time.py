from datetime import datetime

DATE_FMT = "%Y-%m-%d %H:%M:%S.%f %Z"


def iso_to_datetime(iso_str: str, date_format: str = DATE_FMT) -> datetime:
    """Converts a date in iso_str format to datetime object

    Args:
        iso_str: Date string in iso format, e.g. 2015-05-23 23:54:00.123 UTC
        date_format: Format of the date string to be parsed

    Returns:
        datetime object that represents the input date
    """
    return datetime.strptime(iso_str, date_format)


def hour_of_iso_date(iso_str: str, date_format: str = DATE_FMT) -> int:
    return iso_to_datetime(iso_str, date_format).hour


def robust_hour_of_iso_date(iso_str: str, date_format: str = DATE_FMT) -> int:
    try:
        return hour_of_iso_date(iso_str, date_format)
    except:
        return hour_of_iso_date(iso_str, "%Y-%m-%d %H:%M:%S %Z")
