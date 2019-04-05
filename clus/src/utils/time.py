def retrieve_decimals(number, number_of_decimals_wanted=2):
    """ Return the decimals of a number.

    Examples:
    >>> retrieve_decimals(765.4815162342, number_of_decimals_wanted=3)
    "481"
    """
    return str(round(number % 1, number_of_decimals_wanted)).split(".")[-1]


def pretty_time_delta(seconds):
    """ Pretty print a time delta in days, hours, minutes and seconds.

    Source :
    * https://gist.github.com/thatalextaylor/7408395
    """

    milliseconds = retrieve_decimals(seconds, number_of_decimals_wanted=2)
    seconds = int(seconds)
    days, seconds = divmod(seconds, 86400)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    milliseconds, seconds, days, hours, minutes = \
        milliseconds.zfill(2), str(seconds).zfill(2), str(days), str(hours).zfill(2), str(minutes).zfill(2)
    if int(days) > 0:
        return "{days}d{hours}h{minutes}m{seconds}s{milliseconds}ms".format(
            days=days, hours=hours, minutes=minutes, seconds=seconds, milliseconds=milliseconds
        )
    elif int(hours) > 0:
        return "{hours}h{minutes}m{seconds}s{milliseconds}ms".format(
            hours=hours, minutes=minutes, seconds=seconds, milliseconds=milliseconds
        )
    elif int(minutes) > 0:
        return "{minutes}m{seconds}s{milliseconds}ms".format(
            minutes=minutes, seconds=seconds, milliseconds=milliseconds
        )
    else:
        return "{seconds}s{milliseconds}ms".format(
            seconds=seconds, milliseconds=milliseconds
        )


if __name__ == '__main__':
    pass
