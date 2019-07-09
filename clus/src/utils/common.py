def str_to_number(string):
    try:
        return int(string)
    except ValueError:
        return float(string)


def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def is_integer(s):
    try:
        return int(s) == s
    except ValueError:
        return False


if __name__ == "__main__":
    pass
