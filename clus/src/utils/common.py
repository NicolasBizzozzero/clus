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


if __name__ == "__main__":
    pass
