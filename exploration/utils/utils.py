def number_to_str(number: int) -> str:
    return (number < 10) * "0" + str(number)
