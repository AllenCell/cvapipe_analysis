import numpy as np
from datetime import datetime

class Printer:

    style = {
        "m": "\033[95m",
        "b": "\033[94m",
        "c": "\033[96m",
        "g": "\033[92m",
        "y": "\033[93m",
        "r": "\033[91m",
        "e": "\033[0m",
        "d": "\033[1m",
        "u": "\033[4m"}

    hierarchy = ["udy", "udg", "udc", "um", "b"]

    def __init__(self) -> None:
        pass

    def cprint(self, text, args=0):
        tab = "\t"
        if isinstance(args, int):
            tab = "".join(["  "]*args)
            args = np.min([args, len(self.hierarchy)-1])
            args = self.hierarchy[args]
        time = datetime.now().strftime('%H:%M:%S')
        time = f"{self.style['d']}[{time}]{self.style['e']}"
        style = "".join([v for k, v in self.style.items() if k in args])
        msg = f"{time}{tab}{style}{text}{self.style['e']}"
        print(msg)
