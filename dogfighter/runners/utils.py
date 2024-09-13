import os


class AtomicFileWriter:
    def __init__(self, name: str) -> None:
        split_name = name.split(".")
        if len(split_name) > 1:
            suffix = ".".join(split_name[:-1])
            prefix = f".{split_name[-1]}"
        else:
            suffix = name
            prefix = ""

        self.temp_name = f"{suffix}_tmp{prefix}"
        self.name = name

    def __enter__(self):
        return self.temp_name

    def __exit__(self, *_):
        os.rename(self.temp_name, self.name)
