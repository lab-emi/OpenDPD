"""Keep free text inert when CSV exports are opened in spreadsheet software."""
import csv


def safe_cell(value):
    if isinstance(value, str) and value.lstrip(" \t\r\n").startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


class SafeWriter:
    def __init__(self, stream, **kwargs):
        self.writer = csv.writer(stream, **kwargs)

    def writerow(self, values):
        return self.writer.writerow([safe_cell(value) for value in values])

    def writerows(self, rows):
        for row in rows:
            self.writerow(row)
