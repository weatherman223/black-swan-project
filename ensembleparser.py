"""
extract_ensembles.py

Reads a text file containing NetCDF paths—one per line—and writes the
unique ensemble-name patterns (with a * where the dates go) to stdout.

Usage
-----
python extract_ensembles.py filelist.txt   # prints to screen
python extract_ensembles.py filelist.txt > ensembles.txt   # save to file
"""
import re
import sys
from pathlib import Path

PATTERN = re.compile(r"^(.*?\.)\d{8}-\d{8}\.nc$")   # capture everything up to first 8-digit date

def main(path):
    found = set()
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            m = PATTERN.match(line)
            if m:                       # keep only lines that match the expected pattern
                found.add(f"{m.group(1)}*.nc")

    for ensemble in sorted(found):
        print(ensemble)

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Usage: python extract_ensembles.py <file_with_names.txt>")
    main(sys.argv[1])