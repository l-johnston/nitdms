"""Test versioning"""
import os

# pylint: disable=missing-docstring
def test_version():
    file = "./nitdms/version.py"
    with open(file) as f:
        lines = f.readlines()
    cmd_line = f"python {file} 1 2 3"
    with os.popen(cmd_line) as cmd:
        result = cmd.read().strip()
    with open(file, "wt") as f:
        f.writelines(lines)
    assert result == '__version__ = "1.2.3"'
