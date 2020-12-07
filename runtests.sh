#! /bin/sh
#####
# runtests.sh: shortcut for running Python unit tests.
# Paul J. Iutzi
# 2020.11.28    v0.2
#   * Added style checking.
# 2020.11.15    v0.1
#   * Initial implementation.
#####
# Run unit tests
python3 -m unittest discover tests

# Run style check
python3 -m pycodestyle ./pjinoise
python3 -m pycodestyle ./tests