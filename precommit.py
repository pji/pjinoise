#! /usr/local/bin/python3
"""
precommit
~~~~~~~~~

Things that should be done before committing changes to the repo.
"""
import glob
import pycodestyle as pcs
import unittest as ut
import sys

# Remove trailing whitespace.
def remove_whitespace(filename):
    with open(filename, 'r') as fh:
        lines = fh.readlines()
    newlines = [line.rstrip() for line in lines]
    newlines = [line + '\n' for line in newlines]
    with open(filename, 'w') as fh:
        fh.writelines(newlines)

tests = glob.glob('tests/*')
tests = [name for name in tests if name.endswith('.py')]
for file in tests:
    remove_whitespace(file)

files = glob.glob('pjinoise/*')
files = [name for name in files if name.endswith('.py')]
for file in files:
    remove_whitespace(file)

# Run the unit tests.
loader = ut.TestLoader()
tests = loader.discover('tests')
runner = ut.TextTestRunner()
runner.run(tests)

# Check the code style.
style = pcs.StyleGuide(config_file='setup.cfg')
style.check_files(['pjinoise/',])
style.check_files(['tests/',])