#! .venv/bin/python
"""
precommit
~~~~~~~~~

Things that should be done before committing changes to the repo.
"""
import doctest
import glob
import pycodestyle as pcs
import unittest as ut
import sys
import os

from pjinoise import ease
from pjinoise import filters
from pjinoise import model
from pjinoise import operations
from pjinoise import sources


# Ensure this is running from the virtual environment for pjinoise.
# I know this is a little redundant with the shebang line at the top,
# but debugging issues caused by running from the wrong venv are a
# giant pain.
venv_path = '.venv/bin/python'
dir_delim = '/'
cwd = os.getcwd()
exp_path = cwd + dir_delim + venv_path
act_path = sys.executable
if exp_path != act_path:
    msg = (f'precommit run from unexpected python: {act_path}. '
           f'Run from {exp_path} instead.')
    raise ValueError(msg)

# Remove trailing whitespace.
def remove_whitespace(filename):
    with open(filename, 'r') as fh:
        lines = fh.readlines()
    newlines = [line.rstrip() for line in lines]
    newlines = [line + '\n' for line in newlines]
    with open(filename, 'w') as fh:
        fh.writelines(newlines)

print('Removing whitespace...')
tests = glob.glob('tests/*')
tests = [name for name in tests if name.endswith('.py')]
for file in tests:
    remove_whitespace(file)

files = glob.glob('pjinoise/*')
files = [name for name in files if name.endswith('.py')]
for file in files:
    remove_whitespace(file)

files = glob.glob('pjinoise/sources/*')
files = [name for name in files if name.endswith('.py')]
for file in files:
    remove_whitespace(file)

remove_whitespace('mazer.py')
remove_whitespace('template.py')
print('Whitespace removed.')

# Run the unit tests.
print('Running unit tests...')
loader = ut.TestLoader()
tests = loader.discover('tests')
runner = ut.TextTestRunner()
result = runner.run(tests)
print('Unit tests complete.')

# Only continue with precommit checks if the unit tests passed.
if not result.errors and not result.failures:
    # Freeze requirements.
    print('Freezing requirements...')
    os.system('python -m pip freeze > requirements.txt')
    print('Requirements frozen...')

    # Run documentation tests.
    print('Running doctests...')
    mods = [
        ease,
        filters,
        model,
        operations,
        sources,
    ]
    for mod in mods:
        doctest.testmod(mod)
    print('Doctests complete.')

    # Check the code style.
    print('Checking style...')
    style = pcs.StyleGuide(config_file='setup.cfg')
    style.check_files(['pjinoise/',])
    style.check_files(['tests/',])
    style.check_files(['mazer.py',])
    style.check_files(['template.py',])
    print('Style checked.')

else:
    print('Unit tests failed. Precommit checks aborted. Do not commit.')