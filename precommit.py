#! .venv/bin/python
"""
precommit
~~~~~~~~~

Things that should be done before committing changes to the repo.
"""
import doctest
import glob
from itertools import zip_longest
import unittest as ut
import sys
import os

import pycodestyle as pcs
import rstcheck

from pjinoise import ease
from pjinoise import filters
from pjinoise import model
from pjinoise import operations
from pjinoise import sources


# Script configuration.
python_files = [
    'tests/*',
    'pjinoise/*',
    'pjinoise/sources/*',
    'mazer.py',
    'template.py',
    'precommit.py',
]
unit_tests = 'tests'


# Checks.
def check_venv():
    """Ensure this is running from the virtual environment for
    pjinoise. I know this is a little redundant with the shebang
    line at the top, but debugging issues caused by running from
    the wrong venv are a giant pain.
    """
    venv_path = '.venv/bin/python'
    dir_delim = '/'
    cwd = os.getcwd()
    exp_path = cwd + dir_delim + venv_path
    act_path = sys.executable
    if exp_path != act_path:
        msg = (f'precommit run from unexpected python: {act_path}. '
               f'Run from {exp_path} instead.')
        raise ValueError(msg)


def check_requirements():
    """Check requirements."""
    print('Checking requirements...')
    current = os.popen('.venv/bin/python -m pip freeze').readlines()
    with open('requirements.txt') as fh:
        old = fh.readlines()

    # If the packages installed don't match the requirements, it's
    # likely the requirements need to be updated. Display the two
    # lists to the user, and let them make the decision whether
    # to freeze the new requirements.
    if current != old:
        print('requirements.txt out of date.')
        print()
        tmp = '{:<30} {:<30}'
        print(tmp.format('old', 'current'))
        for c, o in zip_longest(current, old, fillvalue=''):
            print(tmp.format(c[:-1], o[:-1]))
        print()
        update = input('Update? [y/N]: ')
        if update.casefold() == 'y':
            os.system('.venv/bin/python -m pip freeze > requirements.txt')
    print('Requirements checked...')


def check_unit_tests(path):
    """Run the unit tests."""
    print('Running unit tests...')
    loader = ut.TestLoader()
    tests = loader.discover(path)
    runner = ut.TextTestRunner()
    result = runner.run(tests)
    print('Unit tests complete.')
    return result


def check_whitespace(check_list):
    """Remove trailing whitespace."""
    print('Checking whitespace...')
    for path in check_list:
        print(f'Removing whitespace from {path}...', end='')
        files = glob.glob(path)
        files = [name for name in files if name.endswith('.py')]
        for file in files:
            remove_whitespace(file)
        print('. Done.')
    print('Whitespace checked.')


# Utility functions.
def remove_whitespace(filename):
    with open(filename, 'r') as fh:
        lines = fh.readlines()
    newlines = [line.rstrip() for line in lines]
    newlines = [line + '\n' for line in newlines]
    with open(filename, 'w') as fh:
        fh.writelines(newlines)


# Main.
check_venv()
check_whitespace(python_files)
result = check_unit_tests(unit_tests)

# Only continue with precommit checks if the unit tests passed.
if not result.errors and not result.failures:
    check_requirements()

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

    # Check the style of the RST docs.
    print('Checking RSTs...')
    files = glob.glob('docs/*')
    files = [name for name in files if name.endswith('.rst')]
    for file in files:
        with open(file) as fh:
            lines = fh.read()
        results = list(rstcheck.check(lines))
        for result in results:
            print(file, *result)
    print('RSTs checked.')


else:
    print('Unit tests failed. Precommit checks aborted. Do not commit.')
