#! .venv/bin/python
"""
precommit
~~~~~~~~~

Things that should be done before committing changes to the repo.
"""
import doctest
import glob
from itertools import zip_longest
import os
import sys
import unittest as ut
from textwrap import wrap

import pycodestyle as pcs
import rstcheck

from pjinoise import common
from pjinoise import ease
from pjinoise import filters
from pjinoise import model
from pjinoise import operations
from pjinoise import sources


# Script configuration.
doctest_modules = [
    common,
    ease,
    filters,
    model,
    operations,
    sources,
]
ignore = []
python_files = [
    './*',
    'tests/*',
    'pjinoise/*',
    'pjinoise/sources/*',
]
rst_files = [
    './*',
    'docs/*',
]
unit_tests = 'tests'


# Checks.
def check_doctests(modules):
    """Run documentation tests."""
    print('Running doctests...')
    for mod in modules:
        doctest.testmod(mod)
    print('Doctests complete.')


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


def check_rst(file_paths):
    """Remove trailing whitespace."""
    def action(files):
        results = []
        for file in files:
            with open(file) as fh:
                lines = fh.read()
            result = list(rstcheck.check(lines))
            if result:
                results.append(file, *result)
        return results

    def result_handler(result):
        if result:
            for line in result:
                print(' ' * 4 + line)

    title = 'Checking RSTs'
    file_ext = '.rst'
    run_check_on_files(title, action, file_paths, file_ext, result_handler)


def check_style(file_paths):
    """Remove trailing whitespace."""
    def result_handler(result):
        if result.get_count():
            for msg in result.result_messages:
                lines = wrap(msg, 78)
                print(' ' * 4 + lines[0])
                for line in lines[1:]:
                    print(' ' * 6 + line)
            result.result_messages = []

    class StyleReport(pcs.BaseReport):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.result_messages = []

        def error(self, line_number, offset, text, check):
            super().error(line_number, offset, text, check)
            msg = (f'{self.filename} {line_number}:{offset} {text}')
            self.result_messages.append(msg)

    title = 'Checking style'
    style = pcs.StyleGuide(config_file='setup.cfg', reporter=StyleReport)
    action = style.check_files
    file_ext = '.py'
    run_check_on_files(title, action, file_paths, file_ext, result_handler)


def check_unit_tests(path):
    """Run the unit tests."""
    print('Running unit tests...')
    loader = ut.TestLoader()
    tests = loader.discover(path)
    runner = ut.TextTestRunner()
    result = runner.run(tests)
    print('Unit tests complete.')
    return result


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


def check_whitespace(file_paths):
    """Remove trailing whitespace."""
    title = 'Checking whitespace'
    action = remove_whitespace
    file_ext = '.py'
    run_check_on_files(title, action, file_paths, file_ext)


# Utility functions.
def in_ignore(name):
    for item in ignore:
        if name.endswith(item):
            return True
    return False


def run_check_on_files(title, action, file_paths,
                       file_ext=None, result_handler=None):
    print(f'{title}...')
    for file_path in file_paths:
        print(' ' * 2 + f'Checking {file_path}...', end='')
        files = glob.glob(file_path)
        if file_ext:
            files = [name for name in files if name.endswith(file_ext)]
        if ignore:
            files = [name for name in files if not in_ignore(name)]
        result = action(files)
        print('. Done.')
        if result and result_handler:
            result_handler(result)
    print(f'{title} complete.')
    return result


def remove_whitespace(filename):
    if isinstance(filename, (list, tuple)):
        for item in filename:
            remove_whitespace(item)
    else:
        with open(filename, 'r') as fh:
            lines = fh.readlines()
        newlines = [line.rstrip() for line in lines]
        newlines = [line + '\n' for line in newlines]
        with open(filename, 'w') as fh:
            fh.writelines(newlines)


def main():
    with open('./.gitignore') as fh:
        lines = fh.readlines()
    for line in lines:
        if line.endswith('\n'):
            line = line[:-1]
        if line:
            ignore.append(line)

    check_venv()
    check_whitespace(python_files)
    result = check_unit_tests(unit_tests)

    # Only continue with precommit checks if the unit tests passed.
    if not result.errors and not result.failures:
        check_requirements()
        check_doctests(doctest_modules)
        check_style(python_files)
        check_rst(rst_files)

    else:
        print('Unit tests failed. Precommit checks aborted. Do not commit.')


main()
