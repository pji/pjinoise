"""
test_ui
~~~~~~~

Tests for the pjinoise.ui module.
"""
import unittest as ut
from unittest.mock import call, patch

from pjinoise import ui


class StatusTestCase(ut.TestCase):
    @patch('pjinoise.ui.print')
    def test_start_and_end_script(self, mock_print):
        filename = 'spam'
        start = ui.TEXT['start'].format(min=0, sec=0, filename=filename)
        end = ui.TEXT['end'].format(min=0, sec=0, filename=filename)
        exp = [
            call(start),
            call(end),
        ]
        
        status = ui.Status(filename)
        status.end(filename)
        act = mock_print.mock_calls
        
        self.assertListEqual(exp, act)