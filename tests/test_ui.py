"""
test_ui
~~~~~~~

Tests for the pjinoise.ui module.
"""
import unittest as ut
from unittest.mock import call, patch

from pjinoise import ui


class StatusTestCase(ut.TestCase):
    @patch('time.time', return_value=0)
    @patch('pjinoise.ui.print')
    def test_start_and_end_script(self, mock_print, mock_time):
        filename = 'spam'
        start = ui.TEXT['start'].format(min=0, sec=0, filename=filename)
        end = ui.TEXT['end'].format(min=1, sec=1, filename=filename)
        exp = [
            call(start),
            call(end),
        ]
        
        status = ui.Status(filename)
        mock_time.return_value = 61
        status.end()
        act = mock_print.mock_calls
        
        self.assertListEqual(exp, act)
    
    @patch('time.time', return_value=0)
    @patch('pjinoise.ui.print')
    def test_update_status(self, mock_print, mock_time):
        filename = 'spam'
        start = ui.TEXT['start'].format(min=0, sec=0, filename=filename)
        update = ui.TEXT['noise'].format(1, 1)
        exp = [
            call(start),
            call(update),
        ]
        
        text_key = 'noise'
        status = ui.Status(filename)
        mock_time.return_value = 61
        status.update(text_key)
        act = mock_print.mock_calls
        
        self.assertListEqual(exp, act)