'''
Test file in the early stage for checking table content

Auther: Hanchen Wang
Date: 2025-05
'''

import sqlite3
import unittest
import os
import importlib.util

script_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../scripts/INFG-simulate.py'))
spec = importlib.util.spec_from_file_location("INFG_simulate", script_path)
INFG_simulate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(INFG_simulate)

META_GAME_TRANSITIONS = INFG_simulate.META_GAME_TRANSITIONS

FILE = "my-experiment1.db"
FILEPATH = f"../scripts/{FILE}"


class TestDBFile(unittest.TestCase):

    def test_metadata_rows(self):
        """
        Test the .db file has correct number of rows in 'metadata' table
        """
        expected_num = len(META_GAME_TRANSITIONS)
        conn = sqlite3.connect(FILEPATH)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM 'metadata';")
        actual_num = len(cursor.fetchall())
        conn.close()

        self.assertTrue((actual_num == expected_num),
                        f"The number of row in metadata table is unexpected: {actual_num}")

    def test_timeseries_rows(self):
        """
        Test the .db file has correct number of rows in 'timeseries' table
        """
        expected_num = len(META_GAME_TRANSITIONS)
        conn = sqlite3.connect(FILEPATH)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM 'timeseries';")
        actual_num = len(cursor.fetchall())
        conn.close()

        self.assertTrue((actual_num == expected_num),
                        f"The number of row in timeseries table is unexpected: {actual_num}")


if __name__ == '__main__':
    unittest.main()
