from unittest import TestCase

import statstock
from statstock.tests import tests_dir

class YahooTest(TestCase):
    def load_data_basic_on_valid_input_test(self):
        data = statstock.data.Yahoo(tests_dir + "/data/sample.csv", "sample")

    def load_data_basic_on_invalid_input_missing_header_test(self):
        data = statstock.data.Yahoo(tests_dir + "/data/sample_no_header.csv", "sample")
