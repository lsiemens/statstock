import os
import unittest

import numpy

import statstock

class TestStockData(unittest.TestCase):
    data_dir = os.path.abspath(os.path.dirname(__file__) + "/example_data")

    def setUp(self):
        self.stockdata = statstock.data.StockData(self.data_dir + "/sample.csv", "smp")
        self.data = numpy.array([43.9687, 43.9687, 44.2187, 44.4062, 44.9687, 44.9687, 44.9687, 44.8125, 44.6562, 44.7812])
        self.time = numpy.array([7.28294400e+08, 7.28553600e+08, 7.28640000e+08, 7.28726400e+08, 7.28812800e+08, 7.28899200e+08, 7.29158400e+08, 7.29244800e+08, 7.29331200e+08, 7.29417600e+08])

    def test_stockdata_attribute_path(self):
        self.assertTrue(os.path.samefile(self.stockdata.path, self.data_dir + "/sample.csv"))

    def test_yahoo_attribute_ticker(self):
        self.assertEqual(self.stockdata.ticker, "smp")

    def test_stockdata_method_price_to_relative(self):
        expected_data = numpy.array([1.0, 1.0, 1.00568586, 1.00424029, 1.01266715, 1.0, 1.0, 0.99652647, 0.99651213, 1.00279916])
        result = self.stockdata.price_to_relative(self.data)
        self.assertTrue(numpy.allclose(result, expected_data))

    def test_stockdata_method_price_to_normalized(self):
        expected_data = numpy.array([1.0, 0.99909619, 1.00731204, 1.00522128, 1.01742799, 0.99909619, 0.99909619, 0.99408728, 0.99406662, 1.00313828])
        result = self.stockdata.price_to_normalized(self.time, self.data)
        self.assertTrue(numpy.allclose(result, expected_data))

class TestYahoo(unittest.TestCase):
    data_dir = os.path.abspath(os.path.dirname(__file__) + "/example_data")

    def test_yahoo_attribute_path(self):
        data = statstock.data.Yahoo(self.data_dir + "/sample.csv", "smp")
        self.assertTrue(os.path.samefile(data.path, self.data_dir + "/sample.csv"))

    def test_yahoo_attribute_ticker(self):
        data = statstock.data.Yahoo(self.data_dir + "/sample.csv", "smp")
        self.assertEqual(data.ticker, "smp")

    def test_yahoo_attribute_data(self):
        expected_data = {'adj_close': numpy.array([27.607176, 27.803539, 27.862408, 28.156963, 28.274775, 28.255106, 28.255106, 28.058754, 28.098019, 28.235523]),
                         'close': numpy.array([43.9375, 44.25, 44.3437, 44.8125, 45.0, 44.9687, 44.9687, 44.6562, 44.7187, 44.9375]),
                         'high': numpy.array([43.9687, 44.25, 44.375, 44.8437, 45.0937, 45.0625, 45.125, 44.8125, 44.75, 45.125]),
                         'low': numpy.array([43.75, 43.9687, 44.125 , 44.375 , 44.4687, 44.7187, 44.9062, 44.5625, 44.5312, 44.7812]),
                         'open': numpy.array([43.9687, 43.9687, 44.2187, 44.4062, 44.9687, 44.9687, 44.9687, 44.8125, 44.6562, 44.7812]),
                         'time': numpy.array([7.28294400e+08, 7.28553600e+08, 7.28640000e+08, 7.28726400e+08, 7.28812800e+08, 7.28899200e+08, 7.29158400e+08, 7.29244800e+08, 7.29331200e+08, 7.29417600e+08]),
                         'volume': numpy.array([1003200., 480500.0, 201300.0, 529400.0, 531500.0, 492100.0, 596100.0, 122100.0, 379600.0, 19500.0])}

        data = statstock.data.Yahoo(self.data_dir + "/sample.csv", "smp")

        for key in ["keys"] + list(expected_data.keys()):
            with self.subTest(data_field=key):
                if key == "keys":
                    self.assertEqual(data.data.keys(), expected_data.keys())
                else:
                    self.assertTrue(numpy.allclose(data.data[key], expected_data[key]))
