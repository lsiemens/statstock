# Testing suite
import unittest

from . import test_data

#setup unittest
loader = unittest.TestLoader()
suite = unittest.TestSuite()
runner = unittest.TextTestRunner(verbosity=3)

#load unit tests
suite.addTests(loader.loadTestsFromModule(test_data))

#run all tests
result = runner.run(suite)
