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

errors, failures, unexpectedSuccesses = result.errors, result.failures, result.unexpectedSuccesses

if (len(errors) != 0) or (len(failures) != 0) or (len(unexpectedSuccesses)):
    raise RuntimeError("One or more unit tests failed, produced an unexpected exception, or resulted in an unexpected succsess.")
