""" 
Preform statistical analysis using markov chains
"""

import numpy

class Markov:
    """ 
    Compute and analyse markov chains
    """

    def __init__(self, data, symbols, order=3):
        """ 
        
        Parameters
        ----------
        data : array
            data must be an array of integers in the range
            [0, len(symbols) - 1].
        """
        self.raw_data = data
        self.symbols = numpy.array(symbols)
        self.order = order
        
        self.data = {}
        
        if not numpy.issubdtype(numpy.array(self.raw_data).dtype, numpy.integer):
            raise ValueError("data must be an integer ndarray")
        
        self._compute_chain()
        
    def get_data(self, key):
        """ 
        Get data from markov chain
        
        Parameters
        ----------
        key : array
            key must be an array of integers in the range
            [0, len(symbols) - 1] or the built-in constant Ellipsis. Note,
            the symbol Ellipsis must only appear as the last element in 
            the key.
        """
        
        if Ellipsis in key:
            if key[-1] is not Ellipsis:
                raise ValueError("An Ellipsis may only appear as the last ellement in key.")
            else:
                if len(key[:-1]) != 0:
                    if not numpy.issubdtype(numpy.array(key[:-1]).dtype, numpy.integer):
                        raise ValueError("list must be an integer ndarray")
        else:
            if not numpy.issubdtype(numpy.array(key).dtype, numpy.integer):
                raise ValueError("list must be an integer ndarray")

        if Ellipsis in key:
            result = numpy.zeros((len(self.symbols),))
            for i in range(len(self.symbols)):
                temp_key = list(key[:-1]) + [i]
                if self._key(temp_key) in self.data:
                    result[i] = self.data[self._key(temp_key)]
            return result
        else:
            if self._key(key) in self.data:
                return self.data[self._key(key)]
            else:
                return 0
            
    def _key(self, list):
        if not numpy.issubdtype(numpy.array(list).dtype, numpy.integer):
            raise ValueError("list must be an integer ndarray")

        return ",".join([str(element) for element in list])

    def _compute_chain(self):
        """ 
        Compute markov chain transitions
        """
        for i in range(len(self.raw_data) - self.order + 1):
            ngram = self.raw_data[i:i + self.order]
            for j in range(self.order):
                jgram = ngram[j:]
                
                if self._key(jgram) not in self.data:
                    self.data[self._key(jgram)] = 1
                else:
                    self.data[self._key(jgram)] += 1
