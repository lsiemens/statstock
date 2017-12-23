""" 
Preform statistical analysis using markov chains
"""

import numpy

class Markov:
    def __init__(self, data, symbols, order=3):
        self.raw_data = list(data)
        self.symbols = numpy.array(symbols)
        self.order = order
        
        self.data = {}
        
        self._compute_chain()
        
        for i in range(len(self.symbols)):
            if self._key([i]) in self.data:
                print(str(self.data[self._key([i])]) + " " + str(self.symbols[i]))
            else:
                print(str(0) + " " + str(self.symbols[i]))
        
    def _key(self, list):
        return ",".join([str(element) for element in list])

    def _compute_chain(self):
        for i in range(len(self.raw_data) - self.order + 1):
            ngram = self.raw_data[i:i + self.order]
            for j in range(self.order):
                jgram = ngram[j:]
                
                if self._key(jgram) not in self.data:
                    self.data[self._key(jgram)] = 1
                else:
                    self.data[self._key(jgram)] += 1
