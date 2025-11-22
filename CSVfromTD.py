#!/usr/bin/env python3

import sys

class holding:
  def __init__(self, text):
    self.ticker = None
    self.quantity = None
    self.price = None
    self.currency = None
    self.malformed = True

    self.from_text(text)

  def from_text(self, text):
    self.ticker = text[0]
    text = text[2:] 

    try:
      self.quantity = float(text[0])
    except ValueError:
      self.malformed = True
      return
    text = text[1:]    

    if text[0][0] != "$":
      self.malformed = True
      return
    else:
      try:
        self.price = float(text[0][1:])
      except:
        self.malformed = True
        return
    text = text[1:]

    if text[0][0] != "(":
      self.currency = "CAD"
      self.ticker += ".TO"
    else:
      self.currency = text[0][1:-1]
      text = text[1:]

    if self.currency not in ["CAD", "USD"]:
      self.malformed = True
      return

    self.malformed = False

  def __str__(self):
    if not self.malformed:
      return f"{self.ticker}, {self.quantity}, {self.price}, {self.currency}"
    else:
      return "Ticker, Quantity, Price, Currency"

  def __repr__(self):
      return f"Holding: malformed = {self.malformed}, ticker = {self.ticker}, quantity =     {self.quantity}, price = {self.price}, currency = {self.currency}"

def main():
  block = []
  for line in sys.stdin:
    if (line.strip() != ""):
      block.append(line.strip())
    else:
      stock = holding(block)
      sys.stdout.write(f"{str(stock)}\n")
      block = []

if __name__ == "__main__":
  main()
