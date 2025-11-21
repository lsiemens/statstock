import os
import time
import json
import keyring
import requests

class AuthError(Exception):
  """Authentication error"""

class QuestradeClient:
  def __init__(self):
    self.login_base = "https://login.questrade.com"
    
    self.access_token = None
    self.refresh_token = None
    self.api_server = None
    self.expires_at = 0.0

    self._key_name = "questrade-token"
    self._username = "tokens"

    self._load_tokens_from_keyring()

  def _load_tokens_from_keyring(self):
    data = keyring.get_password(self._key_name, self._username)
    if not data:
      return
    
    try:
      payload = json.loads(data)
    except json.JSONDecodeError:
      return

    self.access_token = payload.get("access_token")
    self.refresh_token = payload.get("refresh_token")
    self.api_server = payload.get("api_server")
    self.expires_at = float(payload.get("expires_as", 0))

  def _save_tokens_to_keyring(self):
    if (self.access_token is None) or (self.refresh_token is None) or (self.api_server is None):
      raise AuthError("Failed to save tokens to keyring, missing token info!")

    payload = {"access_token":self.access_token,
               "refresh_token":self.refresh_token,
               "api_server":self.api_server,
               "expires_as":self.expires_at}
    keyring.set_password(self._key_name, self._username, json.dumps(payload))

  def clear_tokens(self):
    try:
      keyring.delete_password(self._key_name, self._username)
    except keyring.errors.PasswordDeleteError:
      print("Failed to clear keyring")
    self.access_token = None
    self.refresh_token = None
    self.api_server = None
    self.expires_at = 0.0

  def _bootstrap_refresh_token(self):
    token = input("Paste Questrade refresh token: ").strip()
    if not token:
      raise AuthError("No refresh token provided!")
    return token

  def _refresh_using_current_tokens(self):
    if self.refresh_token is None:
      self.refresh_token = self._bootstrap_refresh_token()
    
    token_url = f"{self.login_base}/oauth2/token"
    data = {"grant_type":"refresh_token",
            "refresh_token":self.refresh_token}
    
    try:
      responce = requests.post(token_url, data=data, timeout=10)
    except requests.RequestException as e:
      raise AuthError(f"Error accessing Questraid login portal: {e}")

    if responce.status_code != 200:
      snippet = responce.text[:200]
      raise AuthError(f"Questraid token refresh failed with status {responce.status_code}: {snippet}")

    payload = responce.json()
    self.access_token = payload["access_token"]
    self.refresh_token = payload["refresh_token"]
    self.api_server = payload["api_server"].rstrip("/")

    expires_in = int(payload.get("expires_in", 0))
    self.expires_at = time.time() + max(expires_in - 60, 0)

    self._save_tokens_to_keyring()

  def _validate_tokens(self):
    now = time.time()
    if self.access_token and (now < (self.expires_at - 30)):
      return
    self._refresh_using_current_tokens()

  def request(self, method, path, params=None, json_body=None):
    self._validate_tokens()
    
    if self.api_server is None:
      raise AuthError("No api server URL found!")

    url = f"{self.api_server}{path}"
    
    header = {"Authorization":f"Bearer {self.access_token}",
              "Accept":"application/json"}

    try:
      responce = requests.request(method=method.upper(),
                                  url=url,
                                  headers=header,
                                  params=params,
                                  json=json_body,
                                  timeout=10)
    except requests.RequestException as e:
      raise AuthError(f"HTTPS request failed: {e}")

    if responce.status_code == 401:
      self._refresh_using_current_tokens()
      return self.request(method, path, params, json_body)

    responce.raise_for_status()
    return responce.json()

  def find_symbol_id(self, ticker):
    symbols = self.request("GET", "/v1/symbols/search", {"prefix":ticker.strip()})
    print(symbols)

  def get_quote(self, symbol_id):
    quote = self.request("GET", f"/v1/markets/quotes/{symbol_id}")
    print(quote)
    
  def get_candels(self, symbol_id, startTime, endTime, interval):
    params = {"startTime":startTime, "endTime":endTime, "interval":interval}
    candels = self.request("GET", f"/v1/markets/candels/{symbol_id}", params)
    print(candels)

if __name__ == "__main__":
  #import logging
  #import http
  #http.client.HTTPConnection.debuglevel = 1

  client = QuestradeClient()
  #client.clear_tokens()
  #client._refresh_using_current_tokens()
  client._validate_tokens()
  client.find_symbol_id("IBIT")
  print()
  client.get_quote(52317225)
  print()
  client.get_candels(52317225, "2023-01-01T09:30:00-05:00", "2024-01-01T09:30:00-05:00", "OneMonth")
