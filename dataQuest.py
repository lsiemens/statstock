import time
import json
import keyring
import requests
import datetime
import zoneinfo


class AuthError(Exception):
    """Authentication error"""


class RateLimit:
    def __init__(self, calls, interval=60):
        self.calls = calls
        self.interval = interval
        self._start = time.time()

        self.call_chain = [-1*self.interval]*self.calls
        self.index = 0
        print(self)

    def call(self):
        current_time = time.time() - self._start
        buffer = current_time - self.call_chain[self.index] - self.interval
        if buffer <= 0:
            time.sleep(abs(buffer))

        self.call_chain[self.index] = current_time
        self.index = (self.index + 1) % self.calls
        print(self)

    def __str__(self):
        string = "["
        for i in range(self.calls):
            if i != 0:
                string += ", "

            index = (self.index - i - 1) % self.calls
            string += f"{self.call_chain[index]:.3f}"
        string += "]"
        return string


class QuestradeClient:
    def __init__(self):
        self.login_base = "https://login.questrade.com"

        self.access_token = None
        self.refresh_token = None
        self.api_server = None
        self.expires_at = 0.0

        self._key_name = "questrade-token"
        self._username = "tokens"

        self._ratelimiter = RateLimit(20, 1)

        self._timezone = zoneinfo.ZoneInfo("America/New_York")

        self._load_tokens_from_keyring()

    def get_time(self, weeks=0, days=0, hours=0, minutes=0):
        time_offset = datetime.timedelta(weeks=weeks, days=days,
                                         hours=hours, minutes=minutes)

        time = datetime.datetime.now(self._timezone) + time_offset

        time_string = time.strftime("%Y-%m-%dT%H:%M:%S.%f%z")
        time_string = time_string[:-2] + ":" + time_string[-2:]
        return time_string

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

        payload = {"access_token": self.access_token,
                   "refresh_token": self.refresh_token,
                   "api_server": self.api_server,
                   "expires_as": self.expires_at}
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
        data = {"grant_type": "refresh_token",
                "refresh_token": self.refresh_token}

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
        self._ratelimiter.call()
        self._validate_tokens()

        if self.api_server is None:
            raise AuthError("No api server URL found!")

        url = f"{self.api_server}{path}"

        header = {"Authorization": f"Bearer {self.access_token}",
                  "Accept": "application/json"}

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
        symbols = self.request("GET",
                               "/v1/symbols/search",
                               {"prefix": ticker.strip()})
        for symbol in symbols["symbols"]:
            if symbol["symbol"] == ticker.upper():
                return symbol["symbolId"]
        raise ValueError(f"Failed to get a symbolId for the ticker: {ticker}")

    def get_quote(self, symbol_id):
        try:
            symbol_id = int(symbol_id)
        except ValueError:
            symbol_id = self.find_symbol_id(symbol_id)

        quote = self.request("GET", f"/v1/markets/quotes/{symbol_id}")
        quote = quote["quotes"][0]
        price = quote["lastTradePriceTrHrs"]
        return price

    def get_candles(self, symbol_id, startTime, endTime, interval):
        try:
            symbol_id = int(symbol_id)
        except ValueError:
            symbol_id = self.find_symbol_id(symbol_id)

        params = {"startTime": startTime,
                  "endTime": endTime,
                  "interval": interval}
        candles = self.request("GET", f"/v1/markets/candles/{symbol_id}", params)
        return candles["candles"]

    def get_n_candles(self, symbol_id, n, interval):
        endTime = self.get_time()
        startTime = None

        match interval:
            case "OneDay":
                days = n%5 + 7*(n//5)
                startTime = self.get_time(days=-days)
            case "OneWeek":
                startTime = self.get_time(weeks=-n)
            case _:
                raise NotImplemented(f"The interval {interval} has not been inplemented")
        return self.get_candles(symbol_id, startTime, endTime, interval)

if __name__ == "__main__":
    #import logging
    #import http
    #http.client.HTTPConnection.debuglevel = 1

    client = QuestradeClient()
    #client.clear_tokens()
    #client._refresh_using_current_tokens()
    #client._validate_tokens()

    print(client.get_quote("TSLA"))
    print()
    data = client.get_n_candles("TSLA", 502, "OneWeek")
    print(f"number of candles {len(data)}")


