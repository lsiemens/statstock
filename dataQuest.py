import time
import json
import keyring
import requests
import datetime
import zoneinfo

import numpy as np


class AuthError(Exception):
    """Authentication error"""


class RateLimit:
    """Class for adding rate limits

    A cyclic list of call times is used to put the process to sleep if the time
    between the current call and previous n calls exceeds a specified interval.
    """

    def __init__(self, calls, interval=60):
        self.calls = calls
        self.interval = interval
        self._start = time.time()

        self.call_chain = [-1*self.interval]*self.calls
        self.index = 0

    def call(self):
        """Add a rate limit controlled call

        This function will sleep if the rate limit is exceeded.
        """

        current_time = time.time() - self._start
        buffer = current_time - self.call_chain[self.index] - self.interval
        if buffer <= 0:
            print(f"INFO: Call limit reached, this process will sleep for {abs(buffer)} seconds")
            time.sleep(abs(buffer))

        self.call_chain[self.index] = current_time
        self.index = (self.index + 1) % self.calls

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
    """Manage the questrade API
    """

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
        """Get datetime string

        A datetime string in the format required for the questrade API.
        """

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
        """Remove the current tokens

        The API tokens will be cleared and deleted from the system keyring
        """

        try:
            keyring.delete_password(self._key_name, self._username)
        except keyring.errors.PasswordDeleteError:
            print("Failed to clear keyring")
        self.access_token = None
        self.refresh_token = None
        self.api_server = None
        self.expires_at = 0.0

    def _bootstrap_refresh_token(self):
        """Get refresh token from user input
        """

        token = input("Paste Questrade refresh token: ").strip()
        if not token:
            raise AuthError("No refresh token provided!")
        return token

    def _refresh_using_current_tokens(self):
        """Request new OAuth tokens using the refresh token
        """

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
        """Refresh tokens if they are empty or have expired
        """

        now = time.time()
        if self.access_token and (now < (self.expires_at - 30)):
            return
        self._refresh_using_current_tokens()

    def request(self, method, path, params=None, json_body=None):
        """Standard questraid info request

        Setup the format of the standard API requests
        """

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

    def find_symbol(self, ticker):
        """Get symbol from ticker

        The symbol contains the Questrade symbolID, ticker and currency in the
        format (Int, String, String).
        """

        symbols = self.request("GET",
                               "/v1/symbols/search",
                               {"prefix": ticker.strip()})
        for symbol in symbols["symbols"]:
            if symbol["symbol"] == ticker.upper():
                return (symbol["symbolId"], symbol["symbol"], symbol["currency"])
        raise ValueError(f"Failed to get a symbol for the ticker: {ticker}")

    def get_quote(self, symbol):
        """Get snap quote

        Get price and symbol (includes the currency) for a given symbol or
        raw stock ticker string.
        """

        try:
            symbolID = int(symbol[0])
        except ValueError:
            symbol = self.find_symbol(symbol)
            return self.get_quote(symbol)

        quote = self.request("GET", f"/v1/markets/quotes/{symbolID}")
        quote = quote["quotes"][0]
        price = quote["lastTradePriceTrHrs"]
        return (price, symbol)

    def read_candle_dict(self, candle):
        """Convert data into a standard format

        Store candle info from a dictionary in a numpy array with elements
        arranged as VWAP OHLC Volume.
        """

        data = np.array([candle["VWAP"], candle["open"], candle["high"],
                         candle["low"], candle["close"], candle["volume"]])
        return data

    def get_candles(self, symbol, startTime, endTime, interval):
        """Get candles from questrade

        Create a (n, 6) numpy array of candle data for the intervals between
        the start and end time. Missing data (weekends, ...) between those
        dates will be filled with float("Nan"). If the start or end date lands
        on a non-trading day this can cause the array to be smaller than
        expected.

        The candle data is packaged with interval and symbol in the form
        (candle, interval, symbol).
        """

        try:
            symbolID = int(symbol[0])
        except ValueError:
            symbol = self.find_symbol(symbol)
            return self.get_candles(symbol, startTime, endTime, interval)

        params = {"startTime": startTime,
                  "endTime": endTime,
                  "interval": interval}
        try:
            candles = self.request("GET", f"/v1/markets/candles/{symbolID}", params)
        except Exception:
            print(f"Exception while getting candles for {symbol[1]}")
            raise

        candles = candles["candles"]

        match interval:
            case "OneDay":
                step = 1
            case "OneWeek":
                step = 7
            case _:
                raise NotImplemented(f"The interval {interval} has not been inplemented")
        first_end = datetime.datetime.fromisoformat(candles[0]["end"])
        last_start = datetime.datetime.fromisoformat(candles[-1]["start"])
        n_elements = 2 + ((last_start - first_end).days//step)

        data = np.full((n_elements, 6), float("Nan"))
        data[0] = self.read_candle_dict(candles[0])
        for candle in candles[1:]:
            start = datetime.datetime.fromisoformat(candle["start"])
            i = 1 + ((start - first_end).days//step)
            data[i] = self.read_candle_dict(candle)
        return (data, interval, symbol)

    def get_n_candles(self, symbol, n, interval):
        """Get n candles from questrade starting at the current date.

        Create a (n, 6) numpy array of candle data.

        The candle data is packaged with interval and symbol in the form
        (candle, interval, symbol).
        """

        endTime = self.get_time()
        startTime = None

        match interval:
            case "OneDay":
                startTime = self.get_time(days=1 - n)
            case "OneWeek":
                startTime = self.get_time(weeks=1 - n)
            case _:
                raise NotImplemented(f"The interval {interval} has not been inplemented")
        candles, _, symbol = self.get_candles(symbol, startTime, endTime, interval)

        if (n == len(candles)):
            return (candles, interval, symbol)

        print(f"Warring: {n} data points requested from {symbol[1]}, {len(candles)} points where received.")
        if (abs(n - len(candles)) <= 4):
            print("This may be due to non trading days on the set boundary.")

        m = n - len(candles)

        extended_candles = np.full(shape=(n, 6), fill_value=float("Nan"))
        extended_candles[m:, :] = candles
        return (extended_candles, interval, symbol)

    def options(self, symbol, strike, width):
        """

        Parameters
        ----------
        symbol : tuple
        strike : float
            Strike price
        width : float
            Range around the strike price for the data

        Returns
        -------
        tuple : (int, int, float, float, float, int, int, string)
            The tuple contains the following
            - days to expiry
            - days from last trade
            - strike price
            - price high
            - price low
            - trade volume
            - open interest
            - option type "call" or "put"
        """

        try:
            symbolID = int(symbol[0])
        except ValueError:
            symbol = self.find_symbol(symbol)
            return self.options(symbol, strike, width)

        self._ratelimiter.call()
        result = self.request("GET", f"/v1/symbols/{symbolID}/options")
        option_chains = result["optionChain"]
        date = datetime.datetime.fromisoformat(self.get_time())

        multiplier = None
        option_type = None
        meta_data = [] # [(days_to_expiry, strike, callSymbolID, putSymbolID)]
        for i in range(len(option_chains)):
            option_chain = option_chains[i]

            expiry_date = datetime.datetime.fromisoformat(option_chain["expiryDate"])
            days_to_expiry = (expiry_date - date).days
            if option_type is None:
                option_type = option_chain["optionExerciseType"]

            assert len(option_chain["chainPerRoot"]) == 1, "Expect only one root"
            chain_root = option_chain["chainPerRoot"][0]
            if multiplier is None:
                multiplier = chain_root["multiplier"]
            chain_strikes = chain_root["chainPerStrikePrice"]

            for chain_strike in chain_strikes:
                strike_price = chain_strike["strikePrice"]
                callSymbolID = chain_strike["callSymbolId"]
                putSymbolID = chain_strike["putSymbolId"]
                if (strike_price >= strike - width/2) and (strike_price <= strike + width/2):
                    meta_data.append((days_to_expiry, strike_price, callSymbolID, putSymbolID))

        data = []

        chunk = meta_data[:50]
        meta_data = meta_data[50:]
        while(len(chunk) > 0):
            data += self._option_data(chunk)
            chunk = meta_data[:50]
            meta_data = meta_data[50:]
        return data

    def _option_data(self, meta_data):
        self._ratelimiter.call()
        date = datetime.datetime.fromisoformat(self.get_time())
        IDs = np.empty(2*len(meta_data), dtype=int)
        for i in range(len(meta_data)):
            IDs[2*i] = meta_data[i][2]
            IDs[2*i + 1] = meta_data[i][3]

        payload = {"optionIds": IDs.tolist()}
        result = self.request("POST", f"/v1/markets/quotes/options", json_body=payload)
        optionQuotes = result["optionQuotes"]

        data = [] # (days_to_expiry, days_from_trade, strike, high, low, volume, open_interest, option_type)
        for i, quote in enumerate(optionQuotes):
            symbolID = quote["symbolId"]
            if IDs[i] != symbolID:
                raise RuntimeError("Questrade symbolID missmatch")
            lastTradeTime = datetime.datetime.fromisoformat(quote["lastTradeTime"])
            lastTradeDay = (date - lastTradeTime).days
            volume = quote["volume"]
            highPrice = quote["highPrice"]
            lowPrice = quote["lowPrice"]
            openInterest = quote["openInterest"]

            option_type = None
            if i % 2 == 0:
                option_type = "call"
            else:
                option_type = "put"

            data.append((meta_data[i//2][0], lastTradeDay, meta_data[i//2][1],
                         highPrice, lowPrice, volume, openInterest, option_type))
        return data

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    client = QuestradeClient()
#    client.clear_tokens()

    ticker = "GLW"
    n = 2000
    interval = "OneWeek"

    print(f"{ticker} price:", client.get_quote(ticker))
    candles, _, (_, _, currency) = client.get_n_candles(ticker, n, interval)
    print("candle data shape", candles.shape)
    plt.plot(candles[:, 0])
    plt.xlabel(interval[3:])
    plt.ylabel(f"VWAP ({currency})")
    plt.title(ticker)
    plt.show()

    price, _ = client.get_quote(ticker)
    options = client.options(ticker, price, price/2)

    strikes1 = [strike for (_, _, strike, high, low, _, _, otype) in options if otype == "call"]
    strikes2 = [strike for (_, _, strike, high, low, _, _, otype) in options if otype == "put"]
    data1 = [(high + low)/2 for (_, _, strike, high, low, _, _, otype) in options if otype == "call"]
    data2 = [(high + low)/2 for (_, _, strike, high, low, _, _, otype) in options if otype == "put"]
    plt.scatter(strikes1, data1)
    plt.scatter(strikes2, data2)
    plt.show()
