import requests
import pandas as pd
import time
import logging
from pathlib import Path


class NSEHistoricalIndex:

    URL = "https://www.nseindia.com/api/historicalOR/indicesHistory"

    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/154.0.0.0 Safari/537.36"
        ),
        "Accept": "application/json",
        "Referer": (
            "https://www.nseindia.com/"
            "reports-indices-historical-index-data"
        ),
    }

    # Keep comfortably below NSE's 70-row limit
    MAX_DAYS_PER_REQUEST = 60

    def __init__(self, index_type="NIFTY 50"):

        self.index_type = index_type

        self.session = requests.Session()
        self.session.headers.update(self.HEADERS)

        self._initialize_session()

    def _initialize_session(self):

        response = self.session.get("https://www.nseindia.com",timeout=20)
        response.raise_for_status()

    def _request(self, start_date, end_date):
        params = {
            "indexType": self.index_type,
            "from": start_date.strftime("%d-%m-%Y"),
            "to": end_date.strftime("%d-%m-%Y"),
        }

        response = self.session.get(
            self.URL,
            params=params,
            timeout=30
        )

        response.raise_for_status()
        result = response.json()
        if "data" not in result:
            raise RuntimeError(
                f"Unexpected NSE response: {result}"
            )

        return result["data"]

    def download(self, start_date, end_date):

        start_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date)

        if start_date > end_date:
            raise ValueError(
                "start_date must be before end_date"
            )

        all_data = []
        current_start = start_date
        while current_start <= end_date:
            current_end = min(
                current_start
                + pd.Timedelta(days=self.MAX_DAYS_PER_REQUEST - 1),
                end_date
            )

            logging.info(
                "Downloading %s → %s",
                current_start.date(),
                current_end.date()
            )

            data = self._request(
                current_start,
                current_end
            )

            logging.info(
                "Received %d records",
                len(data)
            )

            all_data.extend(data)

            current_start = (
                current_end
                + pd.Timedelta(days=1)
            )

            # Avoid hammering NSE
            time.sleep(0.5)

        return self._clean(all_data)

    @staticmethod
    def _clean(data):

        df = pd.DataFrame(data)

        df = df[
            [
                "EOD_TIMESTAMP",
                "EOD_OPEN_INDEX_VAL",
                "EOD_HIGH_INDEX_VAL",
                "EOD_LOW_INDEX_VAL",
                "EOD_CLOSE_INDEX_VAL",
            ]
        ].copy()

        df.columns = [
            "date",
            "open",
            "high",
            "low",
            "close",
        ]

        df["date"] = pd.to_datetime(
            df["date"],
            format="%d-%b-%Y",
            errors="coerce"
        )

        for col in ["open", "high", "low", "close"]:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            )

        df = (
            df
            .dropna(subset=["date", "close"])
            .drop_duplicates(subset=["date"])
            .sort_values("date")
            .reset_index(drop=True)
        )

        return df

    @staticmethod
    def save(df, filename):
        path = Path(filename)
        path.parent.mkdir(parents=True,exist_ok=True)
        df.to_csv(path,index=False)

def main():

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s"
    )

    nse = NSEHistoricalIndex("NIFTY 50")
    
    nifty = nse.download(
        start_date="2025-01-01",
        end_date="2026-12-31"
    )

    nse.save(nifty, "data/nifty_close.csv")

    print(nifty)
    print("\nShape:", nifty.shape)


if __name__ == "__main__":
    main()