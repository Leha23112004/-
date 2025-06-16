import json
import time
import sqlite3
import logging
from datetime import datetime, timedelta

import ccxt
import pandas as pd
import pandas_ta as ta
import feedparser
from transformers import pipeline
import yfinance as yf
from telegram import Bot
import schedule

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s:%(message)s")

CONFIG_PATH = "config.json"
DB_PATH = "trades.db"


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def init_db(path: str):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute(
        """CREATE TABLE IF NOT EXISTS trades(
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT,
                pair TEXT,
                side TEXT,
                price REAL,
                amount REAL,
                profit REAL,
                status TEXT
        )"""
    )
    conn.commit()
    return conn


def fetch_candles(exchange, pair, timeframe, limit=500):
    ohlcv = exchange.fetch_ohlcv(pair, timeframe=timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df.set_index("timestamp", inplace=True)
    df["EMA20"] = ta.ema(df["close"], length=20)
    df["EMA50"] = ta.ema(df["close"], length=50)
    df["RSI"] = ta.rsi(df["close"], length=14)
    macd = ta.macd(df["close"], fast=12, slow=26, signal=9)
    df = pd.concat([df, macd], axis=1)
    df.reset_index(inplace=True)
    return df


def fetch_sentiment():
    feeds = [
        "https://news.google.com/rss/search?q=bitcoin&hl=en&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=cryptocurrency&hl=en&gl=US&ceid=US:en",
        "https://news.google.com/rss/search?q=stock%20market&hl=en&gl=US&ceid=US:en",
    ]
    entries = []
    for url in feeds:
        d = feedparser.parse(url)
        entries.extend([e.get("title", "") + " " + e.get("summary", "") for e in d.entries])
    if not entries:
        return 0.5
    analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
    scores = []
    for text in entries:
        try:
            res = analyzer(text[:512])[0]
            score = res["score"] if res["label"] == "POSITIVE" else 1 - res["score"]
            scores.append(score)
        except Exception as e:
            logging.error(f"Sentiment error: {e}")
    return sum(scores) / len(scores) if scores else 0.5


def fetch_vix():
    try:
        vix = yf.download("^VIX", period="1d", interval="1d")
        if not vix.empty:
            return float(vix["Close"].iloc[-1])
    except Exception as e:
        logging.error(f"VIX fetch error: {e}")
    return 0.0


def generate_signal(latest, sentiment, vix):
    ema20 = latest["EMA20"]
    rsi = latest["RSI"]
    price = latest["close"]
    vix_high = vix > 25
    if price > ema20 and rsi < 30 and sentiment >= 0.7 and not vix_high:
        return "BUY"
    if price < ema20 and rsi > 70 and (sentiment <= 0.3 or vix_high):
        return "SELL"
    return "HOLD"


class TradingBot:
    def __init__(self, config: dict):
        self.config = config
        self.exchange = ccxt.binance({
            "apiKey": config["exchange"]["api_key"],
            "secret": config["exchange"]["api_secret"],
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })
        if config["exchange"].get("testnet"):
            self.exchange.set_sandbox_mode(True)
        self.bot = Bot(token=config["telegram"]["token"])
        self.conn = init_db(DB_PATH)
        self.open_trades = []
        self.last_notification = datetime.utcnow() - timedelta(seconds=config["telegram"]["notification_interval"])

    def notify(self, message: str):
        try:
            self.bot.send_message(chat_id=self.config["telegram"]["chat_id"], text=message)
        except Exception as e:
            logging.error(f"Telegram error: {e}")

    def fetch_balance(self):
        try:
            bal = self.exchange.fetch_balance()
            return bal["total"].get(self.config["exchange"]["stake_currency"], 0)
        except Exception as e:
            logging.error(f"Balance fetch error: {e}")
            return 0

    def record_trade(self, pair, side, price, amount, profit, status):
        c = self.conn.cursor()
        c.execute(
            "INSERT INTO trades(timestamp, pair, side, price, amount, profit, status) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (datetime.utcnow().isoformat(), pair, side, price, amount, profit, status),
        )
        self.conn.commit()

    def update_trade(self, trade_id, profit, status):
        c = self.conn.cursor()
        c.execute("UPDATE trades SET profit=?, status=? WHERE id=?", (profit, status, trade_id))
        self.conn.commit()

    def get_recent_pnl(self, minutes=10):
        since = datetime.utcnow() - timedelta(minutes=minutes)
        c = self.conn.cursor()
        c.execute(
            "SELECT SUM(profit) FROM trades WHERE status='closed' AND timestamp > ?",
            (since.isoformat(),),
        )
        row = c.fetchone()
        return row[0] or 0.0

    def manage_trades(self, pair, price):
        to_close = []
        for trade in self.open_trades:
            side = trade["side"]
            open_price = trade["price"]
            take_profit = open_price * (1 + self.config["risk_management"]["take_profit"] * (1 if side == "long" else -1))
            stop_loss = open_price * (1 - self.config["risk_management"]["stop_loss"] * (1 if side == "long" else -1))
            if side == "long":
                if price >= take_profit or price <= stop_loss:
                    profit = (price - open_price) * trade["amount"]
                    to_close.append((trade, profit))
            else:
                if price <= take_profit or price >= stop_loss:
                    profit = (open_price - price) * trade["amount"]
                    to_close.append((trade, profit))
        for trade, profit in to_close:
            self.open_trades.remove(trade)
            self.update_trade(trade["id"], profit, "closed")
            self.notify(f"Закрыта позиция {trade['side']} {pair} по цене {price}. Прибыль: {profit:.2f}")

    def open_trade(self, pair, side, price):
        if len(self.open_trades) >= self.config["exchange"]["max_open_trades"]:
            return
        amount = self.config["exchange"]["stake_amount"] / price
        if self.config["exchange"].get("dry_run"):
            order_id = None
        else:
            params = {"type": "market", "side": side, "amount": amount}
            order = self.exchange.create_order(pair, "market", side, amount)
            order_id = order["id"]
        trade = {"id": None, "side": "long" if side == "buy" else "short", "price": price, "amount": amount}
        self.record_trade(pair, side, price, amount, 0.0, "open")
        trade["id"] = self.conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        self.open_trades.append(trade)
        self.notify(f"Открыта позиция {side} {pair} по цене {price}")

    def process(self):
        sentiment = fetch_sentiment()
        vix = fetch_vix()
        for pair in self.config["exchange"]["trading_pairs"]:
            tf = self.config["exchange"]["timeframes"][0]
            df = fetch_candles(self.exchange, pair, tf)
            df = compute_indicators(df)
            latest = df.iloc[-1]
            signal = generate_signal(latest, sentiment, vix)
            price = latest["close"]
            self.manage_trades(pair, price)
            if signal == "BUY":
                self.open_trade(pair, "buy", price)
            elif signal == "SELL":
                self.open_trade(pair, "sell", price)
        now = datetime.utcnow()
        if (now - self.last_notification).total_seconds() >= self.config["telegram"]["notification_interval"]:
            balance = self.fetch_balance()
            pnl = self.get_recent_pnl()
            self.notify(f"Баланс {balance} {self.config['exchange']['stake_currency']}. PnL(10m): {pnl:.2f}")
            self.last_notification = now


def main():
    config = load_config(CONFIG_PATH)
    bot = TradingBot(config)
    schedule.every().minute.do(bot.process)
    while True:
        schedule.run_pending()
        time.sleep(1)


if __name__ == "__main__":
    main()
