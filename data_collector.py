import ccxt
import pandas as pd
import logging
from datetime import datetime, timedelta
import json
import numpy as np

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('data_collector.log', mode='w'),
        logging.StreamHandler()
    ]
)

# Загрузка конфигурации
with open('config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

BINANCE_API_KEY = config['binance_api_key']
BINANCE_API_SECRET = config['binance_api_secret']

class DataCollector:
    def __init__(self, test_mode=True):
        """
        Инициализация сборщика данных
        
        Args:
            test_mode (bool): Режим тестирования без подключения к бирже
        """
        self.logger = logging.getLogger(__name__)
        self.test_mode = test_mode
        self.symbols = []
        self.historical_data = {}
        
        if not test_mode:
            try:
                # Инициализация подключения к бирже
                self.exchange = ccxt.binance({
                    'enableRateLimit': True,
                    'options': {
                        'defaultType': 'spot'
                    }
                })
                
                # Получение списка доступных торговых пар
                markets = self.exchange.load_markets()
                self.symbols = [symbol for symbol in markets.keys() if symbol.endswith('/USDT')]
                self.logger.info(f"DataCollector инициализирован. Доступно {len(self.symbols)} торговых пар")
                
            except Exception as e:
                self.logger.error(f"Ошибка при инициализации DataCollector: {str(e)}")
                raise
        else:
            # Тестовый режим
            self.symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT']
            self.logger.info(f"DataCollector инициализирован в тестовом режиме. Доступно {len(self.symbols)} торговых пар")
            
    def get_historical_data(self, symbol, timeframe='1h', limit=200):
        """
        Получение исторических данных
        
        Args:
            symbol (str): Торговая пара
            timeframe (str): Временной интервал
            limit (int): Количество свечей
            
        Returns:
            pd.DataFrame: Исторические данные
        """
        try:
            if not self.test_mode:
                # Получение данных с биржи
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df.set_index('timestamp', inplace=True)
            else:
                # Генерация тестовых данных
                dates = pd.date_range(end=pd.Timestamp.now(), periods=limit, freq='1H')
                np.random.seed(42)  # Для воспроизводимости
                
                # Базовые цены для разных пар
                base_prices = {
                    'BTC/USDT': 100000,
                    'ETH/USDT': 2500,
                    'BNB/USDT': 500
                }
                
                base_price = base_prices.get(symbol, 100)
                prices = np.random.normal(base_price, base_price * 0.02, limit)
                volumes = np.random.normal(1000, 200, limit)
                
                df = pd.DataFrame({
                    'open': prices,
                    'high': prices * (1 + np.random.uniform(0, 0.02, limit)),
                    'low': prices * (1 - np.random.uniform(0, 0.02, limit)),
                    'close': prices * (1 + np.random.normal(0, 0.01, limit)),
                    'volume': volumes
                }, index=dates)
            
            self.historical_data[symbol] = df
            self.logger.info(f"Получено {len(df)} записей для {symbol}")
            return df
            
        except Exception as e:
            self.logger.error(f"Ошибка при получении исторических данных для {symbol}: {str(e)}")
            return None
            
    def get_current_price(self, symbol):
        """
        Получение текущей цены
        
        Args:
            symbol (str): Торговая пара
            
        Returns:
            float: Текущая цена
        """
        try:
            if not self.test_mode:
                ticker = self.exchange.fetch_ticker(symbol)
                return ticker['last']
            else:
                # В тестовом режиме возвращаем последнюю цену из исторических данных
                if symbol in self.historical_data:
                    return self.historical_data[symbol]['close'].iloc[-1]
                return None
                
        except Exception as e:
            self.logger.error(f"Ошибка при получении текущей цены для {symbol}: {str(e)}")
            return None

    def get_all_prices(self):
        """
        Получение текущих цен всех пар
        
        :return: Словарь с ценами {symbol: price}
        """
        try:
            logging.info("Получение цен всех пар...")
            prices = {}
            
            # Получаем все тикеры одним запросом
            tickers = self.exchange.fetch_tickers()
            
            # Фильтруем только пары с USDT
            for symbol, ticker in tickers.items():
                if symbol.endswith('/USDT'):
                    try:
                        if ticker['last'] is not None:
                            prices[symbol] = ticker['last']
                    except Exception as e:
                        logging.warning(f"Ошибка при обработке тикера {symbol}: {str(e)}")
                        continue
                    
            logging.info(f"Получено {len(prices)} цен")
            return prices
            
        except Exception as e:
            logging.error(f"Ошибка при получении цен: {str(e)}")
            return {}

    def get_balance_in_usdt(self):
        """
        Получение баланса с конвертацией в USDT для удобства просмотра
        
        :return: Словарь с балансами {symbol: (amount, usdt_value)}
        """
        try:
            logging.info("Получение баланса...")
            
            # Получение баланса
            balance = self.exchange.fetch_balance()
            
            # Получение текущих цен
            prices = self.get_all_prices()
            
            # Расчет баланса
            balance_info = {}
            total_usdt = 0
            
            for currency, amount in balance['total'].items():
                if amount > 0:
                    if currency == 'USDT':
                        balance_info[currency] = (amount, amount)
                        total_usdt += amount
                    else:
                        symbol = f"{currency}/USDT"
                        if symbol in prices:
                            usdt_value = amount * prices[symbol]
                            balance_info[currency] = (amount, usdt_value)
                            total_usdt += usdt_value
            
            balance_info['TOTAL'] = (None, total_usdt)
            
            logging.info(f"Баланс получен. Всего {len(balance_info)} активов")
            return balance_info
            
        except Exception as e:
            logging.error(f"Ошибка при получении баланса: {str(e)}")
            return {}

    def get_max_historical_data(self, symbol, timeframe='1h', max_records=5000):
        """
        Получить максимум исторических данных (до max_records) для указанной пары и таймфрейма
        """
        try:
            all_ohlcv = []
            since = None
            limit = 1000  # максимум за 1 запрос
            while len(all_ohlcv) < max_records:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
                if not ohlcv or len(ohlcv) == 0:
                    break
                all_ohlcv += ohlcv
                if len(ohlcv) < limit:
                    break  # больше данных нет
                since = ohlcv[-1][0] + 1  # следующий запрос с конца предыдущего
            # Обрезаем до max_records
            all_ohlcv = all_ohlcv[-max_records:]
            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            logging.error(f"Ошибка при получении максимального объема исторических данных для {symbol}: {str(e)}")
            return pd.DataFrame()

    def get_available_historical_data(self, symbol, timeframe='1h', max_records=5000):
        """
        Получить максимально доступное количество исторических данных для указанной пары и таймфрейма.
        Если доступно меньше max_records, то вернёт все доступные данные.
        """
        try:
            all_ohlcv = []
            since = None
            limit = 1000  # максимум за 1 запрос
            while len(all_ohlcv) < max_records:
                ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
                if not ohlcv or len(ohlcv) == 0:
                    break
                all_ohlcv += ohlcv
                if len(ohlcv) < limit:
                    break  # больше данных нет
                since = ohlcv[-1][0] + 1  # следующий запрос с конца предыдущего
            # Обрезаем до max_records
            all_ohlcv = all_ohlcv[-max_records:]
            df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            return df
        except Exception as e:
            logging.error(f"Ошибка при получении доступных исторических данных для {symbol}: {str(e)}")
            return pd.DataFrame()

def test_data_collector():
    """Тестирование DataCollector"""
    # Тестовые ключи
    api_key = 'cOM5irfeAfk4P4jZlPZtEoJizn6xwYBPJ5wHUhT4iP7BqgvZmWUZ6SyG3nkWHptx'
    api_secret = '1EobjQHxtyDkMPQ8F6K6ghmIUOjAyy3T2LCYyl9B6twqSh4G16MBURRAmteEIzzK'
    
    try:
        logging.info("Начало тестирования DataCollector")
        collector = DataCollector(api_key, api_secret)
        
        # Тест 1: Получение исторических данных
        logging.info("\nТест 1: Исторические данные")
        df = collector.get_historical_data('BTC/USDT', limit=5)
        if df is not None:
            print(df)
        
        # Тест 2: Получение текущих цен
        logging.info("\nТест 2: Текущие цены")
        prices = collector.get_all_prices()
        print(f"Получено {len(prices)} цен")
        print("Примеры цен:")
        for symbol, price in list(prices.items())[:5]:
            print(f"{symbol}: {price}")
        
        # Тест 3: Получение баланса
        logging.info("\nТест 3: Баланс")
        balance = collector.get_balance_in_usdt()
        print("Баланс:")
        for currency, (amount, usdt_value) in balance.items():
            if currency == 'TOTAL':
                print(f"Общий баланс: {usdt_value:,.2f} USDT")
            else:
                print(f"{currency}: {amount:,.8f} (≈ {usdt_value:,.2f} USDT)")
        
    except Exception as e:
        logging.error(f"Ошибка при тестировании: {str(e)}")
        raise

if __name__ == "__main__":
    test_data_collector() 