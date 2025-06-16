import pandas as pd
import numpy as np
import pandas_ta as ta
import logging
from datetime import datetime
from data_collector import DataCollector
import json

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trend_analyzer.log'),
        logging.StreamHandler()
    ]
)

# Загрузка конфигурации
with open('config.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

BINANCE_API_KEY = config['binance_api_key']
BINANCE_API_SECRET = config['binance_api_secret']

class TrendAnalyzer:
    def __init__(self, api_key=None, api_secret=None):
        self.data_collector = DataCollector(api_key, api_secret)
        self.logger = logging.getLogger(__name__)
        
    def analyze_trend(self, symbol, timeframe='1h', lookback=200):
        """
        Анализ тренда для указанной пары
        """
        try:
            # Получаем максимально доступные исторические данные
            df = self.data_collector.get_available_historical_data(symbol, timeframe, lookback)
            if df.empty:
                self.logger.error(f"Не удалось получить данные для {symbol}")
                return None
                
            # Выводим информацию о количестве данных
            self.logger.info(f"Получено {len(df)} свечей для {symbol}")
            self.logger.info(f"Период данных: с {df['timestamp'].iloc[0]} по {df['timestamp'].iloc[-1]}")
            
            # Проверяем качество данных
            if len(df) < 20:  # Минимум для расчета индикаторов
                self.logger.warning(f"Слишком мало данных для {symbol}: {len(df)} свечей")
                return None
                
            # Рассчитываем индикаторы тренда
            # 1. Скользящие средние (уменьшенные периоды)
            df['SMA_10'] = ta.sma(df['close'], length=10)
            df['SMA_20'] = ta.sma(df['close'], length=20)
            
            # 2. EMA (уменьшенные периоды)
            df['EMA_10'] = ta.ema(df['close'], length=10)
            df['EMA_20'] = ta.ema(df['close'], length=20)
            
            # 3. ADX (уменьшенный период)
            adx = ta.adx(df['high'], df['low'], df['close'], length=10)
            df['ADX'] = adx['ADX_10']
            
            # 4. MACD (уменьшенные периоды)
            macd = ta.macd(df['close'], fast=8, slow=17, signal=9)
            df['MACD'] = macd['MACD_8_17_9']
            df['MACD_SIGNAL'] = macd['MACDs_8_17_9']
            df['MACD_HIST'] = macd['MACDh_8_17_9']
            
            # 5. RSI (дополнительный индикатор)
            df['RSI'] = ta.rsi(df['close'], length=14)
            
            # Анализ текущего состояния
            current_price = df['close'].iloc[-1]
            current_adx = df['ADX'].iloc[-1]
            current_macd = df['MACD'].iloc[-1]
            current_signal = df['MACD_SIGNAL'].iloc[-1]
            current_rsi = df['RSI'].iloc[-1]
            
            # Проверки на None/NaN
            if any(pd.isna(x) for x in [current_adx, current_macd, current_signal, current_rsi]):
                self.logger.warning(f"Недостаточно данных для анализа {symbol} (NaN в индикаторах)")
                return None
            
            # Определение силы тренда
            trend_strength = "Сильный" if current_adx > 20 else "Слабый"
            
            # Определение направления тренда (по EMA_20)
            if current_price > df['EMA_20'].iloc[-1]:
                trend_direction = "Восходящий"
            elif current_price < df['EMA_20'].iloc[-1]:
                trend_direction = "Нисходящий"
            else:
                trend_direction = "Боковой"
                
            # Анализ MACD
            if current_macd > current_signal:
                macd_signal = "Бычий"
            else:
                macd_signal = "Медвежий"
                
            # Анализ RSI
            if current_rsi > 70:
                rsi_signal = "Перекуплен"
            elif current_rsi < 30:
                rsi_signal = "Перепродан"
            else:
                rsi_signal = "Нейтральный"
                
            # Формируем результат анализа
            analysis = {
                'symbol': symbol,
                'current_price': current_price,
                'trend_strength': trend_strength,
                'trend_direction': trend_direction,
                'adx_value': current_adx,
                'macd_signal': macd_signal,
                'rsi_signal': rsi_signal,
                'rsi_value': current_rsi,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            self.logger.info(f"Анализ тренда для {symbol} завершен")
            return analysis
            
        except Exception as e:
            self.logger.error(f"Ошибка при анализе тренда для {symbol}: {str(e)}")
            return None
            
    def analyze_all_trends(self, timeframe='1h', lookback=200):
        """
        Анализ трендов для всех доступных пар
        """
        try:
            all_pairs = self.data_collector.get_all_prices()
            results = []
            
            for symbol in all_pairs.keys():
                analysis = self.analyze_trend(symbol, timeframe, lookback)
                if analysis:
                    results.append(analysis)
                    
            return results
            
        except Exception as e:
            self.logger.error(f"Ошибка при анализе всех трендов: {str(e)}")
            return None

def test_trend_analyzer():
    """
    Тестирование анализатора трендов
    """
    analyzer = TrendAnalyzer(BINANCE_API_KEY, BINANCE_API_SECRET)
    
    # Тест анализа одной пары
    print("\nТест анализа BTC/USDT:")
    btc_analysis = analyzer.analyze_trend('BTC/USDT', '1h', 200)
    if btc_analysis:
        print(f"Текущая цена: {btc_analysis['current_price']}")
        print(f"Сила тренда: {btc_analysis['trend_strength']}")
        print(f"Направление тренда: {btc_analysis['trend_direction']}")
        print(f"Значение ADX: {btc_analysis['adx_value']:.2f}")
        print(f"Сигнал MACD: {btc_analysis['macd_signal']}")
        print(f"Сигнал RSI: {btc_analysis['rsi_signal']} ({btc_analysis['rsi_value']:.2f})")
    
    # Тест анализа всех пар
    print("\nТест анализа всех пар:")
    all_analysis = analyzer.analyze_all_trends('1h', 200)
    if all_analysis:
        print(f"Проанализировано пар: {len(all_analysis)}")
        # Выводим топ-5 пар с самым сильным трендом
        strong_trends = sorted(all_analysis, key=lambda x: x['adx_value'], reverse=True)[:5]
        print("\nТоп-5 пар с самым сильным трендом:")
        for pair in strong_trends:
            print(f"{pair['symbol']}: ADX = {pair['adx_value']:.2f}, Направление = {pair['trend_direction']}, RSI = {pair['rsi_value']:.2f}")

if __name__ == "__main__":
    test_trend_analyzer() 