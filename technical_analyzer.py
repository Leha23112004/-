import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import talib
from data_collector import DataCollector

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('technical_analyzer.log', mode='w'),
        logging.StreamHandler()
    ]
)

class TechnicalAnalyzer:
    def __init__(self, data_collector):
        """
        Инициализация технического анализатора
        
        Args:
            data_collector (DataCollector): Объект для сбора данных
        """
        self.logger = logging.getLogger(__name__)
        self.data_collector = data_collector
        self.logger.info("TechnicalAnalyzer успешно инициализирован")

    def calculate_indicators(self, df):
        """
        Расчет технических индикаторов
        
        Args:
            df (pd.DataFrame): DataFrame с историческими данными
            
        Returns:
            pd.DataFrame: DataFrame с добавленными индикаторами
        """
        try:
            # Скользящие средние
            df['SMA_20'] = df['close'].rolling(window=20).mean()
            df['SMA_50'] = df['close'].rolling(window=50).mean()
            df['SMA_200'] = df['close'].rolling(window=200).mean()
            
            # EMA для более быстрой реакции
            df['EMA_12'] = df['close'].ewm(span=12, adjust=False).mean()
            df['EMA_26'] = df['close'].ewm(span=26, adjust=False).mean()
            
            # RSI
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            df['RSI'] = 100 - (100 / (1 + rs))
            
            # MACD
            df['MACD'] = df['EMA_12'] - df['EMA_26']
            df['Signal_Line'] = df['MACD'].ewm(span=9, adjust=False).mean()
            df['MACD_Hist'] = df['MACD'] - df['Signal_Line']
            
            # Bollinger Bands
            df['BB_middle'] = df['close'].rolling(window=20).mean()
            df['BB_std'] = df['close'].rolling(window=20).std()
            df['BB_upper'] = df['BB_middle'] + (df['BB_std'] * 2)
            df['BB_lower'] = df['BB_middle'] - (df['BB_std'] * 2)
            
            # Stochastic Oscillator
            low_14 = df['low'].rolling(window=14).min()
            high_14 = df['high'].rolling(window=14).max()
            df['%K'] = 100 * ((df['close'] - low_14) / (high_14 - low_14))
            df['%D'] = df['%K'].rolling(window=3).mean()
            
            # ATR для волатильности
            df['TR'] = pd.DataFrame({
                'HL': df['high'] - df['low'],
                'HC': abs(df['high'] - df['close'].shift(1)),
                'LC': abs(df['low'] - df['close'].shift(1))
            }).max(axis=1)
            df['ATR'] = df['TR'].rolling(window=14).mean()
            
            # Объемные индикаторы
            df['Volume_SMA'] = df['volume'].rolling(window=20).mean()
            df['Volume_Ratio'] = df['volume'] / df['Volume_SMA']
            
            return df
            
        except Exception as e:
            self.logger.error(f"Ошибка при расчете индикаторов: {str(e)}")
            return None

    def analyze_symbol(self, symbol):
        """
        Анализ торговой пары
        
        Args:
            symbol (str): Торговая пара
            
        Returns:
            dict: Результаты анализа
        """
        try:
            self.logger.info(f"Анализ {symbol}...")
            
            # Получение исторических данных
            df = self.data_collector.get_historical_data(symbol)
            if df is None or len(df) < 200:
                self.logger.warning(f"Недостаточно данных для анализа {symbol}")
                return None
                
            # Расчет индикаторов
            df = self.calculate_indicators(df)
            if df is None:
                return None
                
            # Получение последних значений
            current_price = df['close'].iloc[-1]
            sma_20 = df['SMA_20'].iloc[-1]
            sma_50 = df['SMA_50'].iloc[-1]
            sma_200 = df['SMA_200'].iloc[-1]
            rsi = df['RSI'].iloc[-1]
            macd = df['MACD'].iloc[-1]
            signal = df['Signal_Line'].iloc[-1]
            macd_hist = df['MACD_Hist'].iloc[-1]
            stoch_k = df['%K'].iloc[-1]
            stoch_d = df['%D'].iloc[-1]
            bb_upper = df['BB_upper'].iloc[-1]
            bb_lower = df['BB_lower'].iloc[-1]
            volume_ratio = df['Volume_Ratio'].iloc[-1]
            
            # Определение тренда
            trend_strength = 0
            if current_price > sma_20 > sma_50 > sma_200:
                trend = 'ВОСХОДЯЩИЙ'
                trend_strength += 1
            elif current_price < sma_20 < sma_50 < sma_200:
                trend = 'НИСХОДЯЩИЙ'
                trend_strength -= 1
            else:
                trend = 'БОКОВОЙ'
                
            # Анализ RSI
            if rsi > 70:
                rsi_signal = 'ПЕРЕПРОДАНО'
                trend_strength -= 1
            elif rsi < 30:
                rsi_signal = 'ПЕРЕКУПЛЕНО'
                trend_strength += 1
            else:
                rsi_signal = 'НЕЙТРАЛЬНО'
                
            # Анализ MACD
            if macd > signal and macd_hist > 0:
                macd_signal = 'РОСТ'
                trend_strength += 1
            elif macd < signal and macd_hist < 0:
                macd_signal = 'ПАДЕНИЕ'
                trend_strength -= 1
            else:
                macd_signal = 'НЕЙТРАЛЬНО'
                
            # Анализ Stochastic
            if stoch_k > 80 and stoch_d > 80:
                stoch_signal = 'ПЕРЕПРОДАНО'
                trend_strength -= 1
            elif stoch_k < 20 and stoch_d < 20:
                stoch_signal = 'ПЕРЕКУПЛЕНО'
                trend_strength += 1
            else:
                stoch_signal = 'НЕЙТРАЛЬНО'
                
            # Анализ Bollinger Bands
            if current_price > bb_upper:
                bb_signal = 'ПЕРЕПРОДАНО'
                trend_strength -= 1
            elif current_price < bb_lower:
                bb_signal = 'ПЕРЕКУПЛЕНО'
                trend_strength += 1
            else:
                bb_signal = 'НЕЙТРАЛЬНО'
                
            # Анализ объема
            if volume_ratio > 1.5:
                volume_signal = 'ВЫСОКИЙ'
                trend_strength *= 1.2
            elif volume_ratio < 0.5:
                volume_signal = 'НИЗКИЙ'
                trend_strength *= 0.8
            else:
                volume_signal = 'НОРМАЛЬНЫЙ'
                
            # Расчет итоговой уверенности
            confidence = min(abs(trend_strength) / 4, 0.95)  # Нормализация до 0.95
            
            self.logger.info(f"Анализ {symbol} завершен")
            
            return {
                'trend': trend,
                'rsi': rsi,
                'rsi_signal': rsi_signal,
                'macd': macd,
                'macd_signal': macd_signal,
                'stoch_k': stoch_k,
                'stoch_d': stoch_d,
                'stoch_signal': stoch_signal,
                'bb_signal': bb_signal,
                'volume_signal': volume_signal,
                'confidence': confidence
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка при анализе {symbol}: {str(e)}")
            return None

    def format_analysis(self, analysis):
        """
        Форматирование результатов анализа для вывода
        
        :param analysis: Словарь с результатами анализа
        :return: Отформатированная строка
        """
        if analysis is None:
            return "Нет данных для анализа"
            
        output = []
        output.append(f"📊 Технический анализ")
        output.append(f"📈 Тренд: {analysis['trend']}")
        
        output.append("\n📊 RSI:")
        output.append(f"Значение: {analysis['rsi']:.2f}")
        output.append(f"Сигнал: {analysis['rsi_signal']}")
        
        output.append("\n📊 MACD:")
        output.append(f"Значение: {analysis['macd']:.2f}")
        output.append(f"Сигнал: {analysis['macd_signal']}")
        
        output.append("\n📊 Stochastic:")
        output.append(f"%K: {analysis['stoch_k']:.2f}")
        output.append(f"%D: {analysis['stoch_d']:.2f}")
        output.append(f"Сигнал: {analysis['stoch_signal']}")
        
        output.append("\n📊 Bollinger Bands:")
        output.append(f"Сигнал: {analysis['bb_signal']}")
        
        output.append("\n📊 Объем:")
        output.append(f"Сигнал: {analysis['volume_signal']}")
        
        output.append(f"\nУверенность: {analysis['confidence']:.2f}")
        return "\n".join(output)

def test_technical_analyzer():
    """Тестирование TechnicalAnalyzer"""
    try:
        logging.info("Начало тестирования TechnicalAnalyzer")
        
        # Инициализация сборщика данных
        collector = DataCollector(test_mode=True)
        
        # Инициализация анализатора
        analyzer = TechnicalAnalyzer(collector)
        
        # Тест анализа BTC/USDT
        logging.info("\nТест анализа BTC/USDT")
        analysis = analyzer.analyze_symbol('BTC/USDT')
        if analysis is not None:
            print(analyzer.format_analysis(analysis))
        
    except Exception as e:
        logging.error(f"Ошибка при тестировании: {str(e)}")
        raise

if __name__ == "__main__":
    test_technical_analyzer() 