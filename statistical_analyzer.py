import pandas as pd
import numpy as np
from scipy import stats
import logging
from datetime import datetime, timedelta
import ccxt
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import warnings
import json
from data_collector import DataCollector
warnings.filterwarnings('ignore')

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('statistical_analysis.log'),
        logging.StreamHandler()
    ]
)

class StatisticalAnalyzer:
    def __init__(self, api_key=None, api_secret=None):
        """
        Инициализация анализатора статистики
        
        :param api_key: API ключ биржи
        :param api_secret: API секрет биржи
        """
        self.data_collector = DataCollector(test_mode=True)
        self.symbols = ['BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT']
        self.timeframes = ['1h', '4h', '1d']
        self.logger = logging.getLogger(__name__)
        
    def get_historical_data(self, symbol, timeframe='1h', limit=500):
        """Получение исторических данных"""
        try:
            df = self.data_collector.get_historical_data(symbol, timeframe, limit)
            return df
        except Exception as e:
            self.logger.error(f"Ошибка при получении данных для {symbol}: {str(e)}")
            return None

    def analyze_correlations(self):
        """Анализ корреляций между монетами"""
        self.logger.info("Начало анализа корреляций...")
        
        # Получение данных для всех символов
        data = {}
        for symbol in self.symbols:
            df = self.get_historical_data(symbol)
            if df is not None:
                data[symbol] = df['close']
        
        # Создание DataFrame с ценами всех монет
        prices_df = pd.DataFrame(data)
        
        # Расчет корреляций
        correlation_matrix = prices_df.corr()
        
        # Визуализация корреляций
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
        plt.title('Корреляция между криптовалютами')
        plt.tight_layout()
        plt.savefig('correlation_heatmap.png')
        plt.close()
        
        # Анализ сильных корреляций
        strong_correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                if abs(correlation_matrix.iloc[i, j]) > 0.7:
                    strong_correlations.append({
                        'pair': f"{correlation_matrix.columns[i]} - {correlation_matrix.columns[j]}",
                        'correlation': correlation_matrix.iloc[i, j]
                    })
        
        return {
            'correlation_matrix': correlation_matrix,
            'strong_correlations': strong_correlations
        }

    def analyze_volume(self, symbol='BTC/USDT'):
        """Анализ объемов торгов"""
        self.logger.info(f"Начало анализа объемов для {symbol}...")
        
        df = self.get_historical_data(symbol)
        if df is None:
            return None
        
        # Расчет среднего объема
        avg_volume = df['volume'].mean()
        
        # Анализ объема по дням недели
        df['day_of_week'] = df.index.dayofweek
        volume_by_day = df.groupby('day_of_week')['volume'].mean()
        
        # Анализ объема по часам
        df['hour'] = df.index.hour
        volume_by_hour = df.groupby('hour')['volume'].mean()
        
        # Визуализация
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        volume_by_day.plot(kind='bar', ax=ax1)
        ax1.set_title('Средний объем по дням недели')
        ax1.set_xlabel('День недели')
        ax1.set_ylabel('Объем')
        
        volume_by_hour.plot(kind='bar', ax=ax2)
        ax2.set_title('Средний объем по часам')
        ax2.set_xlabel('Час')
        ax2.set_ylabel('Объем')
        
        plt.tight_layout()
        plt.savefig('volume_analysis.png')
        plt.close()
        
        return {
            'average_volume': avg_volume,
            'volume_by_day': volume_by_day,
            'volume_by_hour': volume_by_hour
        }

    def analyze_volatility(self, symbol='BTC/USDT'):
        """Анализ волатильности"""
        self.logger.info(f"Начало анализа волатильности для {symbol}...")
        
        df = self.get_historical_data(symbol)
        if df is None:
            return None
        
        # Расчет дневной волатильности
        df['returns'] = df['close'].pct_change()
        daily_volatility = df['returns'].std() * np.sqrt(24)  # Годовая волатильность
        
        # Расчет волатильности по дням недели
        df['day_of_week'] = df.index.dayofweek
        volatility_by_day = df.groupby('day_of_week')['returns'].std() * np.sqrt(24)
        
        # Расчет волатильности по часам
        df['hour'] = df.index.hour
        volatility_by_hour = df.groupby('hour')['returns'].std() * np.sqrt(24)
        
        # Визуализация
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        volatility_by_day.plot(kind='bar', ax=ax1)
        ax1.set_title('Волатильность по дням недели')
        ax1.set_xlabel('День недели')
        ax1.set_ylabel('Волатильность')
        
        volatility_by_hour.plot(kind='bar', ax=ax2)
        ax2.set_title('Волатильность по часам')
        ax2.set_xlabel('Час')
        ax2.set_ylabel('Волатильность')
        
        plt.tight_layout()
        plt.savefig('volatility_analysis.png')
        plt.close()
        
        return {
            'daily_volatility': daily_volatility,
            'volatility_by_day': volatility_by_day,
            'volatility_by_hour': volatility_by_hour
        }

    def analyze_seasonality(self, symbol='BTC/USDT'):
        """Анализ сезонности"""
        self.logger.info(f"Начало анализа сезонности для {symbol}...")
        
        df = self.get_historical_data(symbol)
        if df is None:
            return None
        
        # Проверка на стационарность
        def check_stationarity(timeseries):
            result = adfuller(timeseries.dropna())
            return result[1] < 0.05
        
        # Декомпозиция временного ряда
        decomposition = seasonal_decompose(df['close'], period=24)
        
        # Визуализация
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(12, 12))
        
        df['close'].plot(ax=ax1)
        ax1.set_title('Исходный ряд')
        
        decomposition.trend.plot(ax=ax2)
        ax2.set_title('Тренд')
        
        decomposition.seasonal.plot(ax=ax3)
        ax3.set_title('Сезонность')
        
        decomposition.resid.plot(ax=ax4)
        ax4.set_title('Остатки')
        
        plt.tight_layout()
        plt.savefig('seasonality_analysis.png')
        plt.close()
        
        # Анализ сезонности по месяцам
        df['month'] = df.index.month
        monthly_avg = df.groupby('month')['close'].mean()
        
        # Анализ сезонности по дням недели
        df['day_of_week'] = df.index.dayofweek
        daily_avg = df.groupby('day_of_week')['close'].mean()
        
        return {
            'is_stationary': check_stationarity(df['close']),
            'monthly_pattern': monthly_avg,
            'daily_pattern': daily_avg,
            'decomposition': decomposition
        }

    def generate_signals(self, results):
        """Генерация торговых сигналов на основе статистического анализа"""
        signals = []
        confidence = 0.0
        
        # 1. Корреляция
        corr = results.get('correlations', {})
        if corr and 'correlation_matrix' in corr:
            matrix = corr['correlation_matrix']
            if 'BTC/USDT' in matrix.columns and 'ETH/USDT' in matrix.columns:
                btc_eth_corr = matrix.loc['BTC/USDT', 'ETH/USDT']
                if btc_eth_corr > 0.8:
                    signals.append({
                        'type': 'CORRELATION',
                        'message': f"BTC и ETH сильно коррелируют ({btc_eth_corr:.2f})",
                        'action': 'FOLLOW_BTC',
                        'confidence': min(btc_eth_corr, 0.9)
                    })
                    confidence += 0.2
        
        # 2. Объёмы
        volume = results.get('volume', {})
        if volume:
            avg_vol = volume.get('average_volume', 0)
            vol_by_hour = volume.get('volume_by_hour')
            if vol_by_hour is not None:
                current_hour = datetime.now().hour
                current_vol = vol_by_hour.get(current_hour, 0)
                vol_ratio = current_vol / avg_vol if avg_vol > 0 else 1
                
                if vol_ratio > 1.5:
                    signals.append({
                        'type': 'VOLUME',
                        'message': f"Объём выше среднего в {current_hour}:00 ({vol_ratio:.1f}x)",
                        'action': 'BUY',
                        'confidence': min(vol_ratio / 2, 0.8)
                    })
                    confidence += 0.15
                elif vol_ratio < 0.5:
                    signals.append({
                        'type': 'VOLUME',
                        'message': f"Объём ниже среднего в {current_hour}:00 ({vol_ratio:.1f}x)",
                        'action': 'SELL',
                        'confidence': min((1 - vol_ratio) / 2, 0.8)
                    })
                    confidence += 0.15
        
        # 3. Волатильность
        volatility = results.get('volatility', {})
        if volatility:
            daily_vol = volatility.get('daily_volatility', 0)
            vol_by_hour = volatility.get('volatility_by_hour')
            if vol_by_hour is not None:
                current_hour = datetime.now().hour
                current_vol = vol_by_hour.get(current_hour, 0)
                
                if current_vol > daily_vol * 1.5:
                    signals.append({
                        'type': 'VOLATILITY',
                        'message': f"Высокая волатильность в {current_hour}:00",
                        'action': 'HOLD',
                        'confidence': 0.9
                    })
                    confidence -= 0.2
                elif current_vol < daily_vol * 0.5:
                    signals.append({
                        'type': 'VOLATILITY',
                        'message': f"Низкая волатильность в {current_hour}:00",
                        'action': 'TRADE',
                        'confidence': 0.7
                    })
                    confidence += 0.1
        
        # 4. Сезонность
        seasonality = results.get('seasonality', {})
        if seasonality:
            monthly_pattern = seasonality.get('monthly_pattern')
            daily_pattern = seasonality.get('daily_pattern')
            
            if monthly_pattern is not None and daily_pattern is not None:
                current_month = datetime.now().month
                current_day = datetime.now().weekday()
                
                month_trend = monthly_pattern[current_month] > monthly_pattern.mean()
                day_trend = daily_pattern[current_day] > daily_pattern.mean()
                
                if month_trend and day_trend:
                    signals.append({
                        'type': 'SEASONALITY',
                        'message': "Положительная сезонность по месяцам и дням",
                        'action': 'BUY',
                        'confidence': 0.6
                    })
                    confidence += 0.15
                elif not month_trend and not day_trend:
                    signals.append({
                        'type': 'SEASONALITY',
                        'message': "Отрицательная сезонность по месяцам и дням",
                        'action': 'SELL',
                        'confidence': 0.6
                    })
                    confidence += 0.15
        
        # Определение итогового действия
        if not signals:
            return {
                'action': 'HOLD',
                'confidence': 0.0,
                'signals': []
            }
            
        # Подсчет действий
        actions = {'BUY': 0, 'SELL': 0, 'HOLD': 0, 'FOLLOW_BTC': 0, 'TRADE': 0}
        for signal in signals:
            actions[signal['action']] += signal['confidence']
            
        # Определение основного действия
        main_action = max(actions.items(), key=lambda x: x[1])[0]
        
        # Нормализация уверенности
        final_confidence = min(confidence / len(signals), 0.95)
        
        return {
            'action': main_action,
            'confidence': final_confidence,
            'signals': signals
        }

    def run_analysis(self):
        """Запуск полного анализа"""
        self.logger.info("Запуск полного статистического анализа...")
        
        results = {
            'correlations': self.analyze_correlations(),
            'volume': self.analyze_volume(),
            'volatility': self.analyze_volatility(),
            'seasonality': self.analyze_seasonality()
        }
        
        # Вывод результатов
        self.logger.info("\nРезультаты анализа:")
        
        # Корреляции
        if results['correlations']:
            self.logger.info("\nСильные корреляции:")
            for corr in results['correlations']['strong_correlations']:
                self.logger.info(f"{corr['pair']}: {corr['correlation']:.2f}")
        
        # Объемы
        if results['volume']:
            self.logger.info(f"\nСредний объем торгов: {results['volume']['average_volume']:.2f}")
        
        # Волатильность
        if results['volatility']:
            self.logger.info(f"\nДневная волатильность: {results['volatility']['daily_volatility']:.2%}")
        
        # Сезонность
        if results['seasonality']:
            self.logger.info(f"\nСтационарность ряда: {'Да' if results['seasonality']['is_stationary'] else 'Нет'}")
        
        self.generate_signals(results)
        return results

    def analyze_symbol(self, symbol):
        """
        Анализ торговой пары и генерация сигнала
        
        :param symbol: Торговая пара
        :return: Словарь с результатами анализа
        """
        try:
            # Получение исторических данных
            df = self.get_historical_data(symbol)
            if df is None:
                return None
                
            # Анализ волатильности
            volatility = self.analyze_volatility(symbol)
            if volatility is None:
                return None
                
            # Анализ сезонности
            seasonality = self.analyze_seasonality(symbol)
            if seasonality is None:
                return None
                
            # Анализ объемов
            volume = self.analyze_volume(symbol)
            if volume is None:
                return None
                
            # Генерация сигнала
            current_price = df['close'].iloc[-1]
            avg_price = df['close'].mean()
            current_volatility = volatility['daily_volatility']
            avg_volume = volume['average_volume']
            current_volume = df['volume'].iloc[-1]
            
            # Определение сигнала
            signal = 'HOLD'
            confidence = 0.5
            
            # Если цена выше среднего и объем высокий
            if current_price > avg_price and current_volume > avg_volume * 1.2:
                signal = 'BUY'
                confidence = 0.7
            # Если цена ниже среднего и объем высокий
            elif current_price < avg_price and current_volume > avg_volume * 1.2:
                signal = 'SELL'
                confidence = 0.7
                
            return {
                'signal': signal,
                'confidence': confidence,
                'current_price': current_price,
                'average_price': avg_price,
                'volatility': current_volatility,
                'volume_ratio': current_volume / avg_volume,
                'is_stationary': seasonality['is_stationary']
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка при анализе {symbol}: {str(e)}")
            return None

if __name__ == "__main__":
    # Загрузка конфигурации
    with open('config.json', 'r', encoding='utf-8') as f:
        config = json.load(f)
    API_KEY = config['binance_api_key']
    API_SECRET = config['binance_api_secret']
    
    # Создание и запуск анализатора
    analyzer = StatisticalAnalyzer(API_KEY, API_SECRET)
    results = analyzer.run_analysis() 