import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import xgboost as xgb
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
import logging
from data_collector import DataCollector
import matplotlib.pyplot as plt
import seaborn as sns
import ta
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ml_analyzer.log', mode='w'),
        logging.StreamHandler()
    ]
)

class MLAnalyzer:
    def __init__(self, data_collector):
        """
        Инициализация ML анализатора
        
        Args:
            data_collector (DataCollector): Объект для сбора данных
        """
        self.logger = logging.getLogger(__name__)
        self.data_collector = data_collector
        self.model = self._create_model()
        self.logger.info("MLAnalyzer успешно инициализирован")
        
    def _create_model(self):
        """
        Создание модели LSTM
        
        Returns:
            tf.keras.Model: Созданная модель
        """
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(60, 5)),
            Dropout(0.2),
            LSTM(50, return_sequences=False),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        
        model.compile(optimizer='adam', loss='mse')
        return model
        
    def prepare_data(self, df):
        """
        Подготовка данных для модели
        
        Args:
            df (pd.DataFrame): DataFrame с историческими данными
            
        Returns:
            tuple: (X, y) - подготовленные данные для обучения
        """
        try:
            # Нормализация данных
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(df[['open', 'high', 'low', 'close', 'volume']])
            
            X, y = [], []
            for i in range(60, len(scaled_data)):
                X.append(scaled_data[i-60:i])
                y.append(scaled_data[i, 3])  # Используем close price
                
            return np.array(X), np.array(y)
            
        except Exception as e:
            self.logger.error(f"Ошибка при подготовке данных: {str(e)}")
            return None, None
            
    def train_model(self, symbol):
        """
        Обучение модели на исторических данных
        
        Args:
            symbol (str): Торговая пара
            
        Returns:
            bool: True если обучение успешно, False в противном случае
        """
        try:
            self.logger.info(f"Обучение модели для {symbol}...")
            
            # Получение исторических данных
            df = self.data_collector.get_historical_data(symbol)
            if df is None or len(df) < 200:
                self.logger.warning(f"Недостаточно данных для обучения {symbol}")
                return False
                
            # Подготовка данных
            X, y = self.prepare_data(df)
            if X is None or y is None:
                return False
                
            # Обучение модели
            self.model.fit(X, y, epochs=50, batch_size=32, verbose=0)
            
            self.logger.info(f"Модель для {symbol} успешно обучена")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка при обучении модели: {str(e)}")
            return False
            
    def predict(self, symbol):
        """
        Прогнозирование цены
        
        Args:
            symbol (str): Торговая пара
            
        Returns:
            dict: Результаты прогноза
        """
        try:
            self.logger.info(f"Прогнозирование для {symbol}...")
            
            # Получение исторических данных
            df = self.data_collector.get_historical_data(symbol)
            if df is None or len(df) < 60:
                self.logger.warning(f"Недостаточно данных для прогноза {symbol}")
                return None
                
            # Подготовка данных
            scaler = MinMaxScaler()
            scaled_data = scaler.fit_transform(df[['open', 'high', 'low', 'close', 'volume']].tail(60))
            
            # Прогноз
            prediction = self.model.predict(np.array([scaled_data]), verbose=0)
            predicted_price = scaler.inverse_transform([[0, 0, 0, prediction[0][0], 0]])[0][3]
            
            # Расчет уверенности
            current_price = df['close'].iloc[-1]
            price_change = abs(predicted_price - current_price) / current_price
            confidence = max(0, 1 - price_change)
            
            self.logger.info(f"Прогноз для {symbol} завершен")
            
            return {
                'predicted_price': predicted_price,
                'current_price': current_price,
                'confidence': confidence
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка при прогнозировании: {str(e)}")
            return None

    def add_technical_indicators(self, df):
        """
        Добавление технических индикаторов
        
        :param df: DataFrame с историческими данными
        :return: DataFrame с добавленными индикаторами
        """
        # RSI
        df['rsi'] = ta.momentum.RSIIndicator(df['close']).rsi()
        
        # MACD
        macd = ta.trend.MACD(df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        
        # Bollinger Bands
        bollinger = ta.volatility.BollingerBands(df['close'])
        df['bb_high'] = bollinger.bollinger_hband()
        df['bb_low'] = bollinger.bollinger_lband()
        df['bb_mid'] = bollinger.bollinger_mavg()
        
        # Stochastic
        stoch = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close'])
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()
        
        # ATR
        df['atr'] = ta.volatility.AverageTrueRange(df['high'], df['low'], df['close']).average_true_range()
        
        # Momentum
        df['momentum'] = ta.momentum.ROCIndicator(df['close']).roc()
        
        # Дополнительные индикаторы
        df['ema_9'] = ta.trend.EMAIndicator(df['close'], window=9).ema_indicator()
        df['ema_21'] = ta.trend.EMAIndicator(df['close'], window=21).ema_indicator()
        df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx()
        
        # Заполняем NaN значения
        df = df.bfill().ffill()
        
        return df

    def create_lstm_model(self, input_shape):
        """
        Создание улучшенной модели LSTM
        
        :param input_shape: Форма входных данных
        :return: Модель LSTM
        """
        model = Sequential([
            Input(shape=input_shape),
            LSTM(units=128, return_sequences=True),
            Dropout(0.3),
            LSTM(units=64, return_sequences=True),
            Dropout(0.3),
            LSTM(units=32, return_sequences=False),
            Dropout(0.3),
            Dense(units=16, activation='relu'),
            Dense(units=1, activation='sigmoid')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'Precision', 'Recall']
        )
        return model

    def create_xgboost_model(self):
        """
        Создание улучшенной модели XGBoost
        
        :return: Модель XGBoost
        """
        return xgb.XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=7,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            random_state=42
        )

    def create_random_forest_model(self):
        """
        Создание улучшенной модели Random Forest
        
        :return: Модель Random Forest
        """
        return RandomForestClassifier(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42
        )

    def evaluate_model(self, y_true, y_pred, model_name):
        """
        Оценка модели
        
        :param y_true: Истинные значения
        :param y_pred: Предсказанные значения
        :param model_name: Название модели
        :return: Словарь с метриками
        """
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred)
        }

    def generate_trading_signals(self, symbol, current_data):
        """
        Генерация торговых сигналов на основе ML моделей
        
        Args:
            symbol (str): Торговая пара
            current_data (pd.DataFrame): Текущие данные
            
        Returns:
            dict: Торговые сигналы
        """
        try:
            signals = []
            confidence = 0.0
            
            # 1. LSTM прогноз
            lstm_prediction = self.predict(symbol)
            if lstm_prediction:
                price_change = (lstm_prediction['predicted_price'] - lstm_prediction['current_price']) / lstm_prediction['current_price']
                if abs(price_change) > 0.01:  # Изменение более 1%
                    signals.append({
                        'type': 'LSTM',
                        'message': f"Прогноз цены: {lstm_prediction['predicted_price']:.2f}",
                        'action': 'BUY' if price_change > 0 else 'SELL',
                        'confidence': lstm_prediction['confidence']
                    })
                    confidence += lstm_prediction['confidence'] * 0.3
            
            # 2. XGBoost прогноз
            xgb_model = self.create_xgboost_model()
            if xgb_model:
                features = self.prepare_features(current_data)
                xgb_pred = xgb_model.predict_proba(features)[0]
                if max(xgb_pred) > 0.7:  # Высокая уверенность
                    action = 'BUY' if xgb_pred[1] > xgb_pred[0] else 'SELL'
                    signals.append({
                        'type': 'XGBoost',
                        'message': f"Вероятность роста: {xgb_pred[1]:.2f}",
                        'action': action,
                        'confidence': max(xgb_pred)
                    })
                    confidence += max(xgb_pred) * 0.3
            
            # 3. Random Forest прогноз
            rf_model = self.create_random_forest_model()
            if rf_model:
                rf_pred = rf_model.predict_proba(features)[0]
                if max(rf_pred) > 0.7:  # Высокая уверенность
                    action = 'BUY' if rf_pred[1] > rf_pred[0] else 'SELL'
                    signals.append({
                        'type': 'RandomForest',
                        'message': f"Вероятность роста: {rf_pred[1]:.2f}",
                        'action': action,
                        'confidence': max(rf_pred)
                    })
                    confidence += max(rf_pred) * 0.3
            
            # 4. Анализ тренда
            trend = self.analyze_trend(current_data)
            if trend:
                signals.append({
                    'type': 'TREND',
                    'message': f"Тренд: {trend['direction']}",
                    'action': trend['action'],
                    'confidence': trend['confidence']
                })
                confidence += trend['confidence'] * 0.1
            
            # Определение итогового действия
            if not signals:
                return {
                    'action': 'HOLD',
                    'confidence': 0.0,
                    'signals': []
                }
            
            # Подсчет действий
            actions = {'BUY': 0, 'SELL': 0, 'HOLD': 0}
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
            
        except Exception as e:
            self.logger.error(f"Ошибка при генерации сигналов: {str(e)}")
            return None
            
    def analyze_trend(self, df):
        """
        Анализ тренда на основе технических индикаторов
        
        Args:
            df (pd.DataFrame): DataFrame с данными
            
        Returns:
            dict: Результаты анализа тренда
        """
        try:
            # Добавление индикаторов
            df = self.add_technical_indicators(df)
            
            # Получение последних значений
            last_row = df.iloc[-1]
            
            # Анализ тренда
            trend_strength = 0
            
            # EMA
            if last_row['ema_9'] > last_row['ema_21']:
                trend_strength += 1
            else:
                trend_strength -= 1
            
            # ADX
            if last_row['adx'] > 25:  # Сильный тренд
                if last_row['close'] > last_row['ema_9']:
                    trend_strength += 1
                else:
                    trend_strength -= 1
            
            # RSI
            if last_row['rsi'] > 70:
                trend_strength -= 1
            elif last_row['rsi'] < 30:
                trend_strength += 1
            
            # MACD
            if last_row['macd'] > last_row['macd_signal']:
                trend_strength += 1
            else:
                trend_strength -= 1
            
            # Определение направления тренда
            if trend_strength > 0:
                direction = 'ВОСХОДЯЩИЙ'
                action = 'BUY'
            elif trend_strength < 0:
                direction = 'НИСХОДЯЩИЙ'
                action = 'SELL'
            else:
                direction = 'БОКОВОЙ'
                action = 'HOLD'
            
            # Расчет уверенности
            confidence = min(abs(trend_strength) / 4, 0.9)
            
            return {
                'direction': direction,
                'action': action,
                'confidence': confidence
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка при анализе тренда: {str(e)}")
            return None

    def analyze_symbol(self, symbol, timeframe='1h', limit=1000):
        """
        Анализ символа с использованием различных моделей ML
        
        :param symbol: Торговая пара
        :param timeframe: Временной интервал
        :param limit: Количество свечей
        :return: Результаты анализа
        """
        try:
            logging.info(f"\nАнализ {symbol}...")
            
            # Получение данных
            df = self.data_collector.get_historical_data(symbol, timeframe, limit)
            if df is None or len(df) < 100:
                logging.error(f"Недостаточно данных для анализа {symbol}")
                return None
            
            # Подготовка данных
            X, y = self.prepare_data(df)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Обучение и оценка моделей
            results = {}
            self.models[symbol] = {}
            
            # LSTM
            lstm_model = self.create_lstm_model(input_shape=(X.shape[1], X.shape[2]))
            lstm_model.fit(
                X_train, y_train,
                epochs=100,
                batch_size=32,
                validation_split=0.1,
                verbose=0
            )
            lstm_pred = (lstm_model.predict(X_test) > 0.5).astype(int)
            results['LSTM'] = self.evaluate_model(y_test, lstm_pred, 'LSTM')
            self.models[symbol]['LSTM'] = lstm_model
            
            # XGBoost
            xgb_model = self.create_xgboost_model()
            xgb_model.fit(
                X_train.reshape(X_train.shape[0], -1),
                y_train,
                eval_set=[(X_test.reshape(X_test.shape[0], -1), y_test)],
                verbose=0
            )
            xgb_pred = xgb_model.predict(X_test.reshape(X_test.shape[0], -1))
            results['XGBoost'] = self.evaluate_model(y_test, xgb_pred, 'XGBoost')
            self.models[symbol]['XGBoost'] = xgb_model
            
            # Random Forest
            rf_model = self.create_random_forest_model()
            rf_model.fit(X_train.reshape(X_train.shape[0], -1), y_train)
            rf_pred = rf_model.predict(X_test.reshape(X_test.shape[0], -1))
            results['Random Forest'] = self.evaluate_model(y_test, rf_pred, 'Random Forest')
            self.models[symbol]['Random Forest'] = rf_model
            
            # Визуализация результатов
            self.plot_results(results, symbol)
            
            # Вывод результатов
            logging.info(f"\nРезультаты анализа {symbol}:")
            for model_name, metrics in results.items():
                logging.info(f"\n{model_name}:")
                for metric_name, value in metrics.items():
                    logging.info(f"{metric_name}: {value:.4f}")

            # Генерация торговых сигналов
            signals = self.generate_trading_signals(symbol, df)
            if signals:
                logging.info(f"\nТорговые сигналы для {symbol}:")
                logging.info(f"Сигнал: {signals['action']}")
                logging.info(f"Уверенность: {signals['confidence']:.2%}")
                logging.info("\nПредсказания моделей:")
                for signal in signals['signals']:
                    logging.info(f"{signal['type']}: {signal['message']}")
            
            return results, signals
            
        except Exception as e:
            logging.error(f"Ошибка при анализе {symbol}: {str(e)}")
            return None

    def plot_results(self, results, symbol):
        """
        Визуализация результатов анализа
        
        :param results: Результаты анализа
        :param symbol: Торговая пара
        """
        metrics = ['accuracy', 'precision', 'recall', 'f1']
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Метрики моделей для {symbol}')
        
        for idx, metric in enumerate(metrics):
            ax = axes[idx // 2, idx % 2]
            values = [results[model][metric] for model in results.keys()]
            ax.bar(results.keys(), values)
            ax.set_title(metric.capitalize())
            ax.set_ylim(0, 1)
            
            # Добавление значений на график
            for i, v in enumerate(values):
                ax.text(i, v + 0.01, f'{v:.4f}', ha='center')
        
        plt.tight_layout()
        plt.savefig(f'ml_analysis_{symbol.replace("/", "_")}.png')
        plt.close()

def test_ml_analyzer():
    """Тестирование MLAnalyzer"""
    try:
        logging.info("Начало тестирования MLAnalyzer")
        collector = DataCollector(test_mode=True)
        analyzer = MLAnalyzer(collector)
        # Тест анализа BTC/USDT
        logging.info("\nТест анализа BTC/USDT")
        result = analyzer.predict('BTC/USDT')
        if result is not None:
            print(result)
    except Exception as e:
        logging.error(f"Ошибка при тестировании MLAnalyzer: {str(e)}")
        raise

if __name__ == "__main__":
    test_ml_analyzer() 