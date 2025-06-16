import logging
from datetime import datetime
import time
from data_collector import DataCollector
from technical_analyzer import TechnicalAnalyzer
from statistical_analyzer import StatisticalAnalyzer
from ml_analyzer import MLAnalyzer
from test_trader import TestTrader
from sentiment_analyzer import SentimentAnalyzer

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('trading_bot.log', mode='w'),
        logging.StreamHandler()
    ]
)

class TradingBot:
    def __init__(self, test_mode=True):
        """
        Инициализация торгового бота
        
        Args:
            test_mode (bool): Режим тестирования
        """
        self.logger = logging.getLogger(__name__)
        
        # Инициализация компонентов
        self.data_collector = DataCollector(test_mode=test_mode)
        self.technical_analyzer = TechnicalAnalyzer(self.data_collector)
        self.statistical_analyzer = StatisticalAnalyzer(self.data_collector)
        self.ml_analyzer = MLAnalyzer(self.data_collector)
        self.sentiment_analyzer = SentimentAnalyzer(self.data_collector)
        self.trader = TestTrader()
        
        # Параметры торговли
        self.symbols = self.data_collector.symbols
        self.min_confidence = 0.7
        
        self.logger.info("TradingBot успешно инициализирован")
        
    def make_trading_decision(self, signals):
        """
        Принятие торгового решения на основе сигналов
        
        :param signals: Список сигналов от анализаторов
        :return: (action, confidence) - действие и уверенность
        """
        if not signals:
            return 'HOLD', 0.0
            
        # Проверяем, есть ли сигнал от ML анализатора с высокой уверенностью
        ml_signals = [s for s in signals if s.get('type') == 'LSTM' and s.get('confidence', 0) > 0.9]
        if ml_signals:
            ml_signal = ml_signals[0]
            return ml_signal['action'], ml_signal['confidence']
            
        # Если нет сильного ML сигнала, используем взвешенное решение
        buy_weight = 0
        sell_weight = 0
        
        for signal in signals:
            weight = signal.get('confidence', 0)
            if signal.get('action') == 'BUY':
                buy_weight += weight
            elif signal.get('action') == 'SELL':
                sell_weight += weight
                
        if buy_weight > sell_weight and buy_weight >= self.min_confidence:
            return 'BUY', buy_weight
        elif sell_weight > buy_weight and sell_weight >= self.min_confidence:
            return 'SELL', sell_weight
        else:
            return 'HOLD', 0.0
            
    def execute_trade(self, symbol, action, confidence):
        """
        Выполнение торговой операции
        
        :param symbol: Торговая пара
        :param action: Действие (BUY/SELL)
        :param confidence: Уверенность в сигнале
        """
        try:
            # Получаем текущую цену
            current_price = self.data_collector.get_current_price(symbol)
            if not current_price:
                self.logger.error(f"Не удалось получить текущую цену для {symbol}")
                return
                
            # Получаем волатильность
            volatility = 0.1  # Фиксированное значение для тестирования
                
            # Рассчитываем размер позиции на основе уверенности и риска
            max_risk = 0.02  # Максимальный риск 2% от баланса
            position_size = self.trader.get_balance() * max_risk * confidence
            
            # Проверяем волатильность
            if volatility > 0.2:  # Высокая волатильность
                position_size *= 0.5  # Уменьшаем размер позиции
                
            # Проверяем тренд
            trend = self.technical_analyzer.analyze_symbol(symbol)['trend']
            if action == 'SELL' and trend == 'ВОСХОДЯЩИЙ' and confidence < 0.8:
                self.logger.info(f"Пропускаем продажу в восходящем тренде с низкой уверенностью: {symbol}")
                return
                
            # Проверяем убытки
            if self.trader.get_unrealized_pnl(symbol) < -0.05:  # Убыток более 5%
                self.logger.info(f"Пропускаем сделку из-за большого убытка: {symbol}")
                return
                
            # Выполняем сделку
            if action == 'BUY':
                if self.trader.get_balance() >= position_size:
                    self.trader.buy(symbol, position_size)
                    self.logger.info(f"Выполнена покупка {symbol} на сумму {position_size:.2f} USDT")
                else:
                    self.logger.warning(f"Недостаточно средств для покупки {symbol}")
            elif action == 'SELL':
                if self.trader.get_position(symbol) >= position_size:
                    self.trader.sell(symbol, position_size)
                    self.logger.info(f"Выполнена продажа {symbol} на сумму {position_size:.2f} USDT")
                else:
                    self.logger.warning(f"Недостаточно позиции для продажи {symbol}")
                    
        except Exception as e:
            self.logger.error(f"Ошибка при выполнении сделки: {str(e)}")
            
    def run(self):
        """Запуск торгового бота"""
        try:
            self.logger.info("Запуск торгового бота...")
            current_iteration = 0
            
            while True:
                for symbol in self.symbols:
                    try:
                        # Получение сигналов от всех анализаторов
                        technical_signal = self.technical_analyzer.analyze_symbol(symbol)
                        if technical_signal:
                            technical_signal['action'] = 'BUY' if technical_signal['trend'] == 'ВОСХОДЯЩИЙ' else 'SELL'
                        statistical_results = {
                            'correlations': None,
                            'volume': None,
                            'volatility': None,
                            'seasonality': None
                        }
                        # Для ускорения — только по одной паре, иначе можно добавить анализ для всех пар
                        statistical_signal = None
                        if hasattr(self.statistical_analyzer, 'generate_signals'):
                            # Если есть быстрый анализ по одной паре
                            statistical_signal = self.statistical_analyzer.generate_signals(statistical_results)
                        # ML анализ
                        ml_signal = None
                        if hasattr(self.ml_analyzer, 'predict'):
                            ml_raw = self.ml_analyzer.predict(symbol)
                            if ml_raw:
                                ml_signal = {
                                    'action': 'BUY' if ml_raw['predicted_price'] > ml_raw['current_price'] else 'SELL',
                                    'confidence': ml_raw['confidence'],
                                    'signals': [
                                        {
                                            'type': 'LSTM',
                                            'message': f"ML прогноз: {ml_raw['predicted_price']:.2f} (текущая цена: {ml_raw['current_price']:.2f})",
                                            'action': 'BUY' if ml_raw['predicted_price'] > ml_raw['current_price'] else 'SELL',
                                            'confidence': ml_raw['confidence']
                                        }
                                    ]
                                }
                        # Собираем все сигналы
                        all_signals = []
                        if technical_signal:
                            all_signals.append(technical_signal)
                        if statistical_signal:
                            all_signals.append(statistical_signal)
                        if ml_signal:
                            all_signals.append(ml_signal)
                        # Принятие решения на основе всех сигналов
                        action, confidence = self.make_trading_decision([
                            {'action': s['action'], 'confidence': s['confidence']} for s in all_signals if s
                        ])
                        # Логирование сигналов
                        self.logger.info(f"\nСигналы для {symbol}:")
                        if technical_signal:
                            self.logger.info(f"Технический анализ: {technical_signal['trend']} (уверенность: {technical_signal['confidence']:.2f})")
                            self.logger.info(f"Детали технического анализа: {technical_signal}")
                        if statistical_signal:
                            self.logger.info(f"Статистический анализ: {statistical_signal['action']} (уверенность: {statistical_signal['confidence']:.2f})")
                            self.logger.info(f"Детали статистического анализа: {statistical_signal}")
                        if ml_signal:
                            self.logger.info(f"ML анализ: {ml_signal['action']} (уверенность: {ml_signal['confidence']:.2f})")
                            self.logger.info(f"Детали ML анализа: {ml_signal}")
                        self.logger.info(f"Решение: {action} (уверенность: {confidence:.2f})")
                        # Выполнение сделки
                        if confidence >= self.min_confidence:
                            self.execute_trade(symbol, action, confidence)
                    except Exception as e:
                        self.logger.error(f"Ошибка при обработке {symbol}: {str(e)}")
                        continue
                current_iteration += 1
                time.sleep(60)  # Пауза между итерациями
        except KeyboardInterrupt:
            self.logger.info("Торговый бот остановлен")
        except Exception as e:
            self.logger.error(f"Критическая ошибка: {str(e)}")
            
if __name__ == "__main__":
    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Запуск бота
    bot = TradingBot(test_mode=True)
    bot.run() 