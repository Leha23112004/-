import logging
from datetime import datetime

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_trader.log', mode='w'),
        logging.StreamHandler()
    ]
)

class TestTrader:
    def __init__(self):
        """
        Инициализация тестового трейдера
        """
        self.positions = {}  # Словарь для хранения открытых позиций
        self.balance = 10000  # Начальный баланс в USDT
        self.trade_history = []  # История сделок
        self.logger = logging.getLogger(__name__)
        logging.info("TestTrader успешно инициализирован")

    def get_balance(self):
        """
        Получение текущего баланса
        
        :return: Текущий баланс в USDT
        """
        return self.balance

    def execute_trade(self, symbol, side, amount, price):
        """
        Исполнение тестовой сделки
        
        :param symbol: Торговая пара
        :param side: Сторона сделки ('buy' или 'sell')
        :param amount: Количество
        :param price: Цена
        :return: Результат сделки
        """
        try:
            # Расчет стоимости сделки
            cost = amount * price
            
            if side == 'buy':
                # Проверка достаточности средств
                if cost > self.balance:
                    self.logger.warning(f"Недостаточно средств для покупки {symbol}")
                    return False
                    
                # Обновление баланса и позиции
                self.balance -= cost
                if symbol not in self.positions:
                    self.positions[symbol] = 0
                self.positions[symbol] += amount
                
            elif side == 'sell':
                # Проверка наличия позиции
                if symbol not in self.positions or self.positions[symbol] < amount:
                    self.logger.warning(f"Недостаточно {symbol} для продажи")
                    return False
                    
                # Обновление баланса и позиции
                self.balance += cost
                self.positions[symbol] -= amount
                
            # Запись в историю
            trade = {
                'timestamp': datetime.now(),
                'symbol': symbol,
                'side': side,
                'amount': amount,
                'price': price,
                'cost': cost,
                'balance': self.balance
            }
            self.trade_history.append(trade)
            
            self.logger.info(f"Исполнена сделка: {side} {amount} {symbol} по цене {price}")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка при исполнении сделки: {str(e)}")
            return False

    def get_position(self, symbol):
        """
        Получение текущей позиции по символу
        
        :param symbol: Торговая пара
        :return: Размер позиции
        """
        return self.positions.get(symbol, 0)

    def get_trade_history(self):
        """
        Получение истории сделок
        
        :return: Список сделок
        """
        return self.trade_history

def main():
    """Основная функция для тестирования"""
    try:
        trader = TestTrader()
        
        # Проверка подключения
        if trader.test_connection():
            # Проверка получения данных
            trader.test_market_data()
            
    except Exception as e:
        logging.error(f"Ошибка в main: {str(e)}")

if __name__ == "__main__":
    main() 