import requests
import logging
import json
from datetime import datetime, timedelta
import pandas as pd
from textblob import TextBlob
import time
import os
from bs4 import BeautifulSoup
import re
import random

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sentiment_analysis.log'),
        logging.StreamHandler()
    ]
)

class SentimentAnalyzer:
    def __init__(self, data_collector):
        """
        Инициализация анализатора настроений
        
        Args:
            data_collector (DataCollector): Объект для сбора данных
        """
        self.logger = logging.getLogger(__name__)
        self.data_collector = data_collector
        self.logger.info("SentimentAnalyzer успешно инициализирован")
        self.fear_greed_url = "https://api.alternative.me/fng/"
        self.cryptopanic_url = "https://cryptopanic.com/api/v1/posts/"
        self.cointelegraph_url = "https://cointelegraph.com/tags/bitcoin"
        
    def get_fear_greed_index(self):
        """
        Получение индекса страха и жадности
        
        :return: Словарь с данными индекса
        """
        try:
            response = requests.get(self.fear_greed_url)
            if response.status_code == 200:
                data = response.json()
                if data['data']:
                    latest = data['data'][0]
                    return {
                        'value': int(latest['value']),
                        'value_classification': latest['value_classification'],
                        'timestamp': latest['timestamp'],
                        'time_until_update': latest['time_until_update']
                    }
            return None
        except Exception as e:
            self.logger.error(f"Ошибка при получении индекса страха и жадности: {str(e)}")
            return None
            
    def get_crypto_news(self, limit=10):
        """
        Получение новостей с CryptoPanic
        
        :param limit: Количество новостей
        :return: Список новостей
        """
        try:
            params = {
                'auth_token': '',  # Можно оставить пустым для публичного API
                'public': 'true',
                'limit': limit,
                'currencies': 'BTC,ETH,BNB',  # Добавляем интересующие нас криптовалюты
                'kind': 'news'  # Только новости
            }
            response = requests.get(self.cryptopanic_url, params=params)
            if response.status_code == 200:
                data = response.json()
                if 'results' in data:
                    news = []
                    for item in data['results']:
                        if 'title' in item:
                            news.append({
                                'title': item['title'],
                                'url': item.get('url', ''),
                                'source': item.get('source', {}).get('title', 'Unknown'),
                                'published_at': item.get('published_at', '')
                            })
                            self.logger.info(f"Найдена новость: {item['title']}")
                    return news
                else:
                    self.logger.error("В ответе CryptoPanic нет поля 'results'")
            else:
                self.logger.error(f"Ошибка при получении новостей с CryptoPanic: {response.status_code}")
            return []
        except Exception as e:
            self.logger.error(f"Ошибка при получении новостей: {str(e)}")
            return []
            
    def get_cointelegraph_news(self):
        """
        Парсинг новостей с Cointelegraph
        
        :return: Список новостей
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(self.cointelegraph_url, headers=headers)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                news = []
                # Ищем все статьи на странице
                articles = soup.find_all('article')
                for article in articles:
                    # Ищем заголовок
                    title_elem = article.find(['h2', 'h3', 'span'], class_=['post-card__title', 'post__title'])
                    if title_elem:
                        title = title_elem.text.strip()
                        # Ищем ссылку
                        link_elem = article.find('a')
                        if link_elem and 'href' in link_elem.attrs:
                            url = link_elem['href']
                            if not url.startswith('http'):
                                url = 'https://cointelegraph.com' + url
                            news.append({
                                'title': title,
                                'url': url
                            })
                            self.logger.info(f"Найдена новость: {title}")
                return news
            else:
                self.logger.error(f"Ошибка при получении страницы Cointelegraph: {response.status_code}")
            return []
        except Exception as e:
            self.logger.error(f"Ошибка при парсинге Cointelegraph: {str(e)}")
            return []
            
    def analyze_sentiment(self, text):
        """
        Анализ сентимента текста
        
        :param text: Текст для анализа
        :return: Словарь с результатами анализа
        """
        try:
            analysis = TextBlob(text)
            return {
                'polarity': analysis.sentiment.polarity,  # от -1 до 1
                'subjectivity': analysis.sentiment.subjectivity,  # от 0 до 1
                'sentiment': 'positive' if analysis.sentiment.polarity > 0 else 'negative' if analysis.sentiment.polarity < 0 else 'neutral'
            }
        except Exception as e:
            self.logger.error(f"Ошибка при анализе сентимента: {str(e)}")
            return None
            
    def get_market_sentiment(self):
        """
        Получение общего сентимента рынка
        
        :return: Словарь с результатами анализа
        """
        try:
            # Получение индекса страха и жадности
            fear_greed = self.get_fear_greed_index()
            
            # Получение новостей
            crypto_news = self.get_crypto_news()
            cointelegraph_news = self.get_cointelegraph_news()
            
            # Анализ сентимента новостей
            news_sentiment = []
            for news in crypto_news + cointelegraph_news:
                if 'title' in news:
                    sentiment = self.analyze_sentiment(news['title'])
                    if sentiment:
                        news_sentiment.append(sentiment)
            
            # Расчет среднего сентимента
            if news_sentiment:
                avg_polarity = sum(s['polarity'] for s in news_sentiment) / len(news_sentiment)
                avg_subjectivity = sum(s['subjectivity'] for s in news_sentiment) / len(news_sentiment)
            else:
                avg_polarity = 0
                avg_subjectivity = 0
                
            return {
                'fear_greed_index': fear_greed,
                'news_sentiment': {
                    'average_polarity': avg_polarity,
                    'average_subjectivity': avg_subjectivity,
                    'news_count': len(news_sentiment)
                },
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка при получении сентимента рынка: {str(e)}")
            return None
            
    def save_sentiment_data(self, data, filename='sentiment_data.json'):
        """
        Сохранение данных сентимента в файл
        
        :param data: Данные для сохранения
        :param filename: Имя файла
        """
        try:
            if os.path.exists(filename):
                with open(filename, 'r') as f:
                    existing_data = json.load(f)
            else:
                existing_data = []
                
            existing_data.append(data)
            
            with open(filename, 'w') as f:
                json.dump(existing_data, f, indent=4)
                
        except Exception as e:
            self.logger.error(f"Ошибка при сохранении данных: {str(e)}")
            
    def get_google_news(self, query="crypto", limit=10):
        """
        Получение новостей через Google News по ключевому слову
        :param query: поисковый запрос
        :param limit: количество новостей
        :return: список новостей
        """
        try:
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            url = f'https://news.google.com/search?q={query}%20when:7d&hl=ru&gl=RU&ceid=RU:ru'
            response = requests.get(url, headers=headers)
            news = []
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                articles = soup.find_all('article')
                for article in articles:
                    # Сначала ищем ссылку с классом DY5T1d
                    title_elem = article.find('a', class_='DY5T1d')
                    if not title_elem:
                        # Если нет, берем первую ссылку внутри article
                        title_elem = article.find('a')
                    if title_elem and title_elem.text.strip():
                        title = title_elem.text.strip()
                        link = title_elem['href']
                        if not link.startswith('http'):
                            link = 'https://news.google.com' + link[1:]
                        news.append({'title': title, 'url': link})
                        self.logger.info(f"Google News: {title}")
                        if len(news) >= limit:
                            break
            else:
                self.logger.error(f"Ошибка при получении Google News: {response.status_code}")
            return news
        except Exception as e:
            self.logger.error(f"Ошибка при парсинге Google News: {str(e)}")
            return []

    def get_all_news(self, limit=10):
        """
        Получение всех новостей с разных источников
        """
        news = []
        # Google News по ключевым словам
        for keyword in ["bitcoin", "ethereum", "crypto"]:
            news += self.get_google_news(query=keyword, limit=limit//3)
        return news

    def run_analysis(self):
        """
        Запуск анализа сентимента
        """
        self.logger.info("Запуск анализа сентимента...")
        
        while True:
            try:
                self.logger.info("Начало нового анализа...")
                
                # Получение индекса страха и жадности
                self.logger.info("Получение индекса страха и жадности...")
                fear_greed = self.get_fear_greed_index()
                if fear_greed:
                    self.logger.info(f"Индекс страха и жадности: {fear_greed['value']} ({fear_greed['value_classification']})")
                
                # Получение новостей с Google News
                self.logger.info("Получение новостей с Google News...")
                all_news = self.get_all_news(limit=12)
                self.logger.info(f"Получено {len(all_news)} новостей с Google News")
                
                # Анализ сентимента новостей
                self.logger.info("Анализ сентимента новостей...")
                news_sentiment = []
                for news in all_news:
                    if 'title' in news:
                        sentiment = self.analyze_sentiment(news['title'])
                        if sentiment:
                            news_sentiment.append(sentiment)
                            self.logger.info(f"Новость: {news['title']}")
                            self.logger.info(f"Сентимент: {sentiment['sentiment']} (полярность: {sentiment['polarity']:.2f})")
                
                # Расчет среднего сентимента
                if news_sentiment:
                    avg_polarity = sum(s['polarity'] for s in news_sentiment) / len(news_sentiment)
                    avg_subjectivity = sum(s['subjectivity'] for s in news_sentiment) / len(news_sentiment)
                    self.logger.info(f"Средний сентимент новостей: {avg_polarity:.2f}")
                    self.logger.info(f"Средняя субъективность: {avg_subjectivity:.2f}")
                else:
                    self.logger.warning("Нет новостей для анализа сентимента")
                
                # Сохранение данных
                sentiment_data = {
                    'fear_greed_index': fear_greed,
                    'news_sentiment': {
                        'average_polarity': avg_polarity if news_sentiment else 0,
                        'average_subjectivity': avg_subjectivity if news_sentiment else 0,
                        'news_count': len(news_sentiment)
                    },
                    'timestamp': datetime.now().isoformat()
                }
                
                self.save_sentiment_data(sentiment_data)
                self.logger.info("Данные сохранены в sentiment_data.json")
                
                # Пауза между анализами (1 час)
                self.logger.info("Ожидание 1 час до следующего анализа...")
                time.sleep(3600)
                
            except Exception as e:
                self.logger.error(f"Ошибка в основном цикле: {str(e)}")
                self.logger.info("Повторная попытка через 1 час...")
                time.sleep(3600)

    def analyze_symbol(self, symbol):
        """
        Анализ настроений для торговой пары
        
        Args:
            symbol (str): Торговая пара
            
        Returns:
            dict: Результаты анализа настроений
        """
        try:
            self.logger.info(f"Анализ настроений для {symbol}...")
            
            # В тестовом режиме генерируем случайные данные
            sentiment = random.uniform(-1, 1)
            volume = random.uniform(0.5, 1.5)
            
            # Определение сигнала на основе настроений
            if sentiment > 0.3:
                signal = 'ПОЗИТИВНЫЙ'
            elif sentiment < -0.3:
                signal = 'НЕГАТИВНЫЙ'
            else:
                signal = 'НЕЙТРАЛЬНЫЙ'
                
            self.logger.info(f"Анализ настроений для {symbol} завершен")
            
            return {
                'sentiment': sentiment,
                'volume': volume,
                'signal': signal,
                'confidence': abs(sentiment)  # Уверенность пропорциональна силе настроений
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка при анализе настроений: {str(e)}")
            return None

if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    analyzer.run_analysis() 