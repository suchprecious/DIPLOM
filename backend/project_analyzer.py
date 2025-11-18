import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pymorphy2

class ProjectAnalyzer:
    def __init__(self):
        self.morph = pymorphy2.MorphAnalyzer()
        
        # Словари для классификации
        self.keywords = {
            'project_type': {
                'mobile': ['мобильное', 'приложение', 'android', 'ios', 'смартфон', 'телефон'],
                'web': ['веб', 'сайт', 'браузер', 'интернет', 'онлайн'],
                'desktop': ['десктоп', 'компьютер', 'программа', 'установка'],
                'data_science': ['анализ', 'данные', 'ml', 'ai', 'нейросеть', 'машинное'],
                'iot': ['iot', 'интернет вещей', 'сенсор', 'устройство'],
                'game': ['игра', 'гейм', 'игровой', 'unity', 'unreal'],
                'blockchain': ['блокчейн', 'крипто', 'nft', 'смарт-контракт'],
                'ai_ml': ['ии', 'искусственный интеллект', 'машинное обучение']
            },
            'scale': {
                'startup': ['стартап', 'мвп', 'гипотеза', 'быстрый выход'],
                'smb': ['малый бизнес', 'средний бизнес', 'компания'],
                'enterprise': ['корпоратив', 'предприятие', 'крупный', 'глобальный']
            },
            'domain': {
                'ecommerce': ['интернет-магазин', 'продажа', 'товар', 'каталог'],
                'finance': ['финанс', 'банк', 'платеж', 'инвестиц', 'крипто'],
                'social': ['социальн', 'чат', 'мессенджер', 'сообщество'],
                'education': ['образован', 'обучен', 'курс', 'учеба'],
                'healthcare': ['здоровье', 'медицин', 'врач', 'диагност'],
                'transport': ['транспорт', 'доставк', 'такси', 'логистик']
            },
            'budget': {
                'low': ['бюджет ограничен', 'недорог', 'эконом', 'дешев'],
                'medium': ['средний бюджет', 'стандартн', 'оптимальн'],
                'high': ['высокий бюджет', 'премиум', 'масштабн', 'дорог']
            }
        }
    
    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        tokens = word_tokenize(text, language='russian')
        tokens = [token for token in tokens if token not in stopwords.words('russian')]
        tokens = [self.morph.parse(token)[0].normal_form for token in tokens]
        return tokens
    
    def extract_characteristics(self, description):
        tokens = self.preprocess_text(description)
        
        characteristics = {
            'project_type': self._classify_category(tokens, 'project_type'),
            'scale': self._classify_category(tokens, 'scale'),
            'domain': self._classify_category(tokens, 'domain'),
            'budget': self._classify_category(tokens, 'budget'),
            'complexity': self._estimate_complexity(tokens, description),
            'features': self._extract_features(tokens)
        }
        
        return characteristics
    
    def _classify_category(self, tokens, category):
        scores = {}
        
        for subcategory, keywords in self.keywords[category].items():
            score = sum(1 for token in tokens if any(keyword in token for keyword in keywords))
            scores[subcategory] = score
        
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]
        return 'unknown'
    
    def _estimate_complexity(self, tokens, description):
        complexity_indicators = [
            'сложн', 'масштаб', 'больш', 'глобаль', 'распределен',
            'интеграц', 'микросервис', 'высоконагружен', 'real-time'
        ]
        
        score = sum(1 for token in tokens if any(indicator in token for indicator in complexity_indicators))
        
        if score >= 3:
            return 'complex'
        elif score >= 1:
            return 'medium'
        else:
            return 'simple'
    
    def _extract_features(self, tokens):
        features = []
        feature_keywords = {
            'realtime': ['realtime', 'real-time', 'онлайн', 'мгновен'],
            'payments': ['платеж', 'оплат', 'деньг', 'банк'],
            'geolocation': ['геолокац', 'карт', 'gps', 'локац'],
            'ai': ['ии', 'ai', 'нейросет', 'машинное обучение'],
            'mobile': ['мобильн', 'телефон', 'android', 'ios']
        }
        
        for feature, keywords in feature_keywords.items():
            if any(any(keyword in token for keyword in keywords) for token in tokens):
                features.append(feature)
        
        return features

# Тестирование NLP-модуля
print("\nТестирование NLP-модуля")
analyzer = ProjectAnalyzer()

# Тестовые примеры
test_descriptions = [
    "Нужно создать мобильное приложение для такси с геолокацией и онлайн-оплатой для стартапа",
    "Разработать корпоративную веб-платформу для банка с интеграцией платежных систем",
    "Создать MVP социальной сети для общения с реальным временем доставки сообщений"
]

for i, desc in enumerate(test_descriptions, 1):
    print(f"\nТЕст{i}")
    print(f"Описание: {desc}")
    characteristics = analyzer.extract_characteristics(desc)
    print(f"Извлеченные характеристики: {characteristics}")