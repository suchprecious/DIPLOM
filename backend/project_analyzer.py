from natasha import (
    Segmenter, MorphVocab, 
    NewsEmbedding, NewsMorphTagger, NewsSyntaxParser, NewsNERTagger,
    Doc
)
import re
from collections import Counter

class NatashaProjectAnalyzer:
    def __init__(self):
        self.segmenter = Segmenter()
        self.morph_vocab = MorphVocab()
        self.emb = NewsEmbedding()
        self.morph_tagger = NewsMorphTagger(self.emb)
        self.syntax_parser = NewsSyntaxParser(self.emb)
        self.ner_tagger = NewsNERTagger(self.emb)
        

        self.keywords = {
            'project_type': {
                'mobile': ['мобильный', 'приложение', 'android', 'ios', 'смартфон', 'телефон', 'мобайл'],
                'web': ['веб', 'сайт', 'браузер', 'интернет', 'онлайн', 'веб-сайт', 'веб-приложение'],
                'desktop': ['десктоп', 'компьютер', 'программа', 'установка', 'настольный', 'клиент'],
                'data_science': ['анализ', 'данные', 'ml', 'ai', 'нейросеть', 'машинный', 'аналитика'],
                'iot': ['iot', 'интернет вещей', 'сенсор', 'устройство', 'датчик', 'умный дом'],
                'game': ['игра', 'гейм', 'игровой', 'unity', 'unreal', 'игропром'],
                'blockchain': ['блокчейн', 'крипто', 'nft', 'смарт-контракт', 'децентрализованный'],
                'ai_ml': ['ии', 'искусственный', 'интеллект', 'машинный', 'обучение', 'нейронный']
            },
            'scale': {
                'startup': ['стартап', 'мвп', 'гипотеза', 'быстрый', 'выход', 'начальный', 'прототип'],
                'smb': ['малый', 'бизнес', 'средний', 'компания', 'фирма', 'предприятие'],
                'enterprise': ['корпоративный', 'предприятие', 'крупный', 'глобальный', 'холдинг', 'корпорация']
            },
            'domain': {
                'ecommerce': ['магазин', 'продажа', 'товар', 'каталог', 'корзина', 'покупка', 'интернет-магазин'],
                'finance': ['финансы', 'банк', 'платеж', 'инвестиция', 'крипто', 'деньги', 'банковский'],
                'social': ['социальный', 'чат', 'мессенджер', 'сообщество', 'общение', 'сеть'],
                'education': ['образование', 'обучение', 'курс', 'учебный', 'студент', 'образовательный'],
                'healthcare': ['здоровье', 'медицинский', 'врач', 'диагностика', 'лечение', 'медицина'],
                'transport': ['транспорт', 'доставка', 'такси', 'логистика', 'перевозка', 'навигация'],
                'entertainment': ['развлечение', 'видео', 'музыка', 'стриминг', 'игра', 'контент'],
                'real_estate': ['недвижимость', 'аренда', 'квартира', 'дом', 'собственность', 'жилье']
            },
            'budget': {
                'low': ['бюджет', 'ограниченный', 'недорогой', 'экономный', 'дешевый', 'бюджетный', 'минимальный'],
                'medium': ['средний', 'бюджет', 'стандартный', 'оптимальный', 'умеренный', 'доступный'],
                'high': ['высокий', 'бюджет', 'премиум', 'масштабный', 'дорогой', 'крупный', 'серьезный']
            }
        }
        

        self.stop_words = set([
            'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 
            'то', 'все', 'она', 'так', 'его', 'но', 'да', 'вы', 'за', 'бы', 'по', 
            'только', 'ее', 'мне', 'было', 'вот', 'от', 'меня', 'еще', 'нет', 'о',
            'из', 'ему', 'теперь', 'когда', 'даже', 'ну', 'ли', 'если', 'уже', 'или',
            'для', 'мы', 'тебя', 'их', 'чем', 'была', 'сам', 'чтоб', 'без', 'будто',
            'чего', 'раз', 'тоже', 'себя', 'ним', 'здесь', 'этот', 'того', 'тем'
        ])

    def preprocess_text(self, text):
        if not isinstance(text, str) or not text.strip():
            return []
            

        doc = Doc(text)
        

        doc.segment(self.segmenter)
        

        doc.tag_morph(self.morph_tagger)
        

        for token in doc.tokens:
            token.lemmatize(self.morph_vocab)
        

        lemmas = []
        for token in doc.tokens:
            lemma = token.lemma.lower()
            if (lemma not in self.stop_words and 
                len(lemma) > 2 and 
                not lemma.isdigit() and
                not re.match(r'^[а-яё]{1,2}$', lemma)):
                lemmas.append(lemma)
        
        return lemmas

    def extract_characteristics(self, description):

        try:
            lemmas = self.preprocess_text(description)
            
            characteristics = {
                'project_type': self._classify_with_confidence(lemmas, 'project_type'),
                'scale': self._classify_with_confidence(lemmas, 'scale'),
                'domain': self._classify_with_confidence(lemmas, 'domain'),
                'budget': self._classify_with_confidence(lemmas, 'budget'),
                'complexity': self._estimate_complexity(lemmas, description),
                'features': self._extract_features(lemmas),
                'technical_requirements': self._extract_technical_requirements(lemmas),
                'confidence_score': self._calculate_overall_confidence(lemmas)
            }
            
            return characteristics
            
        except Exception as e:
            print(f"Ошибка при анализе текста: {e}")
            return self._get_default_characteristics()

    def _classify_with_confidence(self, lemmas, category):

        scores = {}
        
        for subcategory, keywords in self.keywords[category].items():
            score = 0
            for lemma in lemmas:
                for keyword in keywords:
                    if lemma == keyword.lower() or keyword.lower() in lemma:
                        score += 2 
                    elif lemma in keyword.lower():
                        score += 1
            scores[subcategory] = score
        
        if not any(scores.values()):
            return {'category': 'unknown', 'confidence': 0.0}
        
        best_category = max(scores.items(), key=lambda x: x[1])
        total_score = sum(scores.values())
        confidence = best_category[1] / total_score if total_score > 0 else 0.0
        
        return {
            'category': best_category[0],
            'confidence': round(confidence, 2),
            'score': best_category[1]
        }

    def _estimate_complexity(self, lemmas, description):
        complexity_indicators = [
            'сложный', 'масштабный', 'большой', 'глобальный', 'распределенный',
            'интеграция', 'микросервис', 'высоконагруженный', 'real-time',
            'корпоративный', 'предприятие', 'многопользовательский', 'высокий',
            'комплексный', 'интегрированный', 'автоматизированный', 'оптимизированный'
        ]
        
        simple_indicators = [
            'простой', 'легкий', 'базовый', 'начальный', 'минимальный',
            'визитка', 'лендинг', 'одностраничный'
        ]
        
        complexity_score = sum(1 for lemma in lemmas if any(indicator in lemma for indicator in complexity_indicators))
        simplicity_score = sum(1 for lemma in lemmas if any(indicator in lemma for indicator in simple_indicators))
        
        net_score = complexity_score - simplicity_score
        
        if net_score >= 3:
            return {'level': 'complex', 'score': net_score}
        elif net_score >= 1:
            return {'level': 'medium', 'score': net_score}
        elif net_score <= -2:
            return {'level': 'simple', 'score': net_score}
        else:
            return {'level': 'medium', 'score': net_score}

    def _extract_features(self, lemmas):
        features = []
        feature_keywords = {
            'realtime': ['realtime', 'real-time', 'онлайн', 'мгновенный', 'живой', 'синхронный'],
            'payments': ['платеж', 'оплата', 'деньги', 'банк', 'карта', 'транзакция'],
            'geolocation': ['геолокация', 'карта', 'gps', 'локация', 'навигация', 'координаты'],
            'ai_ml': ['ии', 'ai', 'нейросеть', 'машинный', 'интеллект', 'обучение'],
            'mobile': ['мобильный', 'телефон', 'android', 'ios', 'смартфон', 'мобайл'],
            'analytics': ['аналитика', 'отчет', 'статистика', 'метрика', 'дашборд'],
            'cloud': ['облачный', 'cloud', 'aws', 'azure', 'google cloud', 'облако'],
            'security': ['безопасность', 'защита', 'шифрование', 'аутентификация', 'безопасный'],
            'database': ['база', 'данные', 'sql', 'nosql', 'хранилище'],
            'api': ['api', 'интерфейс', 'интеграция', 'rest', 'graphql']
        }
        
        for feature, keywords in feature_keywords.items():
            for lemma in lemmas:
                if any(keyword in lemma for keyword in keywords):
                    features.append(feature)
                    break
        
        return features

    def _extract_technical_requirements(self, lemmas):
        requirements = {
            'performance': self._check_requirement(lemmas, ['производительность', 'скорость', 'быстрый', 'оптимизация']),
            'scalability': self._check_requirement(lemmas, ['масштабируемость', 'масштабирование', 'рост']),
            'security': self._check_requirement(lemmas, ['безопасность', 'защита', 'шифрование']),
            'reliability': self._check_requirement(lemmas, ['надежность', 'стабильность', 'отказоустойчивость']),
            'maintenance': self._check_requirement(lemmas, ['поддержка', 'обслуживание', 'мониторинг'])
        }
        
        return {k: v for k, v in requirements.items() if v}

    def _check_requirement(self, lemmas, keywords):
        return any(any(keyword in lemma for keyword in keywords) for lemma in lemmas)

    def _calculate_overall_confidence(self, lemmas):
        if not lemmas:
            return 0.0
        
        total_matches = 0
        total_possible = sum(len(keywords) for category in self.keywords.values() for keywords in category.values())
        
        for category in self.keywords.values():
            for keywords in category.values():
                if any(any(keyword in lemma for keyword in keywords) for lemma in lemmas):
                    total_matches += 1
        
        confidence = min(total_matches / len(self.keywords) if lemmas else 0, 1.0)
        return round(confidence, 2)

    def _get_default_characteristics(self):
        return {
            'project_type': {'category': 'web', 'confidence': 0.0},
            'scale': {'category': 'startup', 'confidence': 0.0},
            'domain': {'category': 'ecommerce', 'confidence': 0.0},
            'budget': {'category': 'medium', 'confidence': 0.0},
            'complexity': {'level': 'medium', 'score': 0},
            'features': [],
            'technical_requirements': {},
            'confidence_score': 0.0
        }

print("запуск наташи")
analyzer = NatashaProjectAnalyzer()


test_descriptions = [
    "Нужно создать мобильное приложение для такси с геолокацией и онлайн-оплатой для стартапа",
    "Разработать корпоративную веб-платформу для банка с интеграцией платежных систем и высокой безопасностью",
    "Создать MVP социальной сети для общения с реальным временем доставки сообщений",
    "Простой сайт визитка для малого бизнеса с базовой информацией",
    "Сложная система анализа данных с искусственным интеллектом для крупного предприятия с машинным обучением"
]

print("=" * 80)
for i, desc in enumerate(test_descriptions, 1):
    print(f"\nТест {i}")
    print(f"Описание: {desc}")
    
    characteristics = analyzer.extract_characteristics(desc)
    
    print(f"Извлеченные характеристики:")
    print(f"Тип проекта: {characteristics['project_type']['category']} (уверенность: {characteristics['project_type']['confidence']})")
    print(f"Масштаб: {characteristics['scale']['category']} (уверенность: {characteristics['scale']['confidence']})")
    print(f"Домен: {characteristics['domain']['category']} (уверенность: {characteristics['domain']['confidence']})")
    print(f"Бюджет: {characteristics['budget']['category']} (уверенность: {characteristics['budget']['confidence']})")
    print(f"Сложность: {characteristics['complexity']['level']} (score: {characteristics['complexity']['score']})")
    print(f"Особенности: {characteristics['features']}")
    print(f"Технические требования: {list(characteristics['technical_requirements'].keys())}")
    print(f"Общая уверенность: {characteristics['confidence_score']}")
    

    lemmas = analyzer.preprocess_text(desc)
    print(f"Леммы: {lemmas[:15]}")

print("\n" + "=" * 80)
