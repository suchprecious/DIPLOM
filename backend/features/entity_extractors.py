from natasha import (
    Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger,
    NewsSyntaxParser, NewsNERTagger, NamesExtractor, 
    AddrExtractor, DatesExtractor
)
from typing import List, Dict

class EntityExtractor:
    def __init__(self):
        # Инициализация компонентов Natasha
        self.segmenter = Segmenter()
        self.morph_vocab = MorphVocab()
        self.emb = NewsEmbedding()
        
        # Инициализация теггеров и парсеров
        self.morph_tagger = NewsMorphTagger(self.emb)
        self.syntax_parser = NewsSyntaxParser(self.emb)
        self.ner_tagger = NewsNERTagger(self.emb)
        
        # Инициализация экстракторов сущностей с передачей морфологического словаря
        self.names_extractor = NamesExtractor(self.morph_vocab)
        self.addr_extractor = AddrExtractor(self.morph_vocab)
        self.dates_extractor = DatesExtractor(self.morph_vocab)
    
    def extract_entities(self, text: str) -> Dict[str, List]:
        """Извлечение сущностей из текста"""
        try:
            # Создаем документ
            from natasha import Doc
            doc = Doc(text)
            
            # Сегментация
            doc.segment(self.segmenter)
            
            # Морфологический разбор
            doc.tag_morph(self.morph_tagger)
            
            # Извлечение сущностей
            doc.tag_ner(self.ner_tagger)
            
            # Нормализация сущностей
            for span in doc.spans:
                span.normalize(self.morph_vocab)
            
            # Извлечение именованных сущностей
            entities = {
                'names': [],
                'locations': [],
                'organizations': [],
                'dates': []
            }
            
            for span in doc.spans:
                if span.type == 'PER':  # Person
                    entities['names'].append(span.normal)
                elif span.type == 'LOC':  # Location
                    entities['locations'].append(span.normal)
                elif span.type == 'ORG':  # Organization
                    entities['organizations'].append(span.normal)
            
            # Дополнительно извлекаем даты
            dates = self.dates_extractor(text)
            entities['dates'] = [_.fact.as_dict for _ in dates]
            
            return entities
            
        except Exception as e:
            print(f"Ошибка при извлечении сущностей: {e}")
            return {
                'names': [],
                'locations': [], 
                'organizations': [],
                'dates': []
            }