from natasha import (
    Segmenter, MorphVocab, NewsEmbedding, NewsMorphTagger,
    NewsSyntaxParser, NewsNERTagger, NamesExtractor, 
    AddrExtractor, DatesExtractor
)
from typing import List, Dict

class EntityExtractor:
    def __init__(self):
        self.segmenter = Segmenter()
        self.morph_vocab = MorphVocab()
        self.emb = NewsEmbedding()
        

        self.morph_tagger = NewsMorphTagger(self.emb)
        self.syntax_parser = NewsSyntaxParser(self.emb)
        self.ner_tagger = NewsNERTagger(self.emb)
        
        self.names_extractor = NamesExtractor(self.morph_vocab)
        self.addr_extractor = AddrExtractor(self.morph_vocab)
        self.dates_extractor = DatesExtractor(self.morph_vocab)
    
    def extract_entities(self, text: str) -> Dict[str, List]:
        try:
            from natasha import Doc
            doc = Doc(text)
            
            doc.segment(self.segmenter)
            
            doc.tag_morph(self.morph_tagger)
            
            doc.tag_ner(self.ner_tagger)

            for span in doc.spans:
                span.normalize(self.morph_vocab)

            entities = {
                'names': [],
                'locations': [],
                'organizations': [],
                'dates': []
            }
            
            for span in doc.spans:
                if span.type == 'PER': 
                    entities['names'].append(span.normal)
                elif span.type == 'LOC':  
                    entities['locations'].append(span.normal)
                elif span.type == 'ORG':  
                    entities['organizations'].append(span.normal)
            

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