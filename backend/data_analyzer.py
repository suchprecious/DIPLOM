import pandas as pd
import numpy as np
import json
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns

class DataAnalyzer:
    def __init__(self, dataset_path='C:/Users/sleek/Desktop/folders/diplom/data/advanced_tech_stack_dataset_50k.csv'):
        self.dataset = pd.read_csv(dataset_path)

        self.dataset['tech_stack'] = self.dataset['tech_stack'].apply(
            lambda x: json.loads(x.replace("'", "\"")) if isinstance(x, str) else x
        )
    
    def basic_statistics(self):
        print("БАЗОВАЯ СТАТИСТИКА ДАТАСЕТА")
        print(f"Всего записей: {len(self.dataset)}")
        print(f"Колонки: {list(self.dataset.columns)}")
        

        categorical_cols = ['project_type', 'scale', 'domain', 'budget', 'complexity', 'description_style']
        for col in categorical_cols:
            print(f"\nРаспределение по {col}")
            print(self.dataset[col].value_counts())
    
    def analyze_tech_stacks(self):
        print("\nАНАЛИЗ ТЕХНОЛОГИЧЕСКИХ СТЕКОВ")
        

        all_tech = [tech for stack in self.dataset['tech_stack'] for tech in stack]
        tech_counter = Counter(all_tech)
        
        print(f"Всего уникальных технологий: {len(tech_counter)}")
        print("\nТоп-20 самых популярных технологий:")
        for tech, count in tech_counter.most_common(20):
            print(f"  {tech}: {count} раз")
        

        print("\nПопулярные технологии по типам проектов")
        for project_type in self.dataset['project_type'].unique():
            type_tech = [
                tech for stack in self.dataset[self.dataset['project_type'] == project_type]['tech_stack'] 
                for tech in stack
            ]
            top_tech = Counter(type_tech).most_common(5)
            print(f"\n{project_type}: {[tech for tech, count in top_tech]}")
    
    def analyze_text_descriptions(self):
        print("\nАНАЛИЗ ТЕКСТОВЫХ ОПИСАНИЙ")
        
        descriptions = self.dataset['project_description']
        

        desc_lengths = descriptions.str.len()
        word_counts = descriptions.str.split().str.len()
        
        print(f"Средняя длина описания: {desc_lengths.mean():.1f} символов")
        print(f"Среднее количество слов: {word_counts.mean():.1f}")
        print(f"Минимальная длина: {desc_lengths.min()} символов")
        print(f"Максимальная длина: {desc_lengths.max()} символов")
        

        print("\nПримеры описаний по стилям")
        for style in self.dataset['description_style'].unique()[:3]:
            sample = self.dataset[self.dataset['description_style'] == style].iloc[0]
            print(f"\nСтиль: {style}")
            print(f"Описание: {sample['project_description']}")
    
    def find_patterns(self):
        print("\nАНАЛИЗ ПАТТЕРНОВ")
        
        print("Типичные стеки для разных доменов")
        for domain in self.dataset['domain'].unique()[:3]:
            domain_data = self.dataset[self.dataset['domain'] == domain]
            domain_tech = [tech for stack in domain_data['tech_stack'] for tech in stack]
            top_domain_tech = Counter(domain_tech).most_common(5)
            print(f"\n{domain}: {[tech for tech, count in top_domain_tech]}")
        
        print("\nВлияние масштаба на выбор технологий")
        for scale in self.dataset['scale'].unique():
            scale_data = self.dataset[self.dataset['scale'] == scale]
            scale_tech = [tech for stack in scale_data['tech_stack'] for tech in stack]
            top_scale_tech = Counter(scale_tech).most_common(3)
            print(f"{scale}: {[tech for tech, count in top_scale_tech]}")

print("Запуск анализа датасета")
analyzer = DataAnalyzer()
analyzer.basic_statistics()
analyzer.analyze_tech_stacks()
analyzer.analyze_text_descriptions()
analyzer.find_patterns()