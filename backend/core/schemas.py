from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import List, Optional, Dict, Any, Union
from enum import Enum
from datetime import datetime


class ProjectScale(str, Enum):
    STARTUP = "startup"
    SMB = "smb"
    ENTERPRISE = "enterprise"
    ENTERPRISE_GLOBAL = "enterprise_global"

class Industry(str, Enum):
    FINANCE = "finance"
    HEALTHCARE = "healthcare"
    TRANSPORT = "transport"
    REAL_ESTATE = "real_estate"
    ENTERTAINMENT = "entertainment"
    SOCIAL = "social"
    EDUCATION = "education"
    RETAIL = "retail"
    MANUFACTURING = "manufacturing"

class ProjectType(str, Enum):
    WEB = "web"
    MOBILE = "mobile"
    DESKTOP = "desktop"
    DATA_SCIENCE = "data_science"
    AI_ML = "ai_ml"
    IOT = "iot"
    BLOCKCHAIN = "blockchain"
    GAME = "game"

class TeamSize(str, Enum):
    SOLO = "solo"
    SMALL_TEAM = "small_team"
    MEDIUM_TEAM = "medium_team"
    LARGE_TEAM = "large_team"
    DISTRIBUTED = "distributed"

class Complexity(str, Enum):
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    HIGHLY_COMPLEX = "highly_complex"

class BudgetLevel(str, Enum):
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    UNLIMITED = "unlimited"

class PerformanceRequirement(str, Enum):
    LOW_LOAD = "low_load"
    MEDIUM_LOAD = "medium_load"
    HIGH_LOAD = "high_load"
    EXTREME_LOAD = "extreme_load"

class ScalabilityRequirement(str, Enum):
    NONE = "none"
    VERTICAL = "vertical"
    HORIZONTAL = "horizontal"
    AUTO = "auto"

class SecurityRequirement(str, Enum):
    BASIC = "basic"
    STANDARD = "standard"
    HIGH = "high"
    ENTERPRISE_GRADE = "enterprise_grade"

class RealtimeRequirement(str, Enum):
    NONE = "none"
    BASIC = "basic"
    ADVANCED = "advanced"
    CRITICAL = "critical"

class IntegrationRequirement(str, Enum):
    NONE = "none"
    FEW_APIS = "few_apis"
    MULTIPLE_APIS = "multiple_apis"
    LEGACY_SYSTEMS = "legacy_systems"

class ProjectFormat(str, Enum):
    STARTUP_PITCH = "startup_pitch"
    FORMAL_BUSINESS = "formal_business"
    USER_STORY = "user_story"
    TECHNICAL_SPEC = "technical_spec"
    PROBLEM_SOLUTION = "problem_solution"

class TechnologyCategory(str, Enum):
    BACKEND = "backend"
    FRONTEND = "frontend"
    DATABASE = "database"
    DEVOPS = "devops"
    MOBILE = "mobile"
    DATA_SCIENCE = "data_science"
    AI_ML = "ai_ml"
    CLOUD = "cloud"
    TESTING = "testing"
    MONITORING = "monitoring"

class TechnicalRequirements(BaseModel):
    performance: PerformanceRequirement
    scalability: ScalabilityRequirement
    security: SecurityRequirement
    realtime: RealtimeRequirement
    integration: IntegrationRequirement

    @field_validator('performance')
    @classmethod
    def validate_performance(cls, v: str) -> str:
        if v not in [item.value for item in PerformanceRequirement]:
            raise ValueError(f"Invalid performance requirement: {v}")
        return v

    @classmethod
    def from_dict(cls, data: Dict[str, str]) -> 'TechnicalRequirements':
        return cls(
            performance=data.get('performance', 'low_load'),
            scalability=data.get('scalability', 'none'),
            security=data.get('security', 'basic'),
            realtime=data.get('realtime', 'none'),
            integration=data.get('integration', 'none')
        )

    model_config = ConfigDict(extra='forbid')

class ProjectDescription(BaseModel):
    description: str = Field(..., min_length=10, max_length=2000, description="Краткое описание IT-проекта")
    project_type: Optional[ProjectType] = None
    project_scale: Optional[ProjectScale] = None
    industry: Optional[Industry] = None
    budget: Optional[BudgetLevel] = None
    team_size: Optional[TeamSize] = None
    complexity: Optional[Complexity] = None
    technical_requirements: Optional[TechnicalRequirements] = None
    business_requirements: Optional[List[str]] = Field(default_factory=list)
    timeline: Optional[str] = None
    subdomain: Optional[str] = None

    @field_validator('description')
    @classmethod
    def validate_description_length(cls, v: str) -> str:
        cleaned = v.strip()
        if len(cleaned) < 10:
            raise ValueError('Description must be at least 10 characters long')
        if len(cleaned) > 2000:
            raise ValueError('Description must not exceed 2000 characters')
        return cleaned

    @field_validator('business_requirements')
    @classmethod
    def validate_business_requirements(cls, v: Optional[List[str]]) -> List[str]:
        if v is None:
            return []
        return [req.strip() for req in v if req.strip()]

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "description": "Разработка масштабируемого мобильного приложения для финансового сектора с использованием AI. Требуется высокая безопасность и интеграция с legacy системами.",
                "project_type": "mobile",
                "project_scale": "enterprise",
                "industry": "finance",
                "budget": "high",
                "team_size": "medium_team",
                "complexity": "complex",
                "technical_requirements": {
                    "performance": "high_load",
                    "scalability": "horizontal",
                    "security": "enterprise_grade",
                    "realtime": "advanced",
                    "integration": "legacy_systems"
                },
                "business_requirements": [
                    "Быстрая окупаемость инвестиций",
                    "Интеграция с существующими системами"
                ],
                "timeline": "6-12 месяцев"
            }
        }
    )

class TechStackPrediction(BaseModel):
    technology: str = Field(..., description="Название технологии")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Уверенность модели в рекомендации")
    category: TechnologyCategory = Field(..., description="Категория технологии")
    reasoning: str = Field(..., description="Обоснование рекомендации")
    alternatives: List[str] = Field(default_factory=list, description="Альтернативные технологии")

    @field_validator('confidence')
    @classmethod
    def validate_confidence(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError('Confidence must be between 0.0 and 1.0')
        return round(v, 4)

    model_config = ConfigDict(extra='forbid')

class PredictionRequest(BaseModel):
    project: ProjectDescription

class PredictionResponse(BaseModel):
    request_id: str = Field(..., description="Уникальный идентификатор запроса")
    recommended_stack: List[TechStackPrediction] = Field(..., description="Рекомендованный стек технологий")
    primary_technologies: List[str] = Field(..., description="Основные технологии стека")
    alternative_recommendations: List[str] = Field(..., description="Альтернативные рекомендации")
    overall_confidence: float = Field(..., ge=0.0, le=1.0, description="Общая уверенность модели")
    processing_time: float = Field(..., description="Время обработки запроса в секундах")
    model_version: str = Field(..., description="Версия модели")
    timestamp: datetime = Field(default_factory=datetime.now, description="Время создания ответа")

    @field_validator('overall_confidence')
    @classmethod
    def validate_overall_confidence(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError('Overall confidence must be between 0.0 and 1.0')
        return round(v, 4)

    @field_validator('processing_time')
    @classmethod
    def validate_processing_time(cls, v: float) -> float:
        if v < 0:
            raise ValueError('Processing time cannot be negative')
        return round(v, 3)

    model_config = ConfigDict(extra='forbid')

class ModelMetadata(BaseModel):
    model_name: str
    model_version: str
    training_date: datetime
    performance_metrics: Dict[str, float]
    feature_importance: Dict[str, float]
    supported_technologies: List[str]
    model_parameters: Dict[str, Any]
    data_statistics: Dict[str, Any]

    @field_validator('performance_metrics')
    @classmethod
    def validate_performance_metrics(cls, v: Dict[str, float]) -> Dict[str, float]:
        for metric, value in v.items():
            if not 0 <= value <= 1:
                raise ValueError(f'Performance metric {metric} must be between 0 and 1')
        return v

    model_config = ConfigDict(extra='forbid')

class TrainingConfig(BaseModel):
    model_type: str = Field("random_forest", description="Тип модели для обучения")
    test_size: float = Field(0.2, ge=0.1, le=0.3, description="Доля тестовой выборки")
    random_state: int = Field(42, description="Seed для воспроизводимости")
    n_estimators: int = Field(100, ge=10, le=1000, description="Количество estimators для ensemble методов")
    max_depth: Optional[int] = Field(None, ge=3, le=20, description="Максимальная глубина дерева")
    learning_rate: Optional[float] = Field(None, ge=0.01, le=1.0, description="Learning rate для boosting методов")
    validation_strategy: str = Field("cross_validation", description="Стратегия валидации")

    @field_validator('test_size')
    @classmethod
    def validate_test_size(cls, v: float) -> float:
        if v <= 0 or v >= 1:
            raise ValueError('Test size must be between 0 and 1')
        return v

    model_config = ConfigDict(extra='forbid')

class DatasetStatistics(BaseModel):
    total_projects: int
    total_technologies: int
    avg_technologies_per_project: float
    most_common_technologies: List[Dict[str, Any]]
    industry_distribution: Dict[str, int]
    project_scale_distribution: Dict[str, int]
    project_type_distribution: Dict[str, int]
    data_quality_metrics: Dict[str, float]
    missing_data_report: Dict[str, int]

    @field_validator('total_projects', 'total_technologies')
    @classmethod
    def validate_positive_counts(cls, v: int) -> int:
        if v < 0:
            raise ValueError('Counts cannot be negative')
        return v

    model_config = ConfigDict(extra='forbid')

class ErrorResponse(BaseModel):
    error: bool = True
    message: str = Field(..., description="Сообщение об ошибке")
    error_code: str = Field(..., description="Код ошибки")
    details: Optional[Dict[str, Any]] = Field(None, description="Детали ошибки")
    timestamp: datetime = Field(default_factory=datetime.now)
    request_id: Optional[str] = Field(None, description="Идентификатор запроса")

    model_config = ConfigDict(extra='forbid')

class HealthCheckResponse(BaseModel):
    status: str = Field(..., description="Статус сервиса")
    model_loaded: bool = Field(..., description="Модель загружена и готова к работе")
    database_connected: bool = Field(..., description="Подключение к базе данных")
    memory_usage_mb: float = Field(..., description="Использование памяти в MB")
    uptime_seconds: float = Field(..., description="Время работы сервиса в секундах")
    active_requests: int = Field(..., description="Количество активных запросов")
    model_status: Dict[str, Any] = Field(..., description="Статус моделей")

    model_config = ConfigDict(extra='forbid')

class BatchPredictionRequest(BaseModel):
    projects: List[ProjectDescription] = Field(..., max_items=100, description="Список проектов для батчевой обработки")
    batch_id: Optional[str] = Field(None, description="Идентификатор батча")

    @field_validator('projects')
    @classmethod
    def validate_projects_count(cls, v: List[ProjectDescription]) -> List[ProjectDescription]:
        if len(v) > 100:
            raise ValueError('Batch processing limited to 100 projects maximum')
        return v

    model_config = ConfigDict(extra='forbid')

class BatchPredictionResponse(BaseModel):
    batch_id: str = Field(..., description="Идентификатор батча")
    predictions: List[PredictionResponse] = Field(..., description="Результаты предсказаний")
    total_processed: int = Field(..., description="Общее количество обработанных проектов")
    failed_predictions: int = Field(..., description="Количество неудачных предсказаний")
    average_confidence: float = Field(..., description="Средняя уверенность по батчу")
    batch_processing_time: float = Field(..., description="Общее время обработки батча")
    success_rate: float = Field(..., description="Процент успешных предсказаний")

    @field_validator('success_rate')
    @classmethod
    def validate_success_rate(cls, v: float) -> float:
        if not 0 <= v <= 1:
            raise ValueError('Success rate must be between 0 and 1')
        return round(v, 4)

    model_config = ConfigDict(extra='forbid')

class FeatureImportanceResponse(BaseModel):
    features: List[str] = Field(..., description="Названия признаков")
    importance_scores: List[float] = Field(..., description="Важность признаков")
    top_features: List[Dict[str, Any]] = Field(..., description="Топ наиболее важных признаков")

    model_config = ConfigDict(extra='forbid')

class ModelUpdateRequest(BaseModel):
    model_path: str = Field(..., description="Путь к новой модели")
    model_version: str = Field(..., description="Версия модели")
    backup_current: bool = Field(True, description="Создать backup текущей модели")
    validation_required: bool = Field(True, description="Требовать валидацию перед обновлением")

    model_config = ConfigDict(extra='forbid')

class PredictionHistory(BaseModel):
    request_id: str
    project_description: str
    predicted_technologies: List[str]
    confidence: float
    timestamp: datetime
    user_feedback: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(extra='forbid')

class UserFeedback(BaseModel):
    request_id: str = Field(..., description="Идентификатор запроса")
    rating: int = Field(..., ge=1, le=5, description="Оценка рекомендации от 1 до 5")
    comments: Optional[str] = Field(None, description="Комментарии пользователя")
    used_technologies: Optional[List[str]] = Field(None, description="Фактически использованные технологии")
    feedback_date: datetime = Field(default_factory=datetime.now)

    @field_validator('rating')
    @classmethod
    def validate_rating(cls, v: int) -> int:
        if v not in [1, 2, 3, 4, 5]:
            raise ValueError('Rating must be between 1 and 5')
        return v

    model_config = ConfigDict(extra='forbid')

class DatasetRow(BaseModel):
    id: int
    project_description: str
    project_type: str
    project_scale: str
    industry: str
    subdomain: str
    budget: str
    team_size: str
    complexity: str
    technical_requirements: Dict[str, str]
    tech_stack: List[str]
    business_requirements: List[str]
    timeline: str
    project_format: str
    created_at: str

    @field_validator('tech_stack', 'business_requirements')
    @classmethod
    def validate_list_fields(cls, v: Any) -> List[Any]:
        if not isinstance(v, list):
            raise ValueError('Must be a list')
        return v

    @field_validator('technical_requirements')
    @classmethod
    def validate_technical_requirements(cls, v: Dict[str, str]) -> Dict[str, str]:
        required_keys = {'performance', 'scalability', 'security', 'realtime', 'integration'}
        if not all(key in v for key in required_keys):
            raise ValueError(f'Technical requirements must contain all keys: {required_keys}')
        return v
    
    @field_validator('timeline')
    @classmethod
    def validate_timeline(cls, v: Optional[str]) -> Optional[str]:
        if v and len(v.strip()) > 50:
            raise ValueError('Timeline description too long')
        return v

    model_config = ConfigDict(extra='forbid')