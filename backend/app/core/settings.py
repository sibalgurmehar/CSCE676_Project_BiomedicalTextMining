from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field


class ProjectInfo(BaseModel):
    name: str
    tagline: str
    research_question: str


class DatasetInfo(BaseModel):
    document_source: str
    query_source: str
    join_key: str
    output_formats: list[str]


class RetrievalMethod(BaseModel):
    id: str
    label: str
    family: str
    description: str


class FrontendDefaults(BaseModel):
    default_query: str
    results_per_method: int


class ProjectSettings(BaseModel):
    project: ProjectInfo
    datasets: DatasetInfo
    retrieval_methods: list[RetrievalMethod]
    analysis_modules: list[str]
    frontend: FrontendDefaults
    settings_file: str = Field(default="")


@lru_cache
def get_project_settings() -> ProjectSettings:
    root = Path(__file__).resolve().parents[3]
    settings_path = root / "shared" / "config" / "settings.json"
    data = ProjectSettings.model_validate_json(settings_path.read_text(encoding="utf-8"))
    data.settings_file = str(settings_path)
    return data

