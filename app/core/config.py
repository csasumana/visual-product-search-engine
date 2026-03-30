from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    APP_NAME: str = "Visual Product Search Engine"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8002

    CLIP_MODEL_NAME: str = "ViT-B-32"
    CLIP_PRETRAINED: str = "laion2b_s34b_b79k"
    DEVICE: str = "cpu"

    DATA_DIR: str = "data"
    IMAGES_DIR: str = "data/images"
    METADATA_PATH: str = "data/metadata.csv"

    EMBEDDINGS_DIR: str = "embeddings"
    EMBEDDINGS_PATH: str = "embeddings/image_embeddings.npy"
    EMBEDDINGS_FP16_PATH: str = "embeddings/image_embeddings_fp16.npy"
    EMBEDDING_MAP_PATH: str = "embeddings/embedding_map.csv"

    INDEX_DIR: str = "indexes"
    GLOBAL_INDEX_PATH: str = "indexes/global.index"
    CATEGORY_INDEX_DIR: str = "indexes/category_indexes"
    INDEX_META_PATH: str = "indexes/index_meta.json"

    DEFAULT_TOP_K: int = 5
    MAX_SEARCH_K: int = 20

    RAW_DATASET_ROOT: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

    @property
    def project_root(self) -> Path:
        return Path(__file__).resolve().parents[2]

    def resolve_path(self, relative_path: str) -> Path:
        return self.project_root / relative_path

    def resolve_raw_dataset_path(self, relative_path: str) -> Path:
        """
        DeepFashion CategoryAttribute layout on your machine:
        RAW_DATASET_ROOT = F:/datasets/DeepFashion/CategoryAttribute
        metadata path    = img/ProductFolder/img_00000001.jpg
        actual file      = F:/datasets/DeepFashion/CategoryAttribute/img/img/ProductFolder/img_00000001.jpg
        """
        if not self.RAW_DATASET_ROOT:
            raise ValueError("RAW_DATASET_ROOT is not set in .env")

        return Path(self.RAW_DATASET_ROOT) / "img" / relative_path


settings = Settings()