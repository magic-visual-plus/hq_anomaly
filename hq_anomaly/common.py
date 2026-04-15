import pydantic
from typing import List, Optional
import numpy as np


class ModelConfig(pydantic.BaseModel):
    # Placeholder for model configuration
    image_size: int = 512
    checkpoint_path: str = ""

    pass

class TrainConfig(pydantic.BaseModel):
    modelConfig: ModelConfig = ModelConfig()    # Model configuration
    
    # Dataset related
    data_path: str    # Path to dataset
    num_data_workers: int = 0    # Number of data loader workers
    
    # Training related
    num_epochs: int = 100    # Number of training epochs
    warmup_epochs: int = 5    # Number of warmup epochs
    batch_size: int = 4    # Batch size
    gradient_update_interval: int = 1    # Gradient update interval

    lr0: float = 1e-5    # Initial learning rate
    lr_min: float = 1e-6    # Minimum learning rate
    devices: List[int] = [0]    # List of device ids to use

    output_path: str = "output"    # Path to save outputs


class PredictionResult(pydantic.BaseModel):
    class Config:
        arbitrary_types_allowed = True
        pass
    score: np.ndarray
    heat_map: Optional[np.ndarray] = None