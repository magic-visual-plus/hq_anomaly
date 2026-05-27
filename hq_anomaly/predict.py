
from hq_anomaly.models import ViTPatchcore
from hq_anomaly.common import ModelConfig
import sys
import time
import cv2

if __name__ == "__main__":
    model_config = ModelConfig(
        checkpoint_path=sys.argv[1],
    )
    model = ViTPatchcore(model_config)

    model.eval()

    start = time.time()
    image = cv2.imread(sys.argv[2])
    result = model.predict([image], return_heatmap=True)[0]
    print(f"cost: {time.time() - start} seconds")

    cv2.imwrite(sys.argv[3], result.heat_map)
    pass