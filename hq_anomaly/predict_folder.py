
from hq_anomaly.models import ViTPatchcore
from hq_anomaly.common import ModelConfig
import sys
import time
import cv2
import os

if __name__ == "__main__":
    model_config = ModelConfig(
        checkpoint_path=sys.argv[1],
    )
    model = ViTPatchcore(model_config)
    model.eval()

    input_path = sys.argv[2]
    output_path = sys.argv[3]

    filenames = [f for f in os.listdir(input_path) if f.endswith(".jpg")]
    cnt = 0
    for filename in filenames:
        start = time.time()
        image = cv2.imread(os.path.join(input_path, filename))
        result = model.predict([image], return_heatmap=True)[0]
        print(f"{filename}: {result.score.max()}")
        if result.score.max() < 0.7:
            cnt += 1

        cv2.imwrite(os.path.join(output_path, filename), result.heat_map)
        pass
    print(cnt)
    pass