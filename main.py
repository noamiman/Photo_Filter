from engine import CameraEngine
from Filters.composition_filters import *
from Filters.advanced_filters import *
from Filters.base_filter import Complexity

if __name__ == "__main__":
    engine = CameraEngine(model_path="model/yolov8n.pt")

    my_filter = HeroShotFilter(
        name="Hero Shot Pro",
        complexity=Complexity.MEDIUM
    )
    other = RuleOfThirdsFilter(
        "RuleOfThirds",
        Complexity.LOW
    )

    engine.set_filter(my_filter)

    engine.run_live_camera()