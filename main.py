from engine import CameraEngine
from Photo_Filter.Filters.composition_filters import *
from Photo_Filter.Filters.base_filter import Complexity

if __name__ == "__main__":
    engine = CameraEngine(model_path="model/yolov8n.pt")

    my_filter = CenteredFilter(
        name="MainCenterFilter",
        complexity=Complexity.LOW
    )
    other = RuleOfThirdsFilter(
        "RuleOfThirds",
        Complexity.LOW
    )

    engine.set_filter(other)

    engine.run_live_camera()