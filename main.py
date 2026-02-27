from engine import CameraEngine
from Filters.composition_filters import *
from Filters.base_filter import Complexity
from  Filters.technical_filters import HeadroomFilter

if __name__ == "__main__":


    engine = CameraEngine(model_path="model/yolov8n-pose.pt")

    my_filter = CenteredFilter(
        name="MainCenterFilter",
        complexity=Complexity.LOW
    )
    other = RuleOfThirdsFilter(
        "RuleOfThirds",
        Complexity.LOW
    )
    sky = HeadroomFilter(
        "Headroom",
        Complexity.MEDIUM
    )

    look_room_filter = LookRoomFilter(model=engine.model)
    engine.set_filter(sky)

    #shuli
    engine.set_filter(look_room_filter)

    # engine.set_filter(other)

    engine.run_live_camera()