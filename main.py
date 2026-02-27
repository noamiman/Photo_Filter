from engine import CameraEngine
from Filters.template_filter import UniversalTemplateFilter # ייבוא מהקובץ החדש שלך
from Filters.base_filter import Complexity
from Filters.composition_filters import *
from Filters.base_filter import Complexity

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
    # 2. יצירת פילטר השבלונה
    # וודא שיש לך קובץ בשם 'template.jpg' בתיקייה הראשית, או שנה את השם כאן
    my_template = UniversalTemplateFilter(
        name="MyUniversalFilter",
        model=engine.model,
        template_image_path="template.jpg"
    )

    engine.set_filter(other)

    engine.run_live_camera()