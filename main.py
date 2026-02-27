from engine import CameraEngine
from Filters.composition_filters import *
from Filters.advanced_filters import *
from Filters.base_filter import Complexity
from Filters.template_filter import UniversalTemplateFilter # ייבוא מהקובץ החדש שלך
from Filters.technical_filters import HeadroomFilter

if __name__ == "__main__":
    engine = CameraEngine(model_path="model/yolov8n-pose.pt")

    filters = [
        CenteredFilter(name="MainCenterFilter", complexity=Complexity.LOW),
        RuleOfThirdsFilter("RuleOfThirds", Complexity.LOW),
        HeadroomFilter("Headroom", Complexity.MEDIUM),
        LookRoomFilter(model=engine.model),
        HeroShotFilter(name="HeroShotPro",complexity=Complexity.MEDIUM)
    ]
    my_template = UniversalTemplateFilter(
        name="MyUniversalFilter",
        model=engine.model,
        template_image_path="template.jpg"
    )

    def display_filters():
        print("Available Filters:")
        for idx, f in enumerate(filters):
            print(f"{idx + 1}. {f.name} - {f.description}")

    def get_user_choice():
        while True:
            try:
                choice = int(input("Enter the number of the filter you want to use: "))
                if 1 <= choice <= len(filters):
                    return filters[choice - 1]
                else:
                    print(f"Please enter a number between 1 and {len(filters)}.")
            except ValueError:
                print("Invalid input. Please enter a number.")

    def select_filter():
        display_filters()
        selected_filter = get_user_choice()
        print(f"You have selected: {selected_filter.name}")
        return selected_filter

    engine.set_filter(select_filter())

    engine.run_live_camera()