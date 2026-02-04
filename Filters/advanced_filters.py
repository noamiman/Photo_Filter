from Filters.base_filter import BaseFilter

class HorizonLevelerFilter(BaseFilter):
    @property
    def description(self):
        return "Ensures the camera is level and the horizon is not tilted."

    def apply(self, frame):
        pass

class SymmetryFilter(BaseFilter):
    @property
    def description(self):
        return "Helps achieve visual balance between the subject and the background."

    def apply(self, frame):
        pass