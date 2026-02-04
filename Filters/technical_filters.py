from Filters.base_filter import BaseFilter
class HeadroomFilter(BaseFilter):
    @property
    def description(self):
        return "Maintains the ideal gap between the head and the top edge."

    def apply(self, frame):
        pass

class DistanceFilter(BaseFilter):
    @property
    def description(self):
        return "Guides the photographer to the correct distance (Portrait vs Full Body)."

    def apply(self, frame):
        pass