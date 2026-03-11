import tkinter as tk
from tkinter import messagebox
from engine import CameraEngine
from Filters.composition_filters import *
from Filters.advanced_filters import *
from Filters.base_filter import Complexity
from Filters.template_filter import UniversalTemplateFilter # ייבוא מהקובץ החדש שלך
from Filters.technical_filters import HeadroomFilter


class FilterSelectorGUI:
    def __init__(self, filters_list):
        self.filters = filters_list
        self.selected_filter = None

        self.root = tk.Tk()
        self.root.title("Select Your Photography Filter")
        self.root.geometry("450x600")

        # head lines
        tk.Label(self.root, text="Available Filters:", font=("Arial", 16, "bold"), pady=15).pack()

        # creating buttons
        for f in self.filters:
            button_text = f"{f.name}\n{f.description}"

            btn = tk.Button(self.root,
                            text=button_text,
                            font=("Arial", 10),
                            width=45,
                            height=3,
                            wraplength=350,
                            pady=5,
                            command=lambda obj=f: self.finish_selection(obj))
            btn.pack(pady=8)

    def finish_selection(self, filter_obj):
        self.selected_filter = filter_obj
        print(f"You have selected: {self.selected_filter.name}")  # שומר על ההדפסה שלך
        self.root.destroy()

    def run(self):
        self.root.mainloop()
        return self.selected_filter

if __name__ == "__main__":
    engine = CameraEngine(model_path="model/yolov8n-pose.pt")

    filters = [
        CenteredFilter(name="MainCenterFilter", complexity=Complexity.LOW),
        RuleOfThirdsFilter("RuleOfThirds", Complexity.LOW),
        HeadroomFilter("Headroom", Complexity.MEDIUM),
        LookRoomFilter(model=engine.model),
        HeroShotFilter(name="HeroShotPro",complexity=Complexity.MEDIUM),
        UniversalTemplateFilter(name="Gallery Template",model=engine.model)
    ]

    # הפעלת הבחירה הגרפית
    gui = FilterSelectorGUI(filters)
    selected_filter = gui.run()

    # בדיקה שנבחר פילטר (למקרה שהמשתמש סגר את החלון ב-X)
    if selected_filter:
        engine.set_filter(selected_filter)
        engine.run_live_camera()
    else:
        print("Selection cancelled.")