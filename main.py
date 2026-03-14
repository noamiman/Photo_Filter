import tkinter as tk
from tkinter import messagebox
from engine import CameraEngine
from Filters.composition_filters import *
from Filters.advanced_filters import *
from Filters.base_filter import Complexity
from Filters.template_filter import UniversalTemplateFilter
from Filters.technical_filters import HeadroomFilter
import customtkinter as ctk
from tkinter import filedialog
from PIL import Image
import os


# configure customtkinter appearance and theme
ctk.set_appearance_mode("dark")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

class FilterSelectorGUI:
    def __init__(self, filters_list, llm_handler=None, image_handler=None):
        self.filters = filters_list
        self.selected_filter = None
        self.llm_handler = llm_handler  # text analysis logic
        self.image_handler = image_handler  # image analysis logic

        self.uploaded_image_path = None
        self.ai_visible = False

        self.root = ctk.CTk()
        self.root.title("PhotoFilter Pro - Smart Selection")
        self.root.geometry("800x950")

        # --- title ---
        self.title_label = ctk.CTkLabel(self.root, text="PhotoFilter AI", font=ctk.CTkFont(size=32, weight="bold"))
        self.title_label.pack(pady=(30, 5))

        # --- Expander Toggle Button ---
        self.toggle_ai_btn = ctk.CTkButton(
            self.root, text="✨ Use AI Assistants (Text/Image) ▼",
            command=self.toggle_ai_section, fg_color="transparent", border_width=2, width=350
        )
        self.toggle_ai_btn.pack(pady=10)

        # --- AI SECTION (The Expander Frame) ---
        self.ai_section = ctk.CTkFrame(self.root, fg_color="#2b2b2b", corner_radius=15)

        # text prompt part
        self.tab_label = ctk.CTkLabel(self.ai_section, text="Option A: Describe your shot",
                                      font=ctk.CTkFont(size=14, weight="bold"))
        self.tab_label.pack(pady=(15, 5), padx=20, anchor="w")

        self.prompt_entry = ctk.CTkEntry(self.ai_section, placeholder_text="e.g. 'Golden hour portrait'...", height=40)
        self.prompt_entry.pack(pady=5, padx=20, fill="x")

        # photo upload part
        self.img_label = ctk.CTkLabel(self.ai_section, text="Option B: Upload a reference photo",
                                      font=ctk.CTkFont(size=14, weight="bold"))
        self.img_label.pack(pady=(15, 5), padx=20, anchor="w")

        self.upload_btn = ctk.CTkButton(self.ai_section, text="📁 Choose Image", command=self.upload_image,
                                        fg_color="#444444")
        self.upload_btn.pack(pady=5)

        # priview label for uploaded image
        self.preview_label = ctk.CTkLabel(self.ai_section, text="")  # container for image preview
        self.preview_label.pack(pady=5)

        # button to process AI request (either text or image)
        self.magic_btn = ctk.CTkButton(
            self.ai_section, text="Analyze & Apply ✨",
            command=self.process_ai_request, fg_color="#A367B1", hover_color="#5D3587",
            height=45, font=ctk.CTkFont(size=15, weight="bold")
        )
        self.magic_btn.pack(pady=20)

        # --- separator ---
        self.separator = ctk.CTkFrame(self.root, height=2, fg_color="#3d3d3d")
        self.separator.pack(fill="x", padx=100, pady=20)

        self.scrollable_frame = ctk.CTkScrollableFrame(self.root, width=700, height=400)
        self.scrollable_frame.pack(padx=20, pady=10, fill="both", expand=True)

        for f in self.filters:
            self.create_filter_card(f)

    def toggle_ai_section(self):
        if not self.ai_visible:
            self.ai_section.pack(pady=10, padx=40, fill="x", after=self.toggle_ai_btn)
            self.toggle_ai_btn.configure(text="✨ Close AI Assistants ▲")
            self.ai_visible = True
        else:
            self.ai_section.pack_forget()
            self.toggle_ai_btn.configure(text="✨ Use AI Assistants (Text/Image) ▼")
            self.ai_visible = False

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
        if file_path:
            self.uploaded_image_path = file_path
            # make preview thumbnail
            img = Image.open(file_path)
            img.thumbnail((150, 150))  # resize to fit the preview area
            ctk_img = ctk.CTkImage(light_image=img, dark_image=img, size=(150, 150))
            self.preview_label.configure(image=ctk_img, text="")
            print(f"[*] Image uploaded: {file_path}")

    def process_ai_request(self):
        """
        decides whether to analyze the uploaded image or the text prompt, and calls the appropriate handler to get the best filter recommendation.
         The result is then applied immediately.
        """
        # priority to image analysis if an image is uploaded, otherwise fallback to text analysis
        if self.uploaded_image_path and self.image_handler:
            print("[*] Analyzing image...")
            result = self.image_handler.analyze_photo(self.uploaded_image_path, self.filters)
            if result: self.finish_selection(result)
            return

        # if no image analysis, try text analysis if there's input
        user_text = self.prompt_entry.get()
        if user_text and self.llm_handler:
            print(f"[*] Analyzing text: {user_text}")
            result = self.llm_handler.get_filter_from_text(user_text, self.filters)
            if result: self.finish_selection(result)

    def create_filter_card(self, filter_obj):
        card = ctk.CTkFrame(self.scrollable_frame)
        card.pack(pady=10, fill="x", padx=20)

        ctk.CTkLabel(card, text=filter_obj.name, font=ctk.CTkFont(size=18, weight="bold")).pack(pady=(10, 2))
        ctk.CTkLabel(card, text=filter_obj.description, font=ctk.CTkFont(size=13), text_color="lightgray",
                     wraplength=550).pack(pady=(0, 10))

        ctk.CTkButton(card, text="Apply Filter", command=lambda o=filter_obj: self.finish_selection(o), width=150).pack(
            pady=(0, 15))

    def finish_selection(self, filter_obj):
        self.selected_filter = filter_obj
        self.root.after(200, self.root.destroy)

    def run(self):
        try:
            self.root.mainloop()
        except Exception:
            pass
        finally:
            return self.selected_filter


if __name__ == "__main__":
    # initialize the camera engine with the pose estimation model
    engine = CameraEngine(model_path="model/yolov8n-pose.pt")

    # get the logic handlers for LLM and image analysis
    from LLMHandler import LLMHandler
    from ImageHandler import ImageHandler

    llm_logic = LLMHandler()
    image_logic = ImageHandler()

    filters = [
        CenteredFilter(name="MainCenterFilter", complexity=Complexity.LOW, model=engine.model),
        RuleOfThirdsFilter("RuleOfThirds", Complexity.LOW, model=engine.model),
        HeadroomFilter("Headroom", Complexity.MEDIUM, model=engine.model),
        LookRoomFilter(),
        HeroShotFilter(name="HeroShotPro", complexity=Complexity.MEDIUM, model=engine.model),
        UniversalTemplateFilter(name="PoseStencil", model=engine.model)
    ]

    # create and run the GUI for filter selection, passing the filters and logic handlers
    gui = FilterSelectorGUI(
        filters_list=filters,
        llm_handler=llm_logic,
        image_handler=image_logic
    )

    selected_filter = gui.run()

    # chose the filter and start the camera engine with it
    if selected_filter:
        print(f"[*] Engine starting with filter: {selected_filter.name}")
        engine.set_filter(selected_filter)
        engine.run_live_camera()
    else:
        print("[!] Selection cancelled or window closed.")