class ImageHandler:
    def __init__(self, model_path=None):
        # todo: use model or the function that aviv made to analyze the photo and return the best filter for it.
        pass

    def analyze_photo(self, image_path, filters_list):
        """
        This function takes the path to an uploaded image and a list of filter objects,
         analyzes the image using AI techniques,
          and determines which filter best suits the composition of the photo.
        :param image_path: The file path to the uploaded image.
        """
        print(f"Image AI analyzing: {image_path}")

        # todo: logic here

        # sample for returning
        for f in filters_list:
            if "RuleOfThirds" in f.name:
                return f

        return filters_list[0]  # default return if no specific match found, can be changed to None or a specific filter as needed