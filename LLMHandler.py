class LLMHandler:
    def __init__(self, api_key=None):
        self.api_key = api_key

    def get_filter_from_text(self, user_text, filters_list):
        """
        This function takes user input text and a list of filter objects, and uses an LLM to determine which filter best matches the user's description.
        :param user_text: The text input from the user describing what they want.
        :param filters_list: A list of filter objects that the LLM can choose from.
        :return: The filter object that best matches the user's description, or None if no match
        """
        print(f"LLM analyzing: {user_text}")

        # todo: logic here

        # sample for returning
        selected_name = "HeroShotPro"

        for f in filters_list:
            if f.name == selected_name:
                return f

        return None