import wa_utils


class Estimator:
    def __init__(self, model_for_empty, model_for_color):
        self.model_for_empty = model_for_empty
        self.model_for_color = model_for_color

    def estimate_puyo(self, image):
        x = wa_utils.transform_for_recognition(image)
        c = self.model_for_empty.predict([x])[0].upper()
        if c == 'E':
            return ' '
        return self.model_for_color.predict([x])[0].upper()
