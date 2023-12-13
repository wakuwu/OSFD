import os


class BaseAttack(object):

    def __init__(self) -> None:
        super().__init__()
        self.device = os.environ["device"]

    def preprocess_data(self, results):
        return results

    def combine_losses(self, results):
        return results

    def process_gradients(self, results):
        return results

    def update_noise(self, results):
        return results

    def postprocess_data(self, results):
        return results