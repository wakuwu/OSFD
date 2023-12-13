from .IFGSM import IFGSM
from attack.utils.registry import BASEATK
import torch


@BASEATK.register_module()
class MI(IFGSM):
    def __init__(self, alpha=1.0, momentum=1.0) -> None:
        super().__init__(alpha)
        self.momentum = momentum

    def preprocess_data(self, results):
        results = super().preprocess_data(results)
        return results

    def combine_losses(self, results):
        results = super().combine_losses(results)
        return results

    def process_gradients(self, results):
        buffer = results["buffer"]
        idx = results["idx"]
        gradients_adv = results.pop("gradients_adv")

        # Load from buffer
        gradients_last_momentum = results.pop("gradients_with_momentum", None)
        if gradients_last_momentum is None:
            gradients_last_momentum = buffer.load("momentum", idx)
        if gradients_last_momentum is None:
            gradients_last_momentum = 0.
        gradients_with_momentum = self.momentum * gradients_last_momentum + \
                                  gradients_adv / torch.mean(torch.abs(gradients_adv), dim=[1, 2, 3], keepdim=True)

        results["gradients_with_momentum"] = gradients_with_momentum
        results["gradients_adv"] = gradients_with_momentum
        return results

    def update_noise(self, results):
        results = super().update_noise(results)
        return results

    def postprocess_data(self, results):
        buffer = results['buffer']
        idx = results['idx']
        gradients_with_momentum = results['gradients_with_momentum']
        buffer.dump("momentum", idx, gradients_with_momentum)
        return results