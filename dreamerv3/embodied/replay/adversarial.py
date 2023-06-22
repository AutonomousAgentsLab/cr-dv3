import numpy as np
from dreamerv3.embodied.replay import CuriousReplay


class AdversarialReplay(CuriousReplay):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.should_track_visit_counts = False

    @staticmethod
    def _calculate_priority_score(model_loss, visit_count, hyper):
        return np.power((model_loss + hyper['epsilon']), hyper['alpha'])
