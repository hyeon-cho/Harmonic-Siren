import torch.nn as nn

class EnergyCosineSimilarity:
    def __init__(self):
        self.cosine_similarity_fn = nn.CosineSimilarity(dim=1)

    def calculate_energy(self, original_pattern, queried_retrived_pattern):
        direction_loss = -self.cosine_similarity_fn(queried_retrived_pattern, original_pattern)
        energy = direction_loss.sum() / direction_loss.shape[0]
        return energy


class MSEloss:
    def __init__(self):
        self.mse_fn = nn.MSELoss()

    def calculate_energy(self, original_pattern, queried_retrived_pattern):
        # Match the EnergyCosineSimilarity API used by base_trainer.
        return self.mse_fn(queried_retrived_pattern, original_pattern)
