import torch.nn as nn

class EnergyCosineSimilarity:
    def __init__(self):
        self.cosine_similarity_fn = nn.CosineSimilarity(dim=1)

    def calculate_energy(self, original_pattern, queried_retrived_pattern):
        direction_loss = -self.cosine_similarity_fn(queried_retrived_pattern, original_pattern)
        energy = direction_loss.sum() / direction_loss.shape[0]
        return energy

class MSELoss:
    def __init__(self):
        self.mse_loss_fn = nn.MSELoss()

    def calculate_energy(self, original_pattern, queried_retrived_pattern):
        mse = self.mse_loss_fn(queried_retrived_pattern, original_pattern)
        return mse
