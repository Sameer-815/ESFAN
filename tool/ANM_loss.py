import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveWeightScheduler:
    def __init__(self, alpha_start=0.4, alpha_end=0.8,
                 beta_start=0.4, beta_end=0.1,
                 gamma_start=0.2, gamma_end=0.1,
                 tau_start=0.2, tau_end=0.5,
                 total_epochs=20):
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.gamma_start = gamma_start
        self.gamma_end = gamma_end
        self.tau_start = tau_start
        self.tau_end = tau_end
        self.total_epochs = total_epochs

    def get_weights(self, epoch):
        alpha = self.alpha_start + (self.alpha_end - self.alpha_start) * (epoch / self.total_epochs)
        beta = self.beta_start + (self.beta_end - self.beta_start) * (epoch / self.total_epochs)
        gamma = self.gamma_start + (self.gamma_end - self.gamma_start) * (epoch / self.total_epochs)
        tau = self.tau_start + (self.tau_end - self.tau_start) * (epoch / self.total_epochs)
        return alpha, beta, gamma, tau

class AdaptiveNoiseLoss(nn.Module):
    def __init__(self, alpha=0.6, beta=0.3, gamma=0.1, tau=0.5):
        super(AdaptiveNoiseLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tau = tau

    def forward(self, pred, target):

        probs = torch.softmax(pred, dim=1)
        target_one_hot = F.one_hot(target, num_classes=pred.size(1)).permute(0, 3, 1, 2).float()

        conf = probs.max(dim=1)[0]
        weight = (conf > self.tau).float()
        weight = weight.unsqueeze(1)
        weight = weight.expand_as(target_one_hot)
        # print(probs.size(),target_one_hot.size())
        ce_loss = -weight * (target_one_hot * torch.log(probs + 1e-6))
        ce_loss = ce_loss.mean()

        uncertainty_loss = -torch.mean((1 - conf) * torch.log(1 - conf + 1e-6))

        dx = torch.abs(probs[:, :, 1:, :] - probs[:, :, :-1, :])
        dy = torch.abs(probs[:, :, :, 1:] - probs[:, :, :, :-1])
        smooth_loss = torch.mean(dx) + torch.mean(dy)

        loss = self.alpha * ce_loss + self.beta * uncertainty_loss + self.gamma * smooth_loss
        return loss

