# This implementation is inspired from
# https://github.com/HaoMood/bilinear-cnn


import torch
import torchvision

torch.manual_seed(41)
torch.cuda.manual_seed_all(41)


class BCNN(torch.nn.Module):

    def __init__(self, freeze_features):
        torch.nn.Module.__init__(self)
        self.freeze_features = freeze_features

        self.features = torchvision.models.vgg16(pretrained=True).features
        self.features = torch.nn.Sequential(*list(
            self.features.children())[:-1])

        self.fc = torch.nn.Linear(512 ** 2, 200)

        if freeze_features:
            for param in self.features.parameters():
                param.requires_grad = False
            torch.nn.init.kaiming_normal_(self.fc.weight.data)
            if self.fc.bias is not None:
                torch.nn.init.constant_(self.fc.bias.data, val=0)

    def trainable_params(self):
        if self.freeze_features:
            return self.fc.parameters()
        else:
            return self.parameters()

    def forward(self, X):
        X = self.features(X)
        X = X.view(-1, 512, 28 ** 2)
        X = torch.bmm(X, torch.transpose(X, 1, 2)) / (28 ** 2)  # Bilinear

        X = X.view(-1, 512 ** 2)

        # BCNN Normalization
        X = torch.sqrt(X + 1e-5)
        X = torch.nn.functional.normalize(X)
        X = self.fc(X)

        return X
