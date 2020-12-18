# This implementation is inspired from
# https://github.com/HaoMood/bilinear-cnn


import torch
import torchvision
from torch.autograd import Function, profiler

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
        with profiler.record_function("bcnn_normalization"):
            X = X.view(-1, 512, 28 ** 2)
            X = torch.bmm(X, torch.transpose(X, 1, 2)) / (28 ** 2)  # Bilinear

            X = X.view(-1, 512 ** 2)

            # BCNN Normalization
            X = X.sign().mul(torch.sqrt(X.abs() + 1e-5))
            X = torch.nn.functional.normalize(X)
        X = self.fc(X)

        return X


# Kudos https://github.com/DennisLeoUTS/improved-bilinear-pooling/blob/master/utils/bilinear_layers.py
def sqrt_newton_schulz(A, num_iter):
    batch_size = A.shape[0]
    dim = A.shape[1]
    norm_A = A.mul(A).sum(dim=1).sum(dim=1).sqrt()
    Y = A.div(norm_A.view(batch_size, 1, 1).expand_as(A))
    I = torch.eye(dim, dim).view(1, dim, dim).repeat(batch_size, 1, 1).type(torch.cuda.FloatTensor)
    Z = torch.eye(dim, dim).view(1, dim, dim).repeat(batch_size, 1, 1).type(torch.cuda.FloatTensor)
    for i in range(num_iter):
        T = 0.5 * (3.0 * I - Z.bmm(Y))
        Y = Y.bmm(T)
        Z = T.bmm(Z)
    return Y * torch.sqrt(norm_A).view(batch_size, 1, 1).expand_as(A)


def lyap_newton_schulz(z, dldz, num_iter):
    batch_size = z.shape[0]
    dim = z.shape[1]
    norm_z = z.mul(z).sum(dim=1).sum(dim=1).sqrt()
    a = z.div(norm_z.view(batch_size, 1, 1).expand_as(z))
    I = torch.eye(dim, dim).view(1, dim, dim).repeat(batch_size, 1, 1).type(torch.cuda.FloatTensor)
    q = dldz.div(norm_z.view(batch_size, 1, 1).expand_as(z))
    for i in range(num_iter):
        q = 0.5 * (q.bmm(3.0 * I - a.bmm(a)) - a.transpose(1, 2).bmm(a.transpose(1, 2).bmm(q) - q.bmm(a)))
        a = 0.5 * a.bmm(3.0 * I - a.bmm(a))
    dlda = 0.5 * q
    return dlda


class matrix_sqrt(Function):
    @staticmethod
    def forward(ctx, x):
        output = sqrt_newton_schulz(x, 10)
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output = ctx.saved_tensors[0]
        grad_input = lyap_newton_schulz(output, grad_output, 10)
        return grad_input


class IBCNN(BCNN):

    def __init__(self, freeze_features):
        super().__init__(freeze_features)
        self.matrix_sqrt = matrix_sqrt.apply

    def forward(self, X):
        X = self.features(X)
        with profiler.record_function("bcnn_normalization"):
            X = X.view(-1, 512, 28 ** 2)
            X = torch.bmm(X, torch.transpose(X, 1, 2)) / (28 ** 2)  # Bilinear

            # I-BCNN Normalization
            X = self.matrix_sqrt(X)

            X = X.view(-1, 512 ** 2)

            # BCNN Normalization
            X = X.sign().mul(torch.sqrt(X.abs() + 1e-5))
            X = torch.nn.functional.normalize(X)
        X = self.fc(X)

        return X


class BCNNwRUN(BCNN):
    def __init__(self, freeze_features, it_k=2, eta=1.0):
        super().__init__(freeze_features)
        self.run = RUN(it_k, eta)

    def forward(self, X):
        features = self.features(X)
        with profiler.record_function("bcnn_normalization"):
            Fk = self.run(features)

            out = torch.bmm(torch.transpose(Fk, 1, 2), Fk) / (28 ** 2)
            out = out.view(out.shape[0], -1)

            out = out.sign().mul(torch.sqrt(out.abs() + 1e-5))
            out = torch.nn.functional.normalize(out)
        out = self.fc(out)

        return out


class RUN(torch.nn.Module):
    def __init__(self, it_k=2, eta=1.0):
        torch.nn.Module.__init__(self)
        self.it_k = it_k
        self.eta = eta

    def forward(self, features):
        B, D, W, H = features.shape
        F = torch.transpose(features.view(B, D, W * H), 1, 2)  # B x N x D
        v = torch.randn(B, D, 1, requires_grad=False, device=F.device)
        for i in range(self.it_k):
            v = torch.bmm(F, v)  # Fv_k
            v = torch.bmm(torch.transpose(F, 1, 2), v)  # F^TFv_k-1

        v = torch.nn.functional.normalize(v, dim=1)
        vt = torch.transpose(v, 1, 2)

        Fk = F - self.eta * torch.bmm(torch.bmm(F, v), vt)  # F - eta*Fvv^T
        return Fk
