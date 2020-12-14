# This implementation is inspired from
# https://github.com/HaoMood/bilinear-cnn

import os

import torch
import torchvision


class BCNNManager(object):

    def __init__(self, config):

        print('Prepare the network and data.')
        self.device = torch.device(config["device"])

        self._options = config["train_options"]
        self._path = config["path"]

        self._net = config["model"](config["freeze_features"]).to(self.device)

        if "pretrained" in self._path:
            self._net.load_state_dict(torch.load(self._path['pretrained'], map_location=self.device))
        print(self._net)

        self._criterion = torch.nn.CrossEntropyLoss().to(self.device)

        self._solver = torch.optim.SGD(
            self._net.trainable_params(), lr=self._options['base_lr'],
            momentum=0.9, weight_decay=self._options['weight_decay'])
        self._scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self._solver, mode='max', factor=0.1, patience=3, verbose=True,
            threshold=1e-4)

        train_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=448),
            torchvision.transforms.RandomHorizontalFlip(),
            torchvision.transforms.RandomCrop(size=448),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ])
        test_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(size=448),
            torchvision.transforms.CenterCrop(size=448),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                             std=(0.229, 0.224, 0.225))
        ])
        train_data = config["dataset"](
            root=self._path['dataset'], train=True,
            transform=train_transforms)
        test_data = config["dataset"](
            root=self._path['dataset'], train=False,
            transform=test_transforms)
        self._train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=self._options['batch_size'],
            shuffle=True, num_workers=1, pin_memory=True)
        self._test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=self._options['batch_size'],
            shuffle=False, num_workers=1, pin_memory=True)

    def train(self):
        print('Training.')
        best_acc = 0.0
        best_epoch = None
        print('Epoch\tTrain loss\tTrain acc\tTest acc')
        for t in range(self._options['epochs']):
            epoch_loss = []
            num_correct = 0
            num_total = 0
            for X, y in self._train_loader:
                # Data.
                X = X.to(self.device)
                y = y.to(self.device)
                self._solver.zero_grad()

                score = self._net(X)
                loss = self._criterion(score, y)
                epoch_loss.append(loss.item())
                # Prediction.
                _, prediction = torch.max(score.data, 1)
                num_total += y.size(0)
                num_correct += torch.sum(prediction == y.data)
                print(loss.item(), 100 * num_correct / num_total)
                # Backward pass.
                loss.backward()
                self._solver.step()
            train_acc = 100 * num_correct / num_total
            test_acc = self._accuracy(self._test_loader)
            self._scheduler.step(test_acc)
            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = t + 1
                print('*', end='')
                # Save model onto disk.
                torch.save(self._net.state_dict(), self._path['model'] + f".pth")
            print('%d\t%4.3f\t\t%4.2f%%\t\t%4.2f%%' %
                  (t + 1, sum(epoch_loss) / len(epoch_loss), train_acc, test_acc))
        print('Best at epoch %d, test accuracy %f' % (best_epoch, best_acc))

    def _accuracy(self, data_loader):
        self._net.train(False)
        num_correct = 0
        num_total = 0
        with torch.no_grad():
            for X, y in data_loader:
                # Data.
                X = X.to(self.device)
                y = y.to(self.device)

                # Prediction.
                score = self._net(X)
                _, prediction = torch.max(score.data, 1)
                num_total += y.size(0)
                num_correct += torch.sum(prediction == y.data).item()
        self._net.train(True)  # Set the model to training phase
        return 100 * num_correct / num_total
