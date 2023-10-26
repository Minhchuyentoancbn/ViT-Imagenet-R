import pytorch_lightning as pl
import torch
import torch.nn as nn
from utils import Averager


class BaseModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.criteria = nn.CrossEntropyLoss()
        self.automatic_optimization = False
        self.loss_val_avg = Averager()
        self.acc_val_avg = Averager()

    def forward(self, x):
        raise NotImplementedError
    
    def training_step(self, batch, batch_idx):
        # Get the optimizer
        opt = self.optimizers()
        opt.zero_grad()

        # Get the features and labels
        features, labels = batch

        # Compute loss
        self.train()
        preds = self(features)
        loss = self.criteria(preds, labels)

        # Backpropagate
        self.manual_backward(loss)
        self.log('train_loss', loss, prog_bar=True)


    def validation_step(self, batch, batch_idx):
        features, labels = batch
        self.eval()
        preds = self(features)
        loss = self.criteria(preds, labels)
        self.log('val_loss', loss, prog_bar=True)

        # Compute accuracy
        preds = torch.argmax(preds, dim=1)
        acc = torch.sum(preds == labels).float() / len(labels)
        self.log('val_acc', acc, prog_bar=True)

        self.acc_val_avg.add(acc.item())
        self.loss_val_avg.add(loss.item())


    def on_validation_epoch_end(self):
        print(f'Validation Loss: {self.loss_val_avg.val():.4f}. Validation Accuracy: {self.acc_val_avg.val():.4f}')
        self.loss_val_avg.reset()
        self.acc_val_avg.reset()


    def configure_optimizers(self):
        optimizer_params = {
            'lr': self.args.lr,
            'weight_decay': self.args.weight_decay
        }

        if self.args.optim in ['Adam', 'AdamW']:
            optimizer_params['betas'] = (self.args.momentum, 0.999)
        elif self.args.optim == 'SGD':
            optimizer_params['momentum'] = self.args.momentum

        optimizer = torch.optim.__dict__[self.args.optim](self.parameters(), **optimizer_params)

        return optimizer

