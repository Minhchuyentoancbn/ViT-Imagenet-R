import pickle
import sys
import torch
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from utils import parse_arguments
from config import DATA_DIR
from models.mlp import MLP


if __name__ == '__main__':
    args = parse_arguments(sys.argv[1:])
    pl.seed_everything(args.seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Load data
    train_feats = []
    train_labels = []
    eval_feats = []
    eval_labels = []
    for i in range(10):
        with open(f'{DATA_DIR}/train_features_and_labels_task_{i}.pkl', 'rb') as fr:
            feature, label = pickle.load(fr)
            train_feats.append(feature)
            train_labels.append(label)

        with open(f'{DATA_DIR}/eval_features_and_labels_task_{i}.pkl', 'rb') as fr:
            feature, label = pickle.load(fr)
            eval_feats.append(feature)
            eval_labels.append(label)

    train_feats = torch.cat(train_feats, dim=0)
    train_labels = torch.cat(train_labels, dim=0)
    eval_feats = torch.cat(eval_feats, dim=0)
    eval_labels = torch.cat(eval_labels, dim=0)

    # Shuffle
    idx = torch.randperm(len(train_feats))
    train_feats = train_feats[idx]
    train_labels = train_labels[idx]
    idx = torch.randperm(len(eval_feats))
    eval_feats = eval_feats[idx]
    eval_labels = eval_labels[idx]

    # Data loader
    train_dataset = torch.utils.data.TensorDataset(train_feats, train_labels)
    eval_dataset = torch.utils.data.TensorDataset(eval_feats, eval_labels)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        drop_last=True, num_workers=2, pin_memory=True
    )
    val_loader = DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False,
        drop_last=False, num_workers=2, pin_memory=True
    )

    if args.model_name == 'MLP':
        model = MLP(args)

    # train model
    trainer = pl.Trainer(
        default_root_dir=f'logs/{args.model_name}/',
        max_epochs=args.epochs,
        val_check_interval=1.0,
    )
    trainer.fit(
        model=model, train_dataloaders=train_loader, val_dataloaders=val_loader
    )


    # Save models
    torch.save(model.state_dict(), f'saved_models/{args.model_name}.pt')