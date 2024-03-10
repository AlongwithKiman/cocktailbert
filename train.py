import json
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import argparse
from argparse import ArgumentParser
from dataloader import NSMCDataModule
from cocktailbert import BERTClassification


# Hardcoded arguments for running on Colab
# TODO: take this as config

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--config', default="./config/train.json", type=str)
    
    args = parser.parse_args()
    with open(args.config) as config_file:
        hparam = json.load(config_file)
    hparam = argparse.Namespace(**hparam)


    seed_everything(hparam.seed, workers=True)
    num_categories_per_class = [hparam.num_size_category, hparam.num_ABV_category, hparam.num_color_category]


    # set data module
    dm = NSMCDataModule(
        data_path=hparam.data_path,
        mode=hparam.mode,
        valid_size=hparam.valid_size,
        max_seq_len=hparam.max_seq_len,
        batch_size=hparam.batch_size,
        num_categories = len(num_categories_per_class)
    )
    dm.prepare_data()
    dm.setup('fit')

    breakpoint()
    # load model
    model = BERTClassification(num_categories_per_class = num_categories_per_class)

    checkpoint_callback = ModelCheckpoint(
        monitor='val_acc_0',
        dirpath=hparam.save_path,
        filename='{epoch:02d}-{val_acc:.3f}',
        verbose=True,
        save_last=False,
        mode='max',
        save_top_k=1,
    )
    early_stopping = EarlyStopping(
        monitor='val_acc_0',
        mode='max',
    )

    trainer = Trainer(
        max_epochs=hparam.max_epoch,
        accelerator='gpu',
        callbacks=[checkpoint_callback,],
    )
    trainer.fit(model, dm)