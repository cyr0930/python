import pytorch_lightning as pl
from pytorch_lightning import loggers
from machine_learning.nlp import config
from machine_learning.nlp.model import GPT2
from machine_learning.nlp.data_module import DataModule
from machine_learning.nlp.tokenizer import get_tokenizer

if __name__ == "__main__":
    tokenizer = get_tokenizer()
    model = GPT2(tokenizer, config.emsize, config.nhead, config.nlayers, config.bptt, config.ntokens, config.dropout)
    dm = DataModule(tokenizer)
    trainer = pl.Trainer(
        gradient_clip_val=config.gladient_clip_val,
        max_epochs=config.epochs,
        gpus=1,
        logger=loggers.TensorBoardLogger('runs/', name=None)
    )
    trainer.fit(model, datamodule=dm)
