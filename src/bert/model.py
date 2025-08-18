import torch
from torch.optim import AdamW
from pytorch_lightning import LightningModule
from transformers import BertForSequenceClassification, get_linear_schedule_with_warmup
from torchmetrics import Accuracy


class BaseModel(LightningModule):
    def __init__(self, config, tokenizer):
        super().__init__()
        self.cfg = config
        self.tokenizer = tokenizer

        self.model = BertForSequenceClassification.from_pretrained(
            self.cfg.model_pretrained
        )

        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=float(self.cfg.lr))

        # scheduler with warmup
        num_devices = self.trainer.num_devices
        total_steps = (
            len(self.trainer.datamodule.train_dataloader())
            // self.cfg.accumulate_grad_batches
            // num_devices
            * self.cfg.max_epochs
        )
        warmup_steps = int(total_steps * self.cfg.warmup_ratio)

        scheduler = {
            "scheduler": get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=total_steps,
            ),
            "interval": "step",
            "frequency": 1,
            "name": "learning_rate",
        }
        return [optimizer], [scheduler]

    def forward(self, **batch):
        return self.model(**batch)

    def training_step(self, batch, batch_idx):
        output = self(**batch)
        loss = output.loss
        preds = torch.argmax(output.logits, dim=1)
        acc = self.train_accuracy(preds, batch["labels"])

        self.log("train_loss", loss)
        self.log("train_accuracy", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        output = self(**batch)
        loss = output.loss
        preds = torch.argmax(output.logits, dim=1)
        acc = self.val_accuracy(preds, batch["labels"])

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_accuracy", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        output = self(**batch)
        loss = output.loss
        preds = torch.argmax(output.logits, dim=1)
        acc = self.test_accuracy(preds, batch["labels"])

        self.log("test_loss", loss)
        self.log("test_accuracy", acc)
