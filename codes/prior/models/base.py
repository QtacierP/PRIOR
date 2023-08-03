import pytorch_lightning as pl


class PretrainModel(pl.LightningModule):
    def __init__(self, text_encoder, image_encoder) -> None:
        super().__init__()
        self.text_encoder = text_encoder
        self.image_encoder = image_encoder
        self.configure_loss()
    
    def configure_transform(self):
        pass

    def configure_loss(self):
        pass


