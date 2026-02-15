import torch
from torch import nn

from transformers import RobertaModel
from transformers.modeling_outputs import SequenceClassifierOutput

from mechanistic_interpretability.utils.build_model import build_model
from mechanistic_interpretability.models.encoder_mlp import EncoderMLP


class ChembertaRegressorWithFeatures(nn.Module):
    """
    ChemBERTa regression model with optional numerical features.
    """

    def __init__(
        self,
        pretrained,
        num_features=0,
        dropout=0.3,
        hidden_channels=100,
        num_mlp_layers=1,
        num_labels = 138,
        verbose=0,
    ):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained(pretrained, add_pooling_layer=False)
        # Freeze the base language model to avoid overfitting. 
        # Currently commented for better performance.
        # TODO: Choose which parts of the model actually require finetuning and freeze all other parts.
        # for param in self.roberta.parameters():
        #     param.requires_grad = False
        hidden_size = self.roberta.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.verbose_level = verbose

        # Handle the case with no numerical features
        self.has_features = num_features > 0
        if self.has_features:
            # Create the numerical branch next to ChemBERTa branch
            self.num_encoder = EncoderMLP(
                input_dim=num_features,
                hidden_channels=hidden_channels,
                num_layers=num_mlp_layers,
                output_dim=num_labels,
                dropout=dropout,
            )
            num_input_features = hidden_size * 3  # cls, feat, and product
        else:
            num_input_features = hidden_size * 2 # we have cls and mean poole

        self.mlp = build_model(
            model_name="mlp",
            hidden_channels=hidden_channels,
            num_numerical_features=num_input_features,
            num_mlp_layers=num_mlp_layers,
            output_dim=num_labels,
            dropout=dropout,
        )
        print("mlp architecture:")
        print(
            {
                "hidden_channels": hidden_channels,
                "num_numerical_features": num_input_features,
                "num_mlp_layers": num_mlp_layers,
                "dropout": dropout,
                'num_labels': num_labels,
            }
        )

    def forward(self, input_ids=None, attention_mask=None, labels=None, features=None):
        try:
            # Debug input shapes
            if input_ids is None:
                print("WARNING: input_ids is None in forward pass")

            # Call the base model
            outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
            cls_emb = outputs.last_hidden_state[:, 0, :]
            #print(f"cls_emb.shape: {cls_emb.shape}, cls_emb: {cls_emb}")

            # Handle different feature availability
            if self.has_features and features is not None:
                # Use the encoder MLP
                feat_emb = self.num_encoder(features)
                x = torch.cat([cls_emb, feat_emb, cls_emb * feat_emb], dim=1)
            else:
                x = cls_emb

            x = self.dropout(x)
            logits = self.mlp(x).squeeze(-1)
            loss = None
            if labels is not None:
                criterion = nn.MSELoss()
                raw_loss = criterion(logits, labels)
                # Print the raw MSE loss value for debugging
                if self.verbose_level > 1:
                    print(f"DEBUG - Raw MSE Loss: {raw_loss.item():.6f} | \
                           Mean logits: {logits.mean().item():.6f} | \
                           Mean labels: {labels.mean().item():.6f}")

                # This is the loss that will be used for training
                loss = raw_loss
            # We return this class of output for the Trainer to work
            # To get the last hidden state, call 
            # ChembertaRegressorWithFeatures.roberta(input_ids, attention_mask).last_hidden_state
            return SequenceClassifierOutput(loss=loss, logits=logits.unsqueeze(-1))
        except Exception as e:
            print(f"Error in model forward pass: {e}")
            print(f"Input types: input_ids={type(input_ids)}, attention_mask={type(attention_mask)}")
            if input_ids is not None:
                print(f"Input shapes: input_ids={input_ids.shape}")
            if attention_mask is not None:
                print(f"attention_mask={attention_mask.shape}")
            if features is not None:
                print(f"Features shape: {features.shape}")
            raise