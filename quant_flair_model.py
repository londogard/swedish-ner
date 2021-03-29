from typing import Union
from pathlib import Path
import torch
import flair

class QuantizableLanguageModel(flair.models.LanguageModel):
    def forward(self, input, hidden, ordered_sequence_lengths=None):
        encoded = self.encoder(input)
        emb = self.drop(encoded)

        if hasattr(self.rnn, "flatten_parameters"):
            self.rnn.flatten_parameters()

        output, hidden = self.rnn(emb, hidden)
        if self.proj is not None:
            output = self.proj(output)
        output = self.drop(output)
        decoded = self.decoder(
            output.view(output.size(0) * output.size(1), output.size(2))
        )
        return (
            decoded.view(output.size(0), output.size(1), decoded.size(1)),
            output,
            hidden,
        )

    @classmethod
    def load_language_model(cls, model_file: Union[Path, str]):
        state = torch.load(str(model_file), map_location=flair.device)
        document_delimiter = state["document_delimiter"] if "document_delimiter" in state else '\n'
        
        model = cls(
            dictionary=state["dictionary"],
            is_forward_lm=state["is_forward_lm"],
            hidden_size=state["hidden_size"],
            nlayers=state["nlayers"],
            embedding_size=state["embedding_size"],
            nout=state["nout"],
            document_delimiter=document_delimiter,
            dropout=state["dropout"],
        )
        model.load_state_dict(state["state_dict"])
        model.eval()
        model.to(flair.device)

        return model