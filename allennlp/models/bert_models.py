from typing import Dict, Optional, List, Any

from overrides import overrides
from pytorch_pretrained_bert.modeling import BertModel
import torch
from torch.nn.modules.linear import Linear

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.nn import RegularizerApplicator
import allennlp.nn.util as util
from allennlp.training.metrics import CategoricalAccuracy


@Model.register("bert_mc_qa")
class BertMCQAModel(Model):
    """

    """
    def __init__(self,
                 vocab: Vocabulary,
                 pretrained_model: str,
                 num_labels: int,
                 requires_grad: bool = True,
                 # top_layer_only: bool = False,
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self._bert_model = BertModel.from_pretrained(pretrained_model)
        for param in self._bert_model.parameters():
            param.requires_grad = requires_grad

        self._output_dim = self._bert_model.config.hidden_size
        self._classifier = Linear(self._output_dim, num_labels)
        self._accuracy = CategoricalAccuracy()
        self._loss = torch.nn.CrossEntropyLoss()

    def forward(self,
                question: Dict[str, torch.LongTensor],
                segment_ids: torch.LongTensor = None,
                label: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> torch.Tensor:

        input_ids = question['tokens']
        question_mask = (input_ids != 0).long()
        _, pooled_output = self._bert_model(util.combine_initial_dims(input_ids),
                                            util.combine_initial_dims(question_mask),
                                            util.combine_initial_dims(segment_ids),
                                            output_all_encoded_layers=False)

        label_logits = self._classifier(pooled_output)
        output_dict = {}
        output_dict['label_logits'] = label_logits

        if label is not None:
            loss = self._loss(label_logits, label.long().view(-1))
            self._accuracy(label_logits, label)
            output_dict["loss"] = loss

        return output_dict

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {
            'accuracy': self._accuracy.get_metric(reset),
        }








