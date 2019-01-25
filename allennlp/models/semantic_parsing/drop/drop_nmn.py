import logging
from typing import Any, List, Dict
from overrides import overrides

import torch

from allennlp.data.fields.production_rule_field import ProductionRule
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules import Attention, TextFieldEmbedder, Seq2SeqEncoder, Embedding
from allennlp.nn import Activation
from allennlp.nn import util
from allennlp.semparse.domain_languages import DropNmnLanguage
from allennlp.state_machines.states import GrammarBasedState
from allennlp.state_machines.trainers import MaximumMarginalLikelihood
from allennlp.state_machines.transition_functions import BasicTransitionFunction
from allennlp.state_machines import BeamSearch


logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

@Model.register('drop-nmn')
class DropModuleNetwork(Model):
    """
    ``DropModuleNetwork`` is a model for DROP that has a learned execution model.
    """
    def __init__(self,
                 vocab: Vocabulary,
                 passage_embedder: TextFieldEmbedder,
                 question_embedder: TextFieldEmbedder,
                 action_embedding_dim: int,
                 passage_encoder: Seq2SeqEncoder,
                 question_encoder: Seq2SeqEncoder,
                 attention: Attention,
                 decoder_beam_search: BeamSearch,
                 max_decoding_steps: int,
                 dropout: float = 0.0,
                 rule_namespace = 'rule_labels') -> None:
        super().__init__(vocab)
        self.passage_embedder = passage_embedder
        self.question_embedder = question_embedder
        self._action_embedding_dim = action_embedding_dim
        self._passage_encoder = passage_encoder 
        self._question_encoder = question_encoder 
        self._rule_namespace = rule_namespace

        if dropout > 0:
            self._dropout = torch.nn.Dropout(p=dropout)
        else:
            self._dropout = lambda x: x
        self._rule_namespace = rule_namespace
        self._action_embedder = Embedding(num_embeddings=vocab.get_vocab_size(self._rule_namespace),
                                          embedding_dim=self._action_embedding_dim)

    @overrides
    def forward(self,
                question: Dict[str, torch.LongTensor],
                passage: Dict[str, torch.LongTensor],
                span_start: torch.IntTensor = None,
                span_end: torch.IntTensor = None,
                actions: List[List[ProductionRule]] = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ, unused-argument
        raise NotImplementedError
        

