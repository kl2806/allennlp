import logging
from typing import Any, List, Dict, Tuple
from overrides import overrides

import torch

from allennlp.data.fields.production_rule_field import ProductionRule
from allennlp.data.vocabulary import Vocabulary
from allennlp.models import Model
from allennlp.modules import Attention, TextFieldEmbedder, Seq2SeqEncoder, Embedding
from allennlp.nn import Activation
from allennlp.nn import util
from allennlp.semparse.domain_languages import DropNmnLanguage, START_SYMBOL
from allennlp.semparse.domain_languages.drop_nmn_language import DropNmnParameters
from allennlp.state_machines.states import GrammarBasedState, RnnStatelet, GrammarStatelet
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
                 rule_namespace: str = 'rule_labels') -> None:
        super().__init__(vocab)
        self._passage_embedder = passage_embedder
        self._question_embedder = question_embedder
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
        self._first_action_embedding = torch.nn.Parameter(torch.FloatTensor(action_embedding_dim))
        torch.nn.init.normal_(self._first_action_embedding)
        self._decoder_step = BasicTransitionFunction(encoder_output_dim=self._question_encoder.get_output_dim(),
                                                     action_embedding_dim=action_embedding_dim,
                                                     input_attention=attention,
                                                     num_start_types=1,
                                                     activation=Activation.by_name('tanh')(),
                                                     predict_start_type_separately=False,
                                                     add_action_bias=False,
                                                     dropout=dropout)


    @overrides
    def forward(self,
                question: Dict[str, torch.LongTensor],
                passage: Dict[str, torch.LongTensor],
                span_start: torch.IntTensor = None,
                span_end: torch.IntTensor = None,
                actions: List[List[ProductionRule]] = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ, unused-argument
        batch_size = len(actions)
        initial_rnn_state = self._get_initial_rnn_state(question)        

        initial_score_list = [next(iter(question.values())).new_zeros(1, dtype=torch.float)
                              for i in range(batch_size)]

        drop_nmn_language = DropNmnLanguage(None, None)
        initial_grammar_state = [self._create_grammar_state(drop_nmn_language, actions[i]) for i in
                                 range(batch_size)]

        raise NotImplementedError

    def _create_grammar_state(self,
                              world: DropNmnLanguage,
                              possible_actions: List[ProductionRule]) -> GrammarStatelet:
        valid_actions = world.get_nonterminal_productions()
        action_mapping = {}
        for i, action in enumerate(possible_actions):
            action_mapping[action[0]] = i
        translated_valid_actions: Dict[str, Dict[str, Tuple[torch.Tensor, torch.Tensor, List[int]]]] = {}
        for key, action_strings in valid_actions.items():
            translated_valid_actions[key] = {}
            # `key` here is a non-terminal from the grammar, and `action_strings` are all the valid
            # productions of that non-terminal.
            action_indices = [action_mapping[action_string] for action_string in action_strings]
            # All actions in NLVR are global actions.
            global_actions = [(possible_actions[index][2], index) for index in action_indices]

            # Then we get the embedded representations of the global actions.
            global_action_tensors, global_action_ids = zip(*global_actions)
            global_action_tensor = torch.cat(global_action_tensors, dim=0)
            global_input_embeddings = self._action_embedder(global_action_tensor)
            translated_valid_actions[key]['global'] = (global_input_embeddings,
                                                       global_input_embeddings,
                                                       list(global_action_ids))
        return GrammarStatelet([START_SYMBOL],
                               translated_valid_actions,
                               world.is_nonterminal) 

    def _get_initial_rnn_state(self, question: Dict[str, torch.LongTensor]):
        embedded_input = self._question_embedder(question)
        # (batch_size, question_length)
        question_mask = util.get_text_field_mask(question).float()

        batch_size = embedded_input.size(0)

        # (batch_size, question_length, encoder_output_dim)
        encoder_outputs = self._dropout(self._question_encoder(embedded_input, question_mask))

        final_encoder_output = util.get_final_encoder_states(encoder_outputs,
                                                             question_mask,
                                                             self._question_encoder.is_bidirectional())
        memory_cell = encoder_outputs.new_zeros(batch_size, self._question_encoder.get_output_dim())
        attended_sentence, _ = self._decoder_step.attend_on_question(final_encoder_output,
                                                                     encoder_outputs, question_mask)
        encoder_outputs_list = [encoder_outputs[i] for i in range(batch_size)]
        question_mask_list = [question_mask[i] for i in range(batch_size)]
        initial_rnn_state = []
        for i in range(batch_size):
            initial_rnn_state.append(RnnStatelet(final_encoder_output[i],
                                                 memory_cell[i],
                                                 self._first_action_embedding,
                                                 attended_sentence[i],
                                                 encoder_outputs_list,
                                                 question_mask_list))
        return initial_rnn_state
