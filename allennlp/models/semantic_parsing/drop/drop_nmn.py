import logging
from typing import Any, List, Dict, Tuple
from overrides import overrides

import torch
from torch import Tensor

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
        self._decoder_beam_search = decoder_beam_search
        self._max_decoding_steps = max_decoding_steps
        self.drop_nmn_parameters = DropNmnParameters(passage_length=1290,
                                                     question_encoding_dim=self._question_encoder.get_output_dim(),
                                                     passage_encoding_dim=self._passage_encoder.get_output_dim(),
                                                     hidden_dim=self._question_encoder.get_output_dim(),
                                                     num_answers=4)

                                    

    @overrides
    def forward(self,
                question: Dict[str, torch.LongTensor],
                passage: Dict[str, torch.LongTensor],
                numbers_in_passage: Dict[str, torch.LongTensor],
                number_indices: torch.LongTensor,
                answer_as_passage_spans: torch.LongTensor = None,
                answer_as_question_spans: torch.LongTensor = None,
                answer_as_plus_minus_combinations: torch.LongTensor = None,
                answer_as_counts: torch.LongTensor = None,
                actions: List[List[ProductionRule]] = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ, unused-argument
        print('answer_as_passage_spans', answer_as_passage_spans.size())

        batch_size = len(actions)
        initial_rnn_state, encoder_outputs = self._get_initial_rnn_state(question)        

        initial_score_list = [next(iter(question.values())).new_zeros(1, dtype=torch.float)
                              for i in range(batch_size)]
        
        passage_mask = util.get_text_field_mask(passage).float()
        embedded_passage = self._passage_embedder(passage)
        encoded_passage = self._dropout(self._passage_encoder(embedded_passage, passage_mask))

        worlds = [[DropNmnLanguage(encoded_passage, self.drop_nmn_parameters)] for i in range(batch_size)]

        initial_grammar_state = [self._create_grammar_state(worlds[i][0], actions[i]) for i in
                                 range(batch_size)]

        initial_state = GrammarBasedState(batch_indices=list(range(batch_size)),
                                          action_history=[[] for _ in range(batch_size)],
                                          score=initial_score_list,
                                          rnn_state=initial_rnn_state,
                                          grammar_state=initial_grammar_state,
                                          possible_actions=actions)

        outputs: Dict[str, torch.Tensor] = {}

        initial_state.debug_info = [[] for _ in range(batch_size)]
        best_final_states = self._decoder_beam_search.search(self._max_decoding_steps,
                                                             initial_state,
                                                             self._decoder_step,
                                                             keep_final_unfinished_states=False)

        best_action_sequences: Dict[int, List[List[int]]] = {}
        for i in range(batch_size):
            # Decoding may not have terminated with any completed logical forms, if `num_steps`
            # isn't long enough (or if the model is not trained enough and gets into an
            # infinite action loop).
            if i in best_final_states:
                best_action_indices = [best_final_states[i][0].action_history[0]]
                best_action_sequences[i] = best_action_indices
        batch_action_strings = self._get_action_strings(actions, best_action_sequences)
        batch_denotations = self._get_denotations(batch_action_strings, worlds, best_final_states, encoder_outputs)

        print(batch_action_strings)
        print(batch_denotations)
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
        return initial_rnn_state, encoder_outputs

    @classmethod
    def _get_action_strings(cls,
                            possible_actions: List[List[ProductionRule]],
                            action_indices: Dict[int, List[List[int]]]) -> List[List[List[str]]]:
        """
        Takes a list of possible actions and indices of decoded actions into those possible actions
        for a batch and returns sequences of action strings. We assume ``action_indices`` is a dict
        mapping batch indices to k-best decoded sequence lists.
        """
        all_action_strings: List[List[List[str]]] = []
        batch_size = len(possible_actions)
        for i in range(batch_size):
            batch_actions = possible_actions[i]
            batch_best_sequences = action_indices[i] if i in action_indices else []
            # This will append an empty list to ``all_action_strings`` if ``batch_best_sequences``
            # is empty.
            action_strings = [[batch_actions[rule_id][0] for rule_id in sequence]
                              for sequence in batch_best_sequences]
            all_action_strings.append(action_strings)
        return all_action_strings

    @staticmethod
    def _get_denotations(action_strings: List[List[List[str]]],
                         worlds: List[List[DropNmnLanguage]],
                         best_final_states,
                         encoder_outputs: Tensor) -> List[List[List[str]]]:
        all_denotations: List[List[List[str]]] = []
        for idx, (instance_worlds, instance_action_sequences) in enumerate(zip(worlds, action_strings)):
            denotations: List[List[str]] = []
            for instance_action_strings in instance_action_sequences:
                if not instance_action_strings:
                    continue
                logical_form = instance_worlds[0].action_sequence_to_logical_form(instance_action_strings)
                instance_denotations: List[str] = []
                for world in instance_worlds:
                    # Some of the worlds can be None for instances that come with less than 4 worlds
                    # because of padding.
                    if world is not None:
                        attended_questions = [{'attended_question': util.weighted_sum(encoder_outputs[idx],
                                                                                      debug_info['question_attention'])}
                                              for debug_info in best_final_states[idx][0].debug_info[0]]
                        instance_denotations.append(str(world.execute_action_sequence(instance_action_strings,
                                                                                      attended_questions)))
                denotations.append(instance_denotations)
            all_denotations.append(denotations)
        return all_denotations
