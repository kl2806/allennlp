from typing import Any, Dict, List, Optional
import logging

import torch

from allennlp.data import Vocabulary
from allennlp.data.fields.production_rule_field import ProductionRule
from allennlp.models.model import Model
from allennlp.models.reading_comprehension.util import get_best_span
from allennlp.modules import Highway
from allennlp.nn.activations import Activation
from allennlp.modules.feedforward import FeedForward
from allennlp.modules import Attention, Seq2SeqEncoder, TextFieldEmbedder, Embedding
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.modules.matrix_attention.matrix_attention import MatrixAttention
from allennlp.nn import util, InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import masked_softmax
from allennlp.semparse.domain_languages import NaqanetParameters, DropNaqanetLanguage, START_SYMBOL
from allennlp.state_machines.states import GrammarStatelet, RnnStatelet, GrammarBasedState
from allennlp.state_machines import BeamSearch
from allennlp.training.metrics.drop_em_and_f1 import DropEmAndF1

from allennlp.state_machines.transition_functions import LinkingTransitionFunction


logger = logging.getLogger(__name__)


@Model.register("semparse-naqanet")
class SemparseNumericallyAugmentedQaNet(Model):
    """
    This class augments the QANet model with some rudimentary numerical reasoning abilities, as
    published in the original DROP paper.

    The main idea here is that instead of just predicting a passage span after doing all of the
    QANet modeling stuff, we add several different "answer abilities": predicting a span from the
    question, predicting a count, or predicting an arithmetic expression.  Near the end of the
    QANet model, we have a variable that predicts what kind of answer type we need, and each branch
    has separate modeling logic to predict that answer type.  We then marginalize over all possible
    ways of getting to the right answer through each of these answer types.
    """
    def __init__(self, vocab: Vocabulary,
                 text_field_embedder: TextFieldEmbedder,
                 num_highway_layers: int,
                 phrase_layer: Seq2SeqEncoder,
                 matrix_attention_layer: MatrixAttention,
                 modeling_layer: Seq2SeqEncoder,
                 action_embedding_dim: int,
                 input_attention: Attention,
                 decoder_beam_search : BeamSearch,
                 dropout_prob: float = 0.1,
                 add_action_bias: bool = False,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None,
                 answering_abilities: List[str] = None,
                 rule_namespace: str = 'rule_labels',
                 modeling_dim = 128,
                 max_decoding_steps: int = 2) -> None:
        super().__init__(vocab, regularizer)


        if answering_abilities is None:
            self.answering_abilities = ["passage_span_extraction", "question_span_extraction",
                                        "addition_subtraction", "counting"]
        else:
            self.answering_abilities = answering_abilities

        text_embed_dim = text_field_embedder.get_output_dim()
        encoding_in_dim = phrase_layer.get_input_dim()
        encoding_out_dim = phrase_layer.get_output_dim()
        modeling_in_dim = modeling_layer.get_input_dim()
        modeling_out_dim = modeling_layer.get_output_dim()

        self.encoding_in_dim = encoding_in_dim
        self.encoding_out_dim = encoding_out_dim

        self._text_field_embedder = text_field_embedder

        self._embedding_proj_layer = torch.nn.Linear(text_embed_dim, encoding_in_dim)
        self._highway_layer = Highway(encoding_in_dim, num_highway_layers)

        self._encoding_proj_layer = torch.nn.Linear(encoding_in_dim, encoding_in_dim)
        self._phrase_layer = phrase_layer

        self._matrix_attention = matrix_attention_layer

        self._modeling_proj_layer = torch.nn.Linear(encoding_out_dim * 4, modeling_in_dim)
        self._modeling_layer = modeling_layer

        self._passage_weights_predictor = torch.nn.Linear(modeling_out_dim, 1)
        self._question_weights_predictor = torch.nn.Linear(encoding_out_dim, 1)

        self._drop_metrics = DropEmAndF1()
        self._dropout = torch.nn.Dropout(p=dropout_prob)

        self._rule_namespace = rule_namespace
        self._add_action_bias = add_action_bias

        num_actions = vocab.get_vocab_size(self._rule_namespace)

        self.naqanet_parameters = NaqanetParameters(modeling_dim)
        self._first_action_embedding = torch.nn.Parameter(torch.FloatTensor(action_embedding_dim))


        if self._add_action_bias:
            self._action_biases = Embedding(num_embeddings=num_actions, embedding_dim=1)
        self._action_embedder = Embedding(num_embeddings=num_actions, embedding_dim=action_embedding_dim)

        self._decoder_beam_search = decoder_beam_search
        self._max_decoding_steps = max_decoding_steps
        
        self._transition_function = LinkingTransitionFunction(encoder_output_dim=encoding_out_dim,
                                                              action_embedding_dim=action_embedding_dim,
                                                              input_attention=input_attention,
                                                              predict_start_type_separately=False,
                                                              add_action_bias=self._add_action_bias,
                                                              dropout=dropout_prob)



        initializer(self)

    def forward(self,  # type: ignore
                question: Dict[str, torch.LongTensor],
                passage: Dict[str, torch.LongTensor],
                number_indices: torch.LongTensor,
                answer_as_passage_spans: torch.LongTensor = None,
                answer_as_question_spans: torch.LongTensor = None,
                answer_as_add_sub_expressions: torch.LongTensor = None,
                answer_as_counts: torch.LongTensor = None,
                actions: List[List[ProductionRule]] = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        question_mask = util.get_text_field_mask(question).float()
        passage_mask = util.get_text_field_mask(passage).float()
        embedded_question = self._dropout(self._text_field_embedder(question))
        embedded_passage = self._dropout(self._text_field_embedder(passage))
        embedded_question = self._highway_layer(self._embedding_proj_layer(embedded_question))
        embedded_passage = self._highway_layer(self._embedding_proj_layer(embedded_passage))

        batch_size = embedded_question.size(0)

        projected_embedded_question = self._encoding_proj_layer(embedded_question)
        projected_embedded_passage = self._encoding_proj_layer(embedded_passage)
            
        # Shape: (batch_size, question_length, encoding_dim)
        encoded_question = self._dropout(self._phrase_layer(projected_embedded_question, question_mask))
        # Shape: (batch_size, passage_length, encoding_dim)
        encoded_passage = self._dropout(self._phrase_layer(projected_embedded_passage, passage_mask))

        # Shape: (batch_size, passage_length, question_length)
        passage_question_similarity = self._matrix_attention(encoded_passage, encoded_question)
        # Shape: (batch_size, passage_length, question_length)
        passage_question_attention = masked_softmax(passage_question_similarity,
                                                    question_mask,
                                                    memory_efficient=True)
        # Shape: (batch_size, passage_length, encoding_dim)
        passage_question_vectors = util.weighted_sum(encoded_question, passage_question_attention)

        # Shape: (batch_size, question_length, passage_length)
        question_passage_attention = masked_softmax(passage_question_similarity.transpose(1, 2),
                                                    passage_mask,
                                                    memory_efficient=True)

        # Shape: (batch_size, passage_length, passage_length)
        passsage_attention_over_attention = torch.bmm(passage_question_attention, question_passage_attention)
        # Shape: (batch_size, passage_length, encoding_dim)
        passage_passage_vectors = util.weighted_sum(encoded_passage, passsage_attention_over_attention)

        # Shape: (batch_size, passage_length, encoding_dim * 4)
        merged_passage_attention_vectors = self._dropout(
                torch.cat([encoded_passage, passage_question_vectors,
                           encoded_passage * passage_question_vectors,
                           encoded_passage * passage_passage_vectors],
                          dim=-1))

        # The recurrent modeling layers. Since these layers share the same parameters,
        # we don't construct them conditioned on answering abilities.
        modeled_passage_list = [self._modeling_proj_layer(merged_passage_attention_vectors)]
        for _ in range(4):
            modeled_passage = self._dropout(self._modeling_layer(modeled_passage_list[-1], passage_mask))
            modeled_passage_list.append(modeled_passage)
        # Pop the first one, which is input
        modeled_passage_list.pop(0)

        # The first modeling layer is used to calculate the vector representation of passage
        passage_weights = self._passage_weights_predictor(modeled_passage_list[0]).squeeze(-1)
        passage_weights = masked_softmax(passage_weights, passage_mask)
        passage_vector = util.weighted_sum(modeled_passage_list[0], passage_weights)
        # The vector representation of question is calculated based on the unmatched encoding,
        # because we may want to infer the answer ability only based on the question words.

        question_weights = self._question_weights_predictor(encoded_question).squeeze(-1)
        question_weights = masked_softmax(question_weights, question_mask)
        question_vector = util.weighted_sum(encoded_question, question_weights)

        worlds = [[DropNaqanetLanguage(encoded_question=encoded_question[i].unsqueeze(0), 
                                       question_mask=question_mask[i].unsqueeze(0),
                                       passage_vector=passage_vector[i].unsqueeze(0),
                                       passage_mask=passage_mask[i].unsqueeze(0),
                                       modeled_passage_list=[passage[i].unsqueeze(0) for passage in modeled_passage_list],
                                       number_indices=number_indices[0].unsqueeze(0), 
                                       parameters=self.naqanet_parameters)] for i in range(batch_size)]
        
        initial_rnn_state = [] 
        encoded_question_list = [encoded_question[i] for i in range(batch_size)]
        question_mask_list = [question_mask[i] for i in range(batch_size)]
        memory_cell = encoded_question.new_zeros(batch_size, self.encoding_out_dim) 

        final_encoder_output = util.get_final_encoder_states(encoded_question,
                                                             question_mask)
        
        for i in range(batch_size):
            initial_rnn_state.append(RnnStatelet(hidden_state=final_encoder_output[i],
                                                 memory_cell=memory_cell[i], 
                                                 previous_action_embedding=self._first_action_embedding,
                                                 attended_input=question_vector[i],
                                                 encoder_outputs=encoded_question_list,
                                                 encoder_output_mask=question_mask_list))

        initial_score_list = [next(iter(question.values())).new_zeros(1, dtype=torch.float)
                              for i in range(batch_size)]

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
        final_states = self._decoder_beam_search.search(self._max_decoding_steps,
                                                        initial_state,
                                                        self._transition_function,
                                                        keep_final_unfinished_states=False)

        action_mapping = {}
        for batch_index, batch_actions in enumerate(actions):
            for action_index, action in enumerate(batch_actions):
                action_mapping[(batch_index, action_index)] = action[0]
        

        state_scores = []
        answers = []
        log_marginal_likelihood_list = []
        best_final_answers = []

        for i in range(batch_size):
            for state_index, state in enumerate(final_states[i]):
                state_scores.append(state.score[0])

                # Execute each program to get the loss
                action_indices = state.action_history[0]
                action_strings = [action_mapping[(i, action_index)]
                              for action_index in action_indices]
                world = worlds[i][0]
                answer = world.execute_action_sequence(action_strings)

                answers.append(answer)
                log_prob = answer.get_answer_log_prob(answer_as_passage_spans[i].unsqueeze(0),
                                                      answer_as_question_spans[i].unsqueeze(0),
                                                      answer_as_counts[i].unsqueeze(0),
                                                      answer_as_add_sub_expressions[i].unsqueeze(0),
                                                      number_indices[i].unsqueeze(0))

                log_marginal_likelihood_list.append(log_prob)

                
                if state_index == 0:
                    best_final_answers.append(answer)
        scores = torch.stack(state_scores)

        all_log_marginal_likelihoods = torch.stack(log_marginal_likelihood_list, dim=-1)
        all_log_marginal_likelihoods = all_log_marginal_likelihoods + scores 
        marginal_log_likelihood = util.logsumexp(all_log_marginal_likelihoods)

        output_dict = {}
        output_dict["loss"] = -marginal_log_likelihood.mean()
        
        # best_final_answers = answers
        # Compute the metrics and add the tokenized input to the output.
        if metadata is not None:
            output_dict["question_id"] = []
            output_dict["answer"] = []
            question_tokens = []
            passage_tokens = []
            for i in range(batch_size):
                question_tokens.append(metadata[i]['question_tokens'])
                passage_tokens.append(metadata[i]['passage_tokens'])
        
                predicted_answer, answer_json = best_final_answers[i].get_best_answer(metadata[i])

                output_dict["question_id"].append(metadata[i]["question_id"])
                output_dict["answer"].append(answer_json)
                answer_annotations = metadata[i].get('answer_annotations', [])
                if answer_annotations:
                    self._drop_metrics(predicted_answer, answer_annotations)
            # This is used for the demo.
            output_dict["passage_question_attention"] = passage_question_attention
            output_dict["question_tokens"] = question_tokens
            output_dict["passage_tokens"] = passage_tokens
        return output_dict

    def _create_grammar_state(self,
                              world: DropNaqanetLanguage,
                              # TODO(mattg): just remove the need for this input
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
            if self._add_action_bias:
                global_action_biases = self._action_biases(global_action_tensor)
                global_input_embeddings = torch.cat([global_input_embeddings, global_action_biases], dim=-1)
            translated_valid_actions[key]['global'] = (global_input_embeddings,
                                                       global_input_embeddings,
                                                       list(global_action_ids))
        return GrammarStatelet([START_SYMBOL],
                               translated_valid_actions,
                               world.is_nonterminal)
        
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        exact_match, f1_score = self._drop_metrics.get_metric(reset)
        return {'em': exact_match, 'f1': f1_score}
