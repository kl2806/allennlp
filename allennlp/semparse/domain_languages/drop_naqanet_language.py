from typing import NamedTuple, Tuple, Dict, Any, List

import torch

from allennlp.common.util import JsonDict
from allennlp.nn import util, Activation
from allennlp.semparse.domain_languages.domain_language import DomainLanguage, predicate




class NaqanetParameters(torch.nn.Module):
    """
    Stores all of the parameters necessary for the various learned functions in the
    ``DropNaqanetLanguage``.  This needs to be constructed outside the ``Language`` object,
    because we create one of those objects per ``Instance``, while the parameters are shared across
    all ``Instances``.  This also needs to be a ``torch.nn.Module``, so that your ``Model`` can
    save it as a member variable and have its weights get updated.
    """
    def __init__(self, modeling_dim: int) -> None:

        from allennlp.modules import FeedForward
        super().__init__()
        self.passage_span_start_predictor = FeedForward(modeling_dim * 2,
                                                        activations=[Activation.by_name('relu')(),
                                                                     Activation.by_name('linear')()],
                                                        hidden_dims=[modeling_dim, 1],
                                                        num_layers=2)
        self.passage_span_end_predictor = FeedForward(modeling_dim * 2,
                                                      activations=[Activation.by_name('relu')(),
                                                                   Activation.by_name('linear')()],
                                                      hidden_dims=[modeling_dim, 1],
                                                      num_layers=2)
        self.question_span_start_predictor = FeedForward(modeling_dim * 2,
                                                         activations=[Activation.by_name('relu')(),
                                                                      Activation.by_name('linear')()],
                                                         hidden_dims=[modeling_dim, 1],
                                                         num_layers=2)
        self.question_span_end_predictor = FeedForward(modeling_dim * 2,
                                                       activations=[Activation.by_name('relu')(),
                                                                    Activation.by_name('linear')()],
                                                       hidden_dims=[modeling_dim, 1],
                                                       num_layers=2)
        self.number_sign_predictor = FeedForward(modeling_dim * 3,
                                                 activations=[Activation.by_name('relu')(),
                                                              Activation.by_name('linear')()],
                                                 hidden_dims=[modeling_dim, 3],
                                                 num_layers=2)
        self.count_number_predictor = FeedForward(modeling_dim,
                                                  activations=[Activation.by_name('relu')(),
                                                               Activation.by_name('linear')()],
                                                  hidden_dims=[modeling_dim, 10],
                                                  num_layers=2)


class Answer(NamedTuple):
    passage_span: Tuple[torch.Tensor, torch.Tensor] = None
    question_span: Tuple[torch.Tensor, torch.Tensor] = None
    count_answer: torch.Tensor = None
    arithmetic_answer: torch.Tensor = None
    number_indices: torch.LongTensor = None

    def get_answer_log_prob(self,
                            answer_as_passage_span: torch.LongTensor,
                            answer_as_question_span: torch.LongTensor,
                            answer_as_count: torch.LongTensor,
                            answer_as_arithmetic_expression: torch.LongTensor,
                            number_indices: torch.LongTensor) -> torch.Tensor:
        """
        Given a supervision signal (which so far is basically just correct indices in the
        distributions that our various answer types capture), returns the log probability of the
        answer.
        """
        log_prob = None
        if answer_as_passage_span is not None and self.passage_span is not None:
            assert log_prob is None, "Found multiple answer types in a single Answer"
            log_prob = self._get_span_answer_log_prob(answer_as_passage_span, self.passage_span)
        if answer_as_question_span is not None and self.question_span is not None:
            assert log_prob is None, "Found multiple answer types in a single Answer"
            log_prob = self._get_span_answer_log_prob(answer_as_question_span, self.question_span)
        if answer_as_count is not None and self.count_answer is not None:
            assert log_prob is None, "Found multiple answer types in a single Answer"
            log_prob = self._get_count_answer_log_prob(answer_as_count)
        if answer_as_arithmetic_expression is not None and self.arithmetic_answer is not None:
            assert log_prob is None, "Found multiple answer types in a single Answer"
            log_prob = self._get_arithmetic_answer_log_prob(answer_as_arithmetic_expression, number_indices)
        assert log_prob is not None, "Didn't find an answer matching the given supervision"
        return log_prob

    def get_best_answer(self, metadata: Dict[str, Any]) -> Tuple[str, JsonDict]:
        """
        Does an argmax decoding of whatever answer type is represented by this ``Answer``,
        returning both a string and a ``JsonDict`` representation of the answer.
        """
        answer_json = None
        answer_string = None
        if self.passage_span is not None:
            answer_json = self._get_best_span(self.passage_span[0],
                                              self.passage_span[1],
                                              metadata['original_passage'],
                                              metadata['passage_token_offsets'])
            answer_json["answer_type"] = "passage_span"
            answer_string = answer_json["value"]
        if self.question_span is not None:
            answer_json = self._get_best_span(self.question_span[0],
                                              self.question_span[1],
                                              metadata['original_question'],
                                              metadata['question_token_offsets'])
            answer_json["answer_type"] = "question_span"
            answer_string = answer_json["value"]
        if self.count_answer is not None:
            answer_json = self._get_best_count()
            answer_string = str(answer_json["count"])
        if self.arithmetic_answer is not None:
            answer_json = self._get_best_arithmetic_expression(metadata['original_numbers'],
                                                               metadata['passage_token_offsets'],
                                                               metadata['number_indices'])
            answer_string = str(answer_json["value"])
        return answer_string, answer_json

    def _get_span_answer_log_prob(self,
                                  answer: torch.LongTensor,
                                  span_log_probs: Tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        # Shape: (batch_size, # of answer spans)
        gold_span_starts = answer[:, :, 0]
        gold_span_ends = answer[:, :, 1]

        span_start_log_probs, span_end_log_probs = span_log_probs
        # Some spans are padded with index -1,
        # so we clamp those paddings to 0 and then mask after `torch.gather()`.
        gold_span_mask = (gold_span_starts != -1).long()
        clamped_gold_span_starts = util.replace_masked_values(gold_span_starts, gold_span_mask, 0)
        clamped_gold_span_ends = util.replace_masked_values(gold_span_ends, gold_span_mask, 0)
        # Shape: (batch_size, # of answer spans)
        start_log_likelihood = torch.gather(span_start_log_probs, 1, clamped_gold_span_starts)
        end_log_likelihood = torch.gather(span_end_log_probs, 1, clamped_gold_span_ends)
        # Shape: (batch_size, # of answer spans)
        log_likelihood = start_log_likelihood + end_log_likelihood
        # For those padded spans, we set their log probabilities to be very small negative value
        log_likelihood = util.replace_masked_values(log_likelihood, gold_span_mask, -1e7)
        # Shape: (batch_size, )
        return util.logsumexp(log_likelihood)

    def _get_count_answer_log_prob(self, answer: torch.LongTensor) -> torch.Tensor:
        # Count answers are padded with label -1,
        # so we clamp those paddings to 0 and then mask after `torch.gather()`.
        # Shape: (batch_size, # of count answers)
        gold_count_mask = (answer != -1).long()
        # Shape: (batch_size, # of count answers)
        clamped_gold_counts = util.replace_masked_values(answer, gold_count_mask, 0)
        log_likelihood = torch.gather(self.count_answer, 1, clamped_gold_counts)
        # For those padded spans, we set their log probabilities to be very small negative value
        log_likelihood = util.replace_masked_values(log_likelihood, gold_count_mask, -1e7)
        # Shape: (batch_size, )
        return util.logsumexp(log_likelihood)

    def _get_arithmetic_answer_log_prob(self, answer: torch.LongTensor, number_indices: torch.LongTensor) -> torch.Tensor:
        number_indices = number_indices.squeeze(-1)
        number_mask = (number_indices != -1).long()

        # The padded add-sub combinations use 0 as the signs for all numbers, and we mask them here.
        # Shape: (batch_size, # of combinations)
        gold_add_sub_mask = (answer.sum(-1) > 0).float()
        # Shape: (batch_size, # of numbers in the passage, # of combinations)
        gold_add_sub_signs = answer.transpose(1, 2)
        # Shape: (batch_size, # of numbers in the passage, # of combinations)
        sign_log_likelihood = torch.gather(self.arithmetic_answer, 2, gold_add_sub_signs)
        # the log likelihood of the masked positions should be 0
        # so that it will not affect the joint probability
        sign_log_likelihood = util.replace_masked_values(sign_log_likelihood, number_mask.unsqueeze(-1), 0)
        # Shape: (batch_size, # of combinations)
        log_likelihood = sign_log_likelihood.sum(1)
        # For those padded combinations, we set their log probabilities to be very small negative value
        log_likelihood = util.replace_masked_values(log_likelihood, gold_add_sub_mask, -1e7)
        # Shape: (batch_size, )
        return util.logsumexp(log_likelihood)

    def _get_best_span(self,
                       span_start_log_probs: torch.Tensor,
                       span_end_log_probs: torch.Tensor,
                       original_text: str,
                       offsets: List[Tuple[int, int]]) -> JsonDict:

        from allennlp.models.reading_comprehension.util import get_best_span
        answer_json = {}
        # Shape: (batch_size, 2)
        best_passage_span = get_best_span(span_start_log_probs, span_end_log_probs)
        
        predicted_span = tuple(best_passage_span[0].detach().cpu().numpy())
        
        if predicted_span[0] < 0 or predicted_span[1] >= len(offsets) or predicted_span[1] < 1 or predicted_span[0] >=  len(offsets):
            print('predicted_span', predicted_span)
            print('offsets', offsets)

        start_offset = offsets[predicted_span[0]][0]
        end_offset = offsets[predicted_span[1]][1]
        answer_json["value"] = original_text[start_offset:end_offset]
        answer_json["spans"] = [(start_offset, end_offset)]
        return answer_json

    def _get_best_count(self) -> JsonDict:
        answer_json = {}
        answer_json["answer_type"] = "count"

        # Info about the best count number prediction
        # Shape: (batch_size,)
        best_count_number = torch.argmax(self.count_answer, -1)

        predicted_count = best_count_number.detach().cpu().numpy()
        answer_json["count"] = predicted_count
        return answer_json

    def _get_best_arithmetic_expression(self,
                                        original_numbers: List[int],
                                        offsets: List[Tuple[int, int]],
                                        number_indices: List[int]) -> JsonDict:
        answer_json = {}
        answer_json["answer_type"] = "arithmetic"
        sign_remap = {0: 0, 1: 1, 2: -1}
    
        number_indices_tensor = self.number_indices.squeeze(-1)
        number_mask = (number_indices_tensor != -1).long()

        # Shape: (batch_size, # of numbers in passage).
        best_signs_for_numbers = torch.argmax(self.arithmetic_answer, -1)
        # For padding numbers, the best sign masked as 0 (not included).
        best_signs_for_numbers = util.replace_masked_values(best_signs_for_numbers, number_mask, 0)

        predicted_signs = [sign_remap[it] for it in best_signs_for_numbers[0].detach().cpu().numpy()]
        result = sum([sign * number for sign, number in zip(predicted_signs, original_numbers)])
        predicted_answer = str(result)
        # offsets = metadata[i]['passage_token_offsets']
        # number_indices = metadata[i]['number_indices']
        number_positions = [offsets[index] for index in number_indices]
        answer_json['numbers'] = []
        for offset, value, sign in zip(number_positions, original_numbers, predicted_signs):
            answer_json['numbers'].append({'span': offset, 'value': value, 'sign': sign})
        if number_indices[-1] == -1:
            # There is a dummy 0 number at position -1 added in some cases; we are
            # removing that here.
            answer_json["numbers"].pop()
        answer_json["value"] = result
        return answer_json


class DropNaqanetLanguage(DomainLanguage):
    """
    A proof-of-concept implementation of the "language" that NAQANet uses for DROP.  That
    "language" has four top-level functions: (1) predict an answer from the passage, (2) predict an
    answer from the question, (3) predict a count, and (4) predict an arithmetic expression.  The
    goal here is to show that we can get the same performance when implementing this as a
    "language" in a semantic parser, so that we can more easily make the language more complex in
    the future (including by adding nesting / recursion).
    """
    def __init__(self,
                 encoded_question: torch.Tensor,
                 question_mask: torch.LongTensor,
                 passage_vector: torch.Tensor,
                 passage_mask: torch.LongTensor,
                 modeled_passage_list: List[torch.Tensor],
                 number_indices: torch.LongTensor,
                 parameters: NaqanetParameters) -> None:
        super().__init__(start_types={Answer})
        self.encoded_question = encoded_question
        self.question_mask = question_mask
        self.passage_vector = passage_vector
        self.passage_mask = passage_mask
        self.modeled_passage_list = modeled_passage_list
        self.number_indices = number_indices
        self.params = parameters

    @predicate
    def passage_span(self) -> Answer:
        # Shape: (passage_length, modeling_dim * 2))
        passage_for_span_start = torch.cat([self.modeled_passage_list[0],
                                            self.modeled_passage_list[1]],
                                           dim=-1)

        # Shape: (passage_length)
        passage_span_start_logits = self.params.passage_span_start_predictor(passage_for_span_start).squeeze(-1)
        # Shape: (passage_length, modeling_dim * 2)
        passage_for_span_end = torch.cat([self.modeled_passage_list[0],
                                          self.modeled_passage_list[2]],
                                         dim=-1)
        # Shape: (passage_length)
        passage_span_end_logits = self.params.passage_span_end_predictor(passage_for_span_end).squeeze(-1)
        # Shape: (passage_length)
        passage_span_start_log_probs = util.masked_log_softmax(passage_span_start_logits, self.passage_mask)
        passage_span_end_log_probs = util.masked_log_softmax(passage_span_end_logits, self.passage_mask)
        
        # Shape: (passage_length) 
        passage_span_start_log_probs= util.replace_masked_values(passage_span_start_log_probs, self.passage_mask, -1e7)
        passage_span_end_log_probs = util.replace_masked_values(passage_span_end_log_probs, self.passage_mask, -1e7) 

        return Answer(passage_span=(passage_span_start_log_probs, passage_span_end_log_probs), number_indices=self.number_indices)

    @predicate
    def question_span(self) -> Answer:
        # Shape: (batch_size, question_length)

        # Shape (question_length, modeling_dim)
        encoded_question_for_span_prediction = torch.cat(
                [self.encoded_question,
                 self.passage_vector.repeat(self.encoded_question.size(0), 1)],
                -1)

        # Shape (question_length)        
        question_span_start_logits = \
            self.params.question_span_start_predictor(encoded_question_for_span_prediction).squeeze(-1)

        # Shape: (question_length)
        question_span_end_logits = \
            self.params.question_span_end_predictor(encoded_question_for_span_prediction).squeeze(-1)

        question_span_start_log_probs = util.masked_log_softmax(question_span_start_logits, self.question_mask)
        question_span_end_log_probs = util.masked_log_softmax(question_span_end_logits, self.question_mask)

        # Shape: (question_length) 
        question_span_start_log_probs= util.replace_masked_values(question_span_start_log_probs, self.question_mask, -1e7)
        question_span_end_log_probs = util.replace_masked_values(question_span_end_log_probs, self.question_mask, -1e7) 

        return Answer(question_span=(question_span_start_log_probs, question_span_end_log_probs), number_indices=self.number_indices)

    @predicate
    def count(self) -> Answer:
        # Shape: (batch_size, 10)
        count_number_logits = self.params.count_number_predictor(self.passage_vector)
        count_number_log_probs = torch.nn.functional.log_softmax(count_number_logits, -1)
        return Answer(count_answer=count_number_log_probs, number_indices=self.number_indices)

    @predicate
    def arithmetic_expression(self) -> Answer:
        number_indices = self.number_indices.squeeze(-1)
        number_mask = (number_indices != -1).long()
        clamped_number_indices = util.replace_masked_values(number_indices, number_mask, 0)
        encoded_passage_for_numbers = torch.cat([self.modeled_passage_list[0],
                                                 self.modeled_passage_list[3]],
                                                dim=-1)


        # Shape: (# of numbers in the passage, 2 * encoding_dim)
        encoded_numbers = torch.gather(
                encoded_passage_for_numbers,
                0,
                clamped_number_indices.unsqueeze(-1).expand(-1, encoded_passage_for_numbers.size(-1)))

        # Shape: (# of numbers in the passage, 2 * encoding_dim)
        encoded_numbers = torch.cat(
                [encoded_numbers, 
                 self.passage_vector.repeat(encoded_numbers.size(0), 1)], -1)

        # Shape: (# of numbers in the passage, 3)
        number_sign_logits = self.params.number_sign_predictor(encoded_numbers)
        number_sign_log_probs = torch.nn.functional.log_softmax(number_sign_logits, -1)
        return Answer(arithmetic_answer=number_sign_log_probs, number_indices=self.number_indices)
