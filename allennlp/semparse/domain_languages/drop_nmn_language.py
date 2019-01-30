from typing import Dict, List, Tuple

import torch
from torch import Tensor
from torch.nn import Parameter, LSTM

from allennlp.semparse.domain_languages.domain_language import (DomainLanguage, predicate,
                                                                predicate_with_side_args)

# An Attention is a single tensor; we're giving this a type so that we can use it for constructing
# predicates.
class AttentionTensor(Tensor):
    pass


# A ``SpanAnswer`` is a distribution over spans in the passage. It has a distribution over the start
# span and the end span.
class SpanAnswer:
    def __init__(self,
                 start_index: Tensor,
                 end_index: Tensor) -> None:
        self.start_index = start_index
        self.end_index = end_index

# A ``NumberAnswer`` is a distribution over the possible answers from addition and subtraction.
class NumberAnswer(Tensor):
    pass

# A ``CountAnswer`` is a distribution over the possible counts in the passage.
class CountAnswer(Tensor):
    pass

class DropNmnParameters(torch.nn.Module):
    """
    Stores all of the parameters necessary for the various learned functions in the
    ``DropNmnLanguage``.  This needs to be constructed outside the ``Language`` object,
    because we create one of those objects per ``Instance``, while the parameters are shared across
    all ``Instances``.  This also needs to be a ``torch.nn.Module``, so that your ``Model`` can
    save it as a member variable and have its weights get updated.
    """
    def __init__(self,
                 passage_length: int,
                 question_encoding_dim: int,
                 passage_encoding_dim: int,
                 hidden_dim: int,
                 num_number_answers: int,
                 num_count_answers: int,
                 num_span_answers: int) -> None:
        super().__init__()
        from allennlp.modules.attention.dot_product_attention import DotProductAttention
        self.find_attention = DotProductAttention()

        self.relocate_conv1 = torch.nn.Linear(passage_encoding_dim, hidden_dim)
        self.relocate_conv2 = torch.nn.Linear(hidden_dim, 1)
        self.relocate_linear1 = torch.nn.Linear(passage_encoding_dim, 1)
        self.relocate_linear2 = torch.nn.Linear(question_encoding_dim, hidden_dim)

        self.count_lstm = LSTM(1,2)
        self.count_linear = torch.nn.Linear(2, num_count_answers)

        self.more_linear1 = torch.nn.Linear(passage_length, num_span_answers)
        self.more_linear2 = torch.nn.Linear(passage_length, num_span_answers)

        self.less_linear1 = torch.nn.Linear(passage_length, num_span_answers)
        self.less_linear2 = torch.nn.Linear(passage_length, num_span_answers)

        self.compare_linear1 = torch.nn.Linear(hidden_dim, num_span_answers)
        self.compare_linear2 = torch.nn.Linear(passage_encoding_dim, hidden_dim)
        self.compare_linear3 = torch.nn.Linear(passage_encoding_dim, hidden_dim)
        self.compare_linear4 = torch.nn.Linear(question_encoding_dim, hidden_dim)

        self.num_number_answers = num_number_answers


class DropNmnLanguage(DomainLanguage):
    """
    DomainLanguage for the DROP dataset based on neural module networks. This language has a `learned execution model`,
    meaning that the predicates in this language have learned parameters.

    Parameters
    ----------
    parameters : ``DropNmnParameters``
        The learnable parameters that we should use when executing functions in this language.
    """

    def __init__(self, encoded_passage: Tensor, parameters: DropNmnParameters) -> None:
        super().__init__(start_types={CountAnswer, SpanAnswer, NumberAnswer})
        self.encoded_passage = encoded_passage
        self.parameters = parameters

    @predicate_with_side_args(['attended_question'])
    def find(self, attended_question: Tensor) -> AttentionTensor:
        find = self.parameters.find_attention
        return find(attended_question.unsqueeze(0), self.encoded_passage.unsqueeze(0))

    @predicate_with_side_args(['attended_question'])
    def relocate(self, attention: AttentionTensor, attended_question: Tensor) -> AttentionTensor:
        conv1 = self.parameters.relocate_conv1
        conv2 = self.parameters.relocate_conv2
        linear1 = self.parameters.relocate_linear1
        linear2 = self.parameters.relocate_linear2
        attended_passage = (attention.unsqueeze(-1)* self.encoded_passage).sum(dim=[0])
        return conv2(conv1(self.encoded_passage) * linear1(attended_passage) * linear2(attended_question)).squeeze()

    @predicate
    def and_(self, attention1: AttentionTensor, attention2: AttentionTensor) -> AttentionTensor:
        return torch.max(torch.stack([attention1, attention2], dim=0), dim=0)[0]

    @predicate
    def or_(self, attention1: AttentionTensor, attention2: AttentionTensor) -> AttentionTensor:
        return torch.min(torch.stack([attention1, attention2], dim=0), dim=0)[0]

    @predicate
    def count(self, attention: AttentionTensor) -> CountAnswer:
        lstm = self.parameters.count_lstm
        linear = self.parameters.count_linear

        # (1, passage_length, 2)
        hidden_states  = lstm(attention.unsqueeze(-1))[0]
        return linear(hidden_states.squeeze()[-1])

    @predicate_with_side_args(['attended_question'])
    def compare(self, attention1: AttentionTensor, attention2: AttentionTensor, attended_question: Tensor) -> SpanAnswer:
        linear1 = self.parameters.compare_linear1
        linear2 = self.parameters.compare_linear2
        linear3 = self.parameters.compare_linear3
        linear4 = self.parameters.compare_linear4
        attended_passage1 = (attention1.unsqueeze(-1) * self.encoded_passage).sum(dim=[0])
        attended_passage2 = (attention2.unsqueeze(-1) * self.encoded_passage).sum(dim=[0])
        return linear1(linear2(attended_passage1) * linear3(attended_passage2) * linear4(attended_question))

    
    @predicate_with_side_args(['attention_map'])
    def subtract_(self,
                  attention1: AttentionTensor,
                  attention2: AttentionTensor,
                  attention_map: Dict[int, List[Tuple[int, int]]]) -> NumberAnswer:
        attention_product = torch.matmul(attention1.unsqueeze(-1), torch.t(attention2.unsqueeze(-1)))
        answers = torch.zeros(len(attention_map),)
        for candidate_index, (candidate_subtraction, indices) in enumerate(attention_map.items()):
            attention_sum = 0
            for index1, index2 in indices:
                attention_sum += attention_product[index1, index2]
            answers[candidate_index] = attention_sum
        return NumberAnswer(answers)

    @predicate_with_side_args(['attention_map'])
    def add_(self,
             attention1: AttentionTensor,
             attention2: AttentionTensor,
             attention_map: Dict[int, List[Tuple[int, int]]]) -> NumberAnswer:
        attention_product = torch.matmul(attention1.unsqueeze(-1), torch.t(attention2.unsqueeze(-1)))
        answers = torch.zeros(len(attention_map),)
        for candidate_index, (candidate_addition, indices) in enumerate(attention_map.items()):
            attention_sum = 0
            for index1, index2 in indices:
                attention_sum += attention_product[index1, index2]
            answers[candidate_index] = attention_sum
        return NumberAnswer(answers)

