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

class DateDistribution:
    def __init__(self,
                 year_distribution: Tensor,
                 month_distribution: Tensor,
                 day_distribution: Tensor) -> None:
        self.year_distribution = year_distribution
        self.month_distribution = month_distribution 
        self.day_distribution = day_distribution

class DateDelta:
    def __init__(self,
                 year_delta: Tensor,
                 month_delta: Tensor,
                 day_delta: Tensor) -> None:
        self.year_delta = year_delta
        self.month_delta = month_delta
        self.day_delta = day_delta


# Various answer types
# A ``PassageSpanAnswer`` is a distribution over spans in the passage. It has a distribution over the start
# span and the end span.
class PassageSpanAnswer:
    def __init__(self,
                 start_index: Tensor,
                 end_index: Tensor) -> None:
        self.start_index = start_index
        self.end_index = end_index

# A ``QuestionSpanAnswer`` is a distribution over spans in the question. It has a distribution over the start
# span and the end span.
class QuestionSpanAnswer:
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
                 count_hidden_dim: int = 2,
                 maximum_count: int = 10) -> None:
        super().__init__()
        from allennlp.modules.attention.dot_product_attention import DotProductAttention
        self.find_attention = DotProductAttention()

        self.relocate_linear1 = torch.nn.Linear(passage_encoding_dim, hidden_dim)
        self.relocate_linear2 = torch.nn.Linear(hidden_dim, 1)
        self.relocate_linear3 = torch.nn.Linear(passage_encoding_dim, 1)
        self.relocate_linear4 = torch.nn.Linear(question_encoding_dim, hidden_dim)

        self.count_lstm = LSTM(1, count_hidden_dim)
        self.count_linear = torch.nn.Linear(count_hidden_dim, maximum_count)
        
class DropNmnLanguage(DomainLanguage):
    """
    DomainLanguage for the DROP dataset based on neural module networks. This language has a `learned execution model`,
    meaning that the predicates in this language have learned parameters.

    Parameters
    ----------
    parameters : ``DropNmnParameters``
        The learnable parameters that we should use when executing functions in this language.
    """

    def __init__(self, encoded_passage: Tensor, parameters: DropNmnParameters, max_samples=10) -> None:
        super().__init__(start_types={CountAnswer, PassageSpanAnswer, QuestionSpanAnswer, NumberAnswer})
        self.encoded_passage = encoded_passage
        self.parameters = parameters
        self.max_samples = max_samples

    @predicate_with_side_args(['attended_question'])
    def find(self, attended_question: Tensor) -> AttentionTensor:
        find = self.parameters.find_attention
        return find(attended_question.unsqueeze(0), self.encoded_passage.unsqueeze(0)).squeeze()
    
    @predicate_with_side_args(['attended_question'])
    def select(self, attention: AttentionTensor, attended_question: Tensor) -> AttentionTensor:
        raise NotImplementedError 

    @predicate_with_side_args(['attended_question'])
    def argmax(self, attention: AttentionTensor, attended_question: Tensor) -> AttentionTensor:
        raise NotImplementedError 

    @predicate_with_side_args(['attended_question'])
    def argmax_k(self, attention: AttentionTensor, attended_question: Tensor, k: int) -> AttentionTensor:
        raise NotImplementedError

    @predicate_with_side_args(['attended_question'])
    def relocate(self, attention: AttentionTensor, attended_question: Tensor) -> AttentionTensor:
        linear1 = self.parameters.relocate_linear1
        linear2 = self.parameters.relocate_linear2
        linear3 = self.parameters.relocate_linear3
        linear4 = self.parameters.relocate_linear4
        attended_passage = (attention.unsqueeze(-1) * self.encoded_passage).sum(dim=[0])
        return linear2(linear1(self.encoded_passage) * linear3(attended_passage) * linear4(attended_question)).squeeze()

    @predicate_with_side_args(['attended_question'])
    def compare(self, attention1: AttentionTensor, attention2: AttentionTensor, attended_question: Tensor) -> AttentionTensor:
        raise NotImplementedError

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

    @predicate
    def maximum_number(self, numbers: NumberAnswer) -> NumberAnswer:
        cumulative_distribution_function = numbers.cumsum(0) 
        cumulative_distribution_function_n = cumulative_distribution_function**self.max_samples
        maximum_distribution = cumulative_distribution_function_n - torch.cat((torch.zeros(1), cumulative_distribution_function_n[:-1]))
        return maximum_distribution

    @predicate 
    def subtract(self,
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
    
    @predicate
    def add(self,
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
    
    
    @predicate
    def extract_question_span(self,
                              attention: AttentionTensor) -> QuestionSpanAnswer:
        raise NotImplementedError
    
    @predicate
    def extract_passage_span(self,
                              attention: AttentionTensor) -> PassageSpanAnswer:
        raise NotImplementedError

    @predicate
    def passage_to_number_distribution(self,
                                       passage: AttentionTensor) -> NumberAnswer:
        raise NotImplementedError

    @predicate
    def passage_to_question(self,
                            attention: AttentionTensor) -> AttentionTensor:
        raise NotImplementedError


    @predicate
    def subtract_date(self,
                      date1: DateDistribution,
                      date2: DateDistribution) -> DateDelta:
        raise NotImplementedError
    
    @predicate
    def add_date_delta(self
                       date: DateDistribution,
                       date_delta: DateDelta) -> DateDelta:
        raise NotImplementedError
    
    @predicate
    def subtract_numbers(self,
                         numbers: Tuple[NumberAnswer, NumberAnswer]) -> NumberAnswer:
        raise NotImplementedError

    @predicate
    def add_numbers(self,
                    numbers: Tuple[NumberAnswer, NumberAnswer]) -> NumberAnswer:
        raise NotImplementedError
    
    # Extractors without parameters
    @predicate
    def extract_year(self,
                     date: DateDistribution) -> NumberAnswer:
        raise NotImplementedError

    @predicate
    def extract_month(self,
                      date: DateDistribution) -> NumberAnswer:
        raise NotImplementedError

    @predicate
    def extract_day(self,
                    date: DateDistribution) -> NumberAnswer:
        raise NotImplementedError

    @predicate 
    def extract_score(self,
                      passage: AttentionTensor) -> Tuple[NumberAnswer, NumberAnswer]: 
        raise NotImplementedError



