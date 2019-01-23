from typing import Dict

import torch
from torch import Tensor
from torch.nn import Parameter

from allennlp.semparse.domain_languages.domain_language import (DomainLanguage, predicate,
                                                                predicate_with_side_args)


# An Attention is a single tensor; we're giving this a type so that we can use it for constructing
# predicates.
class Attention(Tensor):
    pass


# An Answer is a distribution over answer options, which is a tensor.  Again, we're subclassing
# Tensor here to give a concrete type for our predicates to use.
class Answer(Tensor):
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
                 question_hidden_dim: int,
                 passage_encoding_dim: int,
                 hidden_dim: int,
                 num_answers: int) -> None:
        super().__init__()
        # The paper calls these "convolutions", but if you look at the code, it just does a 1x1
        # convolution, which is a linear transformation on the last dimension.  I'm still calling
        # them "convs" here, to match the original paper, and because they are operating on
        # image-feature-shaped things, and we might change our mind later about how to parameterize
        # them.  All of these parameters try to match what you see in table 1 of the paper, where
        # we write "W" as "linear".
        self.find_conv1 = torch.nn.Linear(passage_encoding_dim, question_hidden_dim)
        self.find_conv2 = torch.nn.Linear(question_hidden_dim, 1)
        self.find_linear = torch.nn.Linear(question_encoding_dim, question_hidden_dim)
        text_encoding_dim = question_encoding_dim  

        self.relocate_conv1 = torch.nn.Linear(passage_encoding_dim, hidden_dim)
        self.relocate_conv2 = torch.nn.Linear(hidden_dim, 1)
        self.relocate_linear1 = torch.nn.Linear(passage_encoding_dim, 1)
        self.relocate_linear2 = torch.nn.Linear(question_encoding_dim, hidden_dim)

        self.exist_linear = torch.nn.Linear(passage_length, num_answers)
        self.count_linear = torch.nn.Linear(passage_length, num_answers)

        self.describe_linear1 = torch.nn.Linear(hidden_dim, num_answers)
        self.describe_linear2 = torch.nn.Linear(passage_encoding_dim, hidden_dim)
        self.describe_linear3 = torch.nn.Linear(question_encoding_dim, hidden_dim)

        self.count_equals_linear1 = torch.nn.Linear(passage_length, num_answers)
        self.count_equals_linear2 = torch.nn.Linear(passage_length, num_answers)

        self.more_linear1 = torch.nn.Linear(passage_length, num_answers)
        self.more_linear2 = torch.nn.Linear(passage_length, num_answers)

        self.less_linear1 = torch.nn.Linear(passage_length, num_answers)
        self.less_linear2 = torch.nn.Linear(passage_length, num_answers)

        self.compare_linear1 = torch.nn.Linear(hidden_dim, num_answers)
        self.compare_linear2 = torch.nn.Linear(passage_encoding_dim, hidden_dim)
        self.compare_linear3 = torch.nn.Linear(passage_encoding_dim, hidden_dim)
        self.compare_linear4 = torch.nn.Linear(question_encoding_dim, hidden_dim)


class DropNmnLanguage(DomainLanguage):
    """
    DomainLanguage for the DROP dataset based on neural module networks This language has a `learned execution model`,
    meaning that the predicates in this language have learned parameters.

    Parameters
    ----------
    parameters : ``DropNmnParameters``
        The learnable parameters that we should use when executing functions in this language.
    """

    def __init__(self, encoded_passage: Tensor, parameters: DropNmnParameters) -> None:
        super().__init__(start_types={Answer})
        self.encoded_passage = encoded_passage 
        self.parameters = parameters

    @predicate_with_side_args(['attended_question'])
    def find(self, attended_question: Tensor) -> Attention:
        conv1 = self.parameters.find_conv1
        conv2 = self.parameters.find_conv2
        return conv2(conv1(self.encoded_passage) * attended_question).squeeze()

    @predicate_with_side_args(['attended_question'])
    def relocate(self, attention: Attention, attended_question: Tensor) -> Attention:
        conv1 = self.parameters.relocate_conv1
        conv2 = self.parameters.relocate_conv2
        linear1 = self.parameters.relocate_linear1
        linear2 = self.parameters.relocate_linear2
        attended_passage = (attention.unsqueeze(-1)* self.encoded_passage).sum(dim=[0])
        return conv2(conv1(self.encoded_passage) * linear1(attended_passage) * linear2(attended_question)).squeeze()

    @predicate_with_side_args(['attended_question'])
    def filter(self, attention: Attention, attended_question: Tensor) -> Attention:
        return self.and_(attention, self.find(attended_question))

    @predicate
    def and_(self, attention1: Attention, attention2: Attention) -> Attention:
        return torch.max(torch.stack([attention1, attention2], dim=0), dim=0)[0]

    @predicate
    def or_(self, attention1: Attention, attention2: Attention) -> Attention:
        return torch.min(torch.stack([attention1, attention2], dim=0), dim=0)[0]

    @predicate
    def exist(self, attention: Attention) -> Answer:
        linear = self.parameters.exist_linear
        return linear(attention.view(-1))

    @predicate
    def count(self, attention: Attention) -> Answer:
        linear = self.parameters.count_linear
        return linear(attention.view(-1))

    @predicate_with_side_args(['attended_question'])
    def describe(self, attention: Attention, attended_question: Tensor) -> Answer:
        linear1 = self.parameters.describe_linear1
        linear2 = self.parameters.describe_linear2
        linear3 = self.parameters.describe_linear3
        attended_passage = (attention.unsqueeze(-1) * self.encoded_passage).sum(dim=[0])
        return linear1(linear2(attended_passage) * linear3(attended_question))

    @predicate_with_side_args(['attended_question'])
    def compare(self, attention1: Attention, attention2: Attention, attended_question: Tensor) -> Answer:
        linear1 = self.parameters.compare_linear1
        linear2 = self.parameters.compare_linear2
        linear3 = self.parameters.compare_linear3
        linear4 = self.parameters.compare_linear4
        attended_passage1 = (attention1.unsqueeze(-1) * self.encoded_passage).sum(dim=[0])
        attended_passage2 = (attention2.unsqueeze(-1) * self.encoded_passage).sum(dim=[0])
        return linear1(linear2(attended_passage1) * linear3(attended_passage2) * linear4(attended_question))

    @predicate
    def count_equals(self, attention1: Attention, attention2: Attention) -> Answer:
        linear1 = self.parameters.count_equals_linear1
        linear2 = self.parameters.count_equals_linear2
        return linear1(attention1.view(-1)) + linear2(attention2.view(-1))

    @predicate
    def more(self, attention1: Attention, attention2: Attention) -> Answer:
        linear1 = self.parameters.more_linear1
        linear2 = self.parameters.more_linear2
        return linear1(attention1.view(-1)) + linear2(attention2.view(-1))

    @predicate
    def less(self, attention1: Attention, attention2: Attention) -> Answer:
        linear1 = self.parameters.less_linear1
        linear2 = self.parameters.less_linear2
        return linear1(attention1.view(-1)) + linear2(attention2.view(-1))