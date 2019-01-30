from numpy.testing import assert_almost_equal
import torch

from allennlp.common.testing import AllenNlpTestCase
from allennlp.semparse.domain_languages import DropNmnLanguage
from allennlp.semparse.domain_languages.drop_nmn_language import DropNmnParameters
from allennlp.tests.semparse.domain_languages.domain_language_test import check_productions_match


class DropNmnLanguageTest(AllenNlpTestCase):
    def setUp(self):
        super().setUp()
        self.text_encoding_dim = 6
        self.hidden_dim = 6
        self.num_answers = 7

        self.passage_length = 11
        self.passage_encoding_dim = 6
        
        self.num_count_answers= 10 
        self.num_number_answers= 11
        self.num_span_answers= 12

        self.encoded_passage = torch.rand(self.passage_length, self.passage_encoding_dim)
        self.parameters = DropNmnParameters(passage_length=self.passage_length,
                                            question_encoding_dim=self.text_encoding_dim,
                                            passage_encoding_dim=self.passage_encoding_dim,
                                            hidden_dim=self.hidden_dim,
                                            num_count_answers=self.num_count_answers,
                                            num_number_answers=self.num_number_answers,
                                            num_span_answers=self.num_span_answers)

        self.language = DropNmnLanguage(self.encoded_passage, self.parameters)

    def test_get_nonterminal_productions(self):
        productions = self.language.get_nonterminal_productions()
        assert set(productions.keys()) == {
                '@start@',
                'Attention',
                'Answer',
                '<Attention:Answer>',
                '<Attention:Attention>',
                '<Attention,Attention:Answer>',
                '<Attention,Attention:Attention>',
                }
        check_productions_match(productions['@start@'],
                                ['Answer'])
        check_productions_match(productions['Attention'],
                                ['find',
                                 '[<Attention:Attention>, Attention]',
                                 '[<Attention,Attention:Attention>, Attention, Attention]'])
        check_productions_match(productions['Answer'],
                                ['[<Attention:Answer>, Attention]',
                                 '[<Attention,Attention:Answer>, Attention, Attention]'])
        check_productions_match(productions['<Attention:Answer>'],
                                ['exist', 'count', 'describe'])
        check_productions_match(productions['<Attention:Attention>'],
                                ['relocate', 'filter'])
        check_productions_match(productions['<Attention,Attention:Answer>'],
                                ['compare', 'count_equals', 'more', 'less'])
        check_productions_match(productions['<Attention,Attention:Attention>'],
                                ['and_', 'or_'])

    def test_find_returns_correct_shape(self):
        attended_question = torch.rand(self.text_encoding_dim)
        attention = self.language.find(attended_question)
        attention.size() == self.text_encoding_dim

    def test_relocate_returns_correct_shape(self):
        attended_question = torch.rand(self.text_encoding_dim)
        passage_attention = torch.rand(1, self.passage_length)
        new_attention = self.language.relocate(passage_attention, attended_question)
        assert new_attention.size() == (self.passage_length,)
    
    def test_and_returns_correct_shape(self):
        attention1 = torch.rand(self.passage_length,)
        attention2 = torch.rand(self.passage_length) 
        new_attention = self.language.and_(attention1, attention2)
        assert new_attention.size() == (self.passage_length,)

    def test_or_returns_correct_shape(self):
        attention1 = torch.rand(self.passage_length,)
        attention2 = torch.rand(self.passage_length) 
        new_attention = self.language.or_(attention1, attention2)
        assert new_attention.size() == (self.passage_length,)


    def test_count_returns_correct_shape(self):
        attention = torch.rand(1, self.passage_length)
        answer = self.language.count(attention)
        assert answer.size() == (self.num_count_answers,)


    def test_addition_indices(self):
        attention1 = torch.tensor([0, 1, 1])
        attention2 = torch.tensor([0, 1, 1])
        attention_map = {2: [(0, 0)], 3: [(0, 1), (1, 0)],
                         4: [(0, 2), (1, 1), (2, 0)],
                         5: [(1, 2), (2, 1)], 6: [(2, 2)]}        
        answer = self.language.add_(attention1, attention2, attention_map)
        
        assert torch.allclose(answer, torch.tensor([0, 0, 1, 2, 1], dtype=torch.float))

    def test_subtraction_indices(self):
        attention1 = torch.tensor([0, 1, 1])
        attention2 = torch.tensor([0, 1, 1])
        attention_map = {0: [(0, 0), (1, 1), (2, 2)],
                        -1: [(0, 1), (1, 2)], -2: [(0, 2)],
                        1: [(1, 0), (2, 1)], 2: [(2, 0)]}
        
        answer = self.language.subtract_(attention1, attention2, attention_map)
        assert torch.allclose(answer, torch.tensor([2, 1, 0, 1, 0], dtype=torch.float))


    def test_execute_logical_forms(self):
        # This just tests that execution _succeeds_ - we're not going to bother checking the
        # computation performed by each function, because there are learned parameters in there.
        # We'll treat this as similar to a model test, just making sure the tensor operations work.
        attended_question = {'attended_question': torch.rand(self.text_encoding_dim)}
        # A simple one to start with: (count find)
        action_sequence = ['@start@ -> CountAnswer', 'CountAnswer -> [<AttentionTensor:CountAnswer>, AttentionTensor]',
                           '<AttentionTensor:CountAnswer> -> count', 'AttentionTensor -> find']

        assert self.language.execute_action_sequence(action_sequence, [attended_question] * len(action_sequence)).shape == (self.num_answers,)

    def test_nmn_action_sequence(self):
        # A simple one to start with: find two numbers and add them
        logical_form = "(add_ find find)"
        action_sequence = self.language.logical_form_to_action_sequence(logical_form)
        assert action_sequence == ['@start@ -> NumberAnswer', 'NumberAnswer -> [<AttentionTensor,AttentionTensor:NumberAnswer>, AttentionTensor, AttentionTensor]',
                                   '<AttentionTensor,AttentionTensor:NumberAnswer> -> add_', 'AttentionTensor -> find', 'AttentionTensor -> find']

        # ``How many field goals did Nick Folk attempt?``
        logical_form = "(count find)"
        action_sequence = self.language.logical_form_to_action_sequence(logical_form)
        assert action_sequence == ['@start@ -> CountAnswer', 'CountAnswer -> [<AttentionTensor:CountAnswer>, AttentionTensor]',
                                   '<AttentionTensor:CountAnswer> -> count', 'AttentionTensor -> find']



