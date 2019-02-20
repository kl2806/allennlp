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
        self.num_span_answers= 12

        self.encoded_passage = torch.rand(self.passage_length, self.passage_encoding_dim)
        self.parameters = DropNmnParameters(passage_length=self.passage_length,
                                            question_encoding_dim=self.text_encoding_dim,
                                            passage_encoding_dim=self.passage_encoding_dim,
                                            hidden_dim=self.hidden_dim)

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
        assert attention.size() == (self.passage_length,)

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

    def test_maximum_computes_correctly(self):
        numbers = torch.tensor([0.01, 0.69, 0.29, 0.01])
        maximum_distribution = self.language.maximum_number(numbers)
        assert_almost_equal(maximum_distribution.data.numpy(), [1.00e-20, 2.82e-02, 8.76e-01, 9.56e-02], 3)


    def test_addition_indices(self):
        attention1 = torch.tensor([0, 1, 1])
        attention2 = torch.tensor([0, 1, 1])
        attention_map = {2: [(0, 0)], 3: [(0, 1), (1, 0)],
                         4: [(0, 2), (1, 1), (2, 0)],
                         5: [(1, 2), (2, 1)], 6: [(2, 2)]}        
        answer = self.language.add(attention1, attention2, attention_map)
        
        assert torch.allclose(answer, torch.tensor([0, 0, 1, 2, 1], dtype=torch.float))

    def test_subtraction_indices(self):
        attention1 = torch.tensor([0, 1, 1])
        attention2 = torch.tensor([0, 1, 1])
        attention_map = {0: [(0, 0), (1, 1), (2, 2)],
                        -1: [(0, 1), (1, 2)], -2: [(0, 2)],
                        1: [(1, 0), (2, 1)], 2: [(2, 0)]}
        
        answer = self.language.subtract(attention1, attention2, attention_map)
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
        # ``How many field goals did Nick Folk attempt?``
        logical_form = "(count find)"
        action_sequence = self.language.logical_form_to_action_sequence(logical_form)
        assert action_sequence == ['@start@ -> CountAnswer', 'CountAnswer -> [<AttentionTensor:CountAnswer>, AttentionTensor]',
                                   '<AttentionTensor:CountAnswer> -> count', 'AttentionTensor -> find']

        # ``How many yards was the longest touchdown pass``
        logical_form = "(maximum_number (passage_to_number_distribution find))"
        action_sequence = self.language.logical_form_to_action_sequence(logical_form)

        # ``How much longer was the longest field goal than the shortest?``
        logical_form = "(subtract_numbers (maximum_number (passage_to_number find)) (minimum_number (passage_to_number find)))"
        
        # ``How many points did they win by?``
        logical_form = "(subtract_numbers (extract_score find))" 
        action_sequence = self.language.logical_form_to_action_sequence(logical_form)

        # ``Which kicker kicked the most field goals?``
        logical_form = "(extract_passage_span (argmax find))"
        action_sequence = self.language.logical_form_to_action_sequence(logical_form)

        # ``Which kicker kicked the second most field goals?``
        logical_form = "(extract_passage_span (argmax_k find 2))"
        action_sequence = self.language.logical_form_to_action_sequence(logical_form)


        # ``What group in percent is larger: male or female?``
        logical_form = "(extract_question_span (passage_to_question (compare find find)))"
        action_sequence = self.language.logical_form_to_action_sequence(logical_form)

        # ``Where did Charles travel to first, Castile or Barcelona?``
        logical_form = "(extract_question_span (passage_to_question (compare find find)))"
        action_sequence = self.language.logical_form_to_action_sequence(logical_form)
        
        
        # ``How many years after he married Elizabeth did James Douglas suceed to the title
        # and the estates of his father-in-law``
        logical_form = "(extract_question_span (passage_to_question (compare find find)))"
        action_sequence = self.language.logical_form_to_action_sequence(logical_form)





