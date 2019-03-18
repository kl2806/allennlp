import torch
from allennlp.common.testing import AllenNlpTestCase

from allennlp.semparse.domain_languages import DropNaqanetLanguage, NaqanetParameters
from allennlp.tests.semparse.domain_languages.domain_language_test import check_productions_match


class SemparseNaqanetLanguageTest(AllenNlpTestCase):
    def setUp(self):
        super().setUp()

        self.modeling_dim = 12
        self.naqanet_parameters = NaqanetParameters(self.modeling_dim) 
        
        self.batch_size = 16
        self.question_length = 20
        self.passage_length = 100

        self.encoded_question = torch.rand(self.batch_size,
                                           self.question_length,
                                           self.modeling_dim)
        
        self.question_mask = torch.ones((self.batch_size,
                                        self.question_length), dtype=torch.long)

        self.encoded_passage = torch.rand(self.batch_size,
                                           self.passage_length,
                                           self.modeling_dim)
        self.passage_vector = torch.rand(self.batch_size,
                                         self.modeling_dim)
        self.passage_mask = torch.ones((self.batch_size,
                                       self.passage_length), dtype=torch.long)
 
        self.modeled_passage_list = [torch.rand(self.batch_size, self.passage_length, self.modeling_dim) for _ in range(4)] 
        self.number_indices = torch.ones((self.batch_size, 10, 1), dtype=torch.long)

        self.language = DropNaqanetLanguage(encoded_question=self.encoded_question[0].unsqueeze(0), 
                                            question_mask=self.question_mask[0].unsqueeze(0),
                                            passage_vector=self.passage_vector[0].unsqueeze(0),
                                            passage_mask=self.passage_mask[0].unsqueeze(0),
                                            modeled_passage_list=[passage[0].unsqueeze(0) for passage in self.modeled_passage_list],
                                            number_indices=self.number_indices[0].unsqueeze(0), 
                                            parameters=self.naqanet_parameters) 

        
    def test_semparse_naqanet_execute(self):
        answer = self.language.execute('count')
        assert answer.count_answer.squeeze().size() == torch.Size([10])
        
        answer = self.language.execute('question_span')
        assert answer.question_span[0].squeeze().size() == torch.Size([self.question_length])

        answer = self.language.execute('passage_span')
        assert answer.passage_span[0].squeeze().size() == torch.Size([self.passage_length])

        answer = self.language.execute('arithmetic_expression')
        assert answer.arithmetic_answer.squeeze().size() == torch.Size([10, 3])

        

    def test_semparse_naqanet_log_probs(self):
        answer = self.language.execute('count')
        assert answer.get_answer_log_prob(answer_as_passage_span=None,
                                          answer_as_question_span=None,
                                          answer_as_count=torch.ones((1, 10), dtype=torch.long),
                                          answer_as_arithmetic_expression=None,
                                          number_indices = self.number_indices).size() == torch.Size([1])

        answer = self.language.execute('question_span')
        assert answer.get_answer_log_prob(answer_as_passage_span=None,
                                          answer_as_question_span=torch.ones((1, 2, self.question_length), dtype=torch.long),
                                          answer_as_count=None,
                                          answer_as_arithmetic_expression=None,
                                          number_indices = self.number_indices).size() == torch.Size([1])
        
        answer = self.language.execute('passage_span')
        assert answer.get_answer_log_prob(answer_as_passage_span=torch.ones((1, 2, self.passage_length), dtype=torch.long),
                                          answer_as_question_span=None,
                                          answer_as_count=None,
                                          answer_as_arithmetic_expression=None,
                                          number_indices = self.number_indices).size() == torch.Size([1])

        answer = self.language.execute('arithmetic_expression')
        assert answer.get_answer_log_prob(answer_as_passage_span=None,
                                          answer_as_question_span=None,
                                          answer_as_count=None,
                                          answer_as_arithmetic_expression=torch.ones((1, 3, 10), dtype=torch.long),
                                          number_indices = self.number_indices[0].unsqueeze(0)).size() == torch.Size([1])



