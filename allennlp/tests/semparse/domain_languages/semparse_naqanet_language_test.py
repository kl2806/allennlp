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

        self.encoded_question = torch.ones(self.batch_size,
                                           self.question_length,
                                           self.modeling_dim)
        
        self.question_mask = torch.ones((self.batch_size,
                                        self.question_length), dtype=torch.long)

        self.encoded_passage = torch.rand(self.batch_size,
                                           self.passage_length,
                                           self.modeling_dim)
        self.passage_vector = torch.ones(self.batch_size,
                                         self.modeling_dim)
        self.passage_mask = torch.ones((self.batch_size,
                                       self.passage_length), dtype=torch.long)
 
        self.modeled_passage_list = [torch.rand(self.batch_size, self.passage_length, self.modeling_dim) for _ in range(4)] 
        self.number_indices = torch.ones((self.batch_size, 10, 1), dtype=torch.long)
        self.batch_index = 0
        
        self.language = DropNaqanetLanguage(encoded_question=self.encoded_question,
                                            question_mask=self.question_mask,
                                            passage_vector=self.passage_vector,
                                            passage_mask=self.passage_mask,
                                            modeled_passage_list=[passage for passage in self.modeled_passage_list],
                                            number_indices=self.number_indices,
                                            parameters=self.naqanet_parameters,
                                            batch_index = self.batch_index) 

        
    def test_semparse_naqanet_execute(self):
        answer = self.language.execute('count')
        assert answer.count_answer.size() == torch.Size([10])
        
        answer = self.language.execute('question_span')
        assert answer.question_span[0].size() == torch.Size([self.question_length])

        answer = self.language.execute('passage_span')
        assert answer.passage_span[0].size() == torch.Size([self.passage_length])

        answer = self.language.execute('arithmetic_expression')
        assert answer.arithmetic_answer.size() == torch.Size([10, 3])

        answer = self.language.execute_action_sequence(['Answer -> passage_span'])
        assert answer.passage_span[0].size() == torch.Size([self.passage_length])


        

    def test_semparse_naqanet_log_probs(self):
        from allennlp.semparse.domain_languages.drop_naqanet_language import Answer
        answer = self.language.execute('count')
        answer.get_answer_log_prob(answer_as_passage_span=None,
                                   answer_as_question_span=None,
                                   answer_as_count=torch.ones(10, dtype=torch.long),
                                   answer_as_arithmetic_expression=None,
                                   number_indices = self.number_indices).size() == torch.Size([])

        answer = self.language.execute('question_span')
        assert answer.get_answer_log_prob(answer_as_passage_span=None,
                                          answer_as_question_span=torch.ones((2, self.question_length), dtype=torch.long),
                                          answer_as_count=None,
                                          answer_as_arithmetic_expression=None,
                                          number_indices = self.number_indices).size() == torch.Size([])
        
        answer = self.language.execute('passage_span')
        assert answer.get_answer_log_prob(answer_as_passage_span=torch.ones((2, self.passage_length), dtype=torch.long),
                                          answer_as_question_span=None,
                                          answer_as_count=None,
                                          answer_as_arithmetic_expression=None,
                                          number_indices = self.number_indices).size() == torch.Size([])

        answer = self.language.execute('arithmetic_expression')
        print(answer)
        answer = Answer(arithmetic_answer=torch.ones((10, 3), dtype=torch.float), number_indices=self.number_indices[0])
        print(answer)

        arithmetic_log_prob = answer.get_answer_log_prob(answer_as_passage_span=None,
                                                         answer_as_question_span=None,
                                                         answer_as_count=None,
                                                         answer_as_arithmetic_expression=torch.ones((3, 10), dtype=torch.long),
                                                         number_indices = self.number_indices[0])
        print(arithmetic_log_prob)
        


    def test_semparse_naqanet_best_answer(self):
        answer = self.language.execute('count')
        best_count = answer._get_best_count()
        
        span_begin_probs = torch.FloatTensor([0.1, 0.3, 0.05, 0.3, 0.25]).log()
        span_end_probs = torch.FloatTensor([0.65, 0.05, 0.2, 0.05, 0.05]).log()
        answer = self.language.execute('question_span')
        best_span = answer._get_best_span(span_begin_probs,
                                          span_end_probs,
                                          'this is the original text',
                                          [(0,4), (5,6)])

        answer = self.language.execute('arithmetic_expression')
        best_arithmetic = answer._get_best_arithmetic_expression(original_numbers=[1,2,3],
                                                                 offsets=[(0,3),(4,5),(6,7)],
                                                                 number_indices=[-1,-1,-1])

