# pylint: disable=no-self-use,invalid-name
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.dataset_readers import DROPReader
from allennlp.semparse.domain_languages import DropNmnLanguage 


class TestDropDatasetReader(AllenNlpTestCase):
    def test_addition_computes_correct_attention_map(self):
        numbers_in_passage = [1, 2, 3]
        number_indices = [0, 1, 2]
        candidate_additions = DROPReader.get_candidate_additions(numbers_in_passage, number_indices)
        assert candidate_additions == {2: [(0, 0)], 3: [(0, 1), (1, 0)],
                                       4: [(0, 2), (1, 1), (2, 0)],
                                       5: [(1, 2), (2, 1)], 6: [(2, 2)]}
    def test_drop_reader(self):
        test_file = str(self.FIXTURES_ROOT / "data" / "drop.json")
        dataset = DROPReader().read(test_file)
        instances = list(dataset)
        assert len(instances) == 18
        instance = instances[3]

        assert instance.fields.keys() == {'passage', 'question', 'number_indices',
                                          'numbers_in_passage', 'answer_as_passage_spans', 
                                          'answer_as_question_spans', 'answer_as_plus_minus_combinations',
                                          'answer_as_counts', 'metadata', 'actions'}
        print(instance['answer_as_passage_spans'])
        print(instance['answer_as_question_spans'])
        print('numbers in passage', instance['numbers_in_passage'])
        print('numbers in passage', type(instance['numbers_in_passage']))
        print('number indices', instance['number_indices'])
        print('plus minus combinations', instance['answer_as_plus_minus_combinations'])
        print('answers_as_counts', instance['answer_as_counts'])
        print(instance['passage'])
        print(instance['question'])
        print(instance['metadata'].metadata)
