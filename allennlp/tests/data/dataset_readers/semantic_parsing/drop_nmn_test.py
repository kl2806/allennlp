# pylint: disable=no-self-use,invalid-name
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.dataset_readers import DROPReader
from allennlp.semparse.domain_languages import DropNmnLanguage 


class TestDropDatasetReader(AllenNlpTestCase):
    def test_drop_reader(self):
        test_file = str(self.FIXTURES_ROOT / "data" / "drop.json")
        dataset = DROPReader().read(test_file)
        instances = list(dataset)
        assert len(instances) == 18
        instance = instances[0]

        assert instance.fields.keys() == {'passage', 'question', 'number_indices', 'numbers_in_passage', 'answer_as_passage_spans', 'answer_as_question_spans', 'answer_as_plus_minus_combinations', 'answer_as_counts', 'metadata'}

        print(instance['answer_as_passage_spans'])
        print(instance['answer_as_question_spans'])
