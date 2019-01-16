# pylint: disable=no-self-use,invalid-name
from allennlp.common.testing import AllenNlpTestCase
from allennlp.data.dataset_readers import QuarelDatasetReader
from allennlp.semparse.worlds import QuarelWorld 


class TestQuaRelDatasetReader(AllenNlpTestCase):
    def test_zeroshot_quarel_reader(self):
        test_file = str(self.FIXTURES_ROOT / "data" / "quarel" /
                        "sample_v1.json")
        dataset = QuarelDatasetReader(lf_syntax="quarel_v1_attr_entities").read(test_file)
        instances = list(dataset)
        assert len(instances) == 3
        instance = instances[0]
        assert instance.fields.keys() == {'question', 'table', 'world', 'actions',
                                          'target_action_sequences', 'metadata'}
        sentence_tokens = instance.fields["question"].tokens
        expected_tokens = ['tommy', 'glided', 'across', 'the', 'marble', 'floor', 'with', 'ease', ',',
                           'but', 'slipped', 'and', 'fell', 'on', 'the', 'wet', 'floor', 'because', 'blankblank',
                           'has', 'more', 'resistance', '.', 'answeroptiona', 'marble', 'floor', 'answeroptionb',
                           'wet', 'floor']
        assert [t.text for t in sentence_tokens] == expected_tokens
        actions = [action.rule for action in instance.fields["actions"].field_list]
        
        print(actions)
        assert len(actions) == 32

        action_sequence = instance.fields["target_action_sequences"].field_list[0]
        action_indices = [l.sequence_index for l in action_sequence.field_list]
        action_strings = [actions[i] for i in action_indices]

        assert action_strings == \
               ['@start@ -> n',
                'n -> [<a,<a,<a,n>>>, a, a, a]',
                '<a,<a,<a,n>>> -> infer',
                'a -> [<a,<a,a>>, a, a]',
                '<a,<a,a>> -> and',
                'a -> [<r,<w,a>>, r, w]',
                '<r,<w,a>> -> a:friction',
                'r -> low',
                'w -> world1',
                'a -> [<r,<w,a>>, r, w]',
                '<r,<w,a>> -> a:friction',
                'r -> high',
                'w -> world2',
                'a -> [<r,<w,a>>, r, w]',
                '<r,<w,a>> -> a:friction',
                'r -> higher',
                'w -> world1',
                'a -> [<r,<w,a>>, r, w]',
                '<r,<w,a>> -> a:friction',
                'r -> higher',
                'w -> world2']

    def test_quarel_reader(self):
        test_file = str(self.FIXTURES_ROOT / "data" / "quarel" /
                        "sample_v1.json")
        dataset = QuarelDatasetReader(lf_syntax="quarel_v1").read(test_file)
        instances = list(dataset)
        assert len(instances) == 3
        instance = instances[0]
        assert instance.fields.keys() == {'question', 'table', 'world', 'actions',
                                          'target_action_sequences', 'metadata'}
        sentence_tokens = instance.fields["question"].tokens
        expected_tokens = ['tommy', 'glided', 'across', 'the', 'marble', 'floor', 'with', 'ease', ',',
                           'but', 'slipped', 'and', 'fell', 'on', 'the', 'wet', 'floor', 'because', 'blankblank',
                           'has', 'more', 'resistance', '.', 'answeroptiona', 'marble', 'floor', 'answeroptionb',
                           'wet', 'floor']
        assert [t.text for t in sentence_tokens] == expected_tokens
        actions = [action.rule for action in instance.fields["actions"].field_list]

        assert len(actions) == 31

        action_sequence = instance.fields["target_action_sequences"].field_list[0]
        action_indices = [l.sequence_index for l in action_sequence.field_list]
        action_strings = [actions[i] for i in action_indices]

        assert action_strings == \
                ['@start@ -> n',
                 'n -> [<a,<a,<a,n>>>, a, a, a]',
                 '<a,<a,<a,n>>> -> infer',
                 'a -> [<a,<a,a>>, a, a]',
                 '<a,<a,a>> -> and',
                 'a -> [<r,<w,a>>, r, w]',
                 '<r,<w,a>> -> friction',
                 'r -> low',
                 'w -> world1',
                 'a -> [<r,<w,a>>, r, w]',
                 '<r,<w,a>> -> friction',
                 'r -> high',
                 'w -> world2',
                 'a -> [<r,<w,a>>, r, w]',
                 '<r,<w,a>> -> friction',
                 'r -> higher',
                 'w -> world1',
                 'a -> [<r,<w,a>>, r, w]',
                 '<r,<w,a>> -> friction',
                 'r -> higher',
                 'w -> world2']

    def test_quarel_reader_domain_language(self):
        test_file = str(self.FIXTURES_ROOT / "data" / "quarel" /
                        "sample_v1.json")
        dataset = QuarelDatasetReader(lf_syntax="quarel_v1").read(test_file)
        instances = list(dataset)
        assert len(instances) == 3
        instance = instances[0]
        '''
        assert instance.fields.keys() == {'question', 'table', 'world', 'actions',
                                          'target_action_sequences', 'metadata'}
        sentence_tokens = instance.fields["question"].tokens
        expected_tokens = ['tommy', 'glided', 'across', 'the', 'marble', 'floor', 'with', 'ease', ',',
                           'but', 'slipped', 'and', 'fell', 'on', 'the', 'wet', 'floor', 'because', 'blankblank',
                           'has', 'more', 'resistance', '.', 'answeroptiona', 'marble', 'floor', 'answeroptionb',
                           'wet', 'floor']
        assert [t.text for t in sentence_tokens] == expected_tokens
        '''
        actions = [action.rule for action in instance.fields["actions"].field_list]

        # assert len(actions) == 31
        

        action_sequence = instance.fields["target_action_sequences"].field_list[0]
        action_indices = [l.sequence_index for l in action_sequence.field_list]
        action_strings = [actions[i] for i in action_indices]

        assert action_strings == \
                ['@start@ -> int',
                 'int -> [<QuaRelType,QuaRelType,QuaRelType:int>, QuaRelType, QuaRelType, '
                 'QuaRelType]',
                 '<QuaRelType,QuaRelType,QuaRelType:int> -> infer',
                 'QuaRelType -> [<QuaRelType,QuaRelType:QuaRelType>, QuaRelType, QuaRelType]',
                 '<QuaRelType,QuaRelType:QuaRelType> -> and',
                 'QuaRelType -> [<Direction,World:QuaRelType>, Direction, World]',
                 '<Direction,World:QuaRelType> -> friction',
                 'Direction -> low',
                 'World -> world1',
                 'QuaRelType -> [<Direction,World:QuaRelType>, Direction, World]',
                 '<Direction,World:QuaRelType> -> friction',
                 'Direction -> high',
                 'World -> world2',
                 'QuaRelType -> [<Direction,World:QuaRelType>, Direction, World]',
                 '<Direction,World:QuaRelType> -> friction',
                 'Direction -> higher',
                 'World -> world1',
                 'QuaRelType -> [<Direction,World:QuaRelType>, Direction, World]',
                 '<Direction,World:QuaRelType> -> friction',
                 'Direction -> higher',
                 'World -> world2']
