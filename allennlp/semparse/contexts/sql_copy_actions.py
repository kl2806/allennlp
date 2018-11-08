from typing import List
from allennlp.semparse.contexts.sql_context_utils import action_sequence_to_sql, format_action
from allennlp.semparse.worlds import AtisWorld
from allennlp.models.semantic_parsing.atis.atis_semantic_parser import AtisSemanticParser
from allennlp.state_machines.states import GrammarStatelet
from pprint import pprint
from nltk import ngrams
import more_itertools

def get_action_sequence_copy_candidates(world: AtisWorld, action_sequence: List[str],
                                        candidate_node_types: List[str]) -> List[str]:
    """
    Given a sequence of actions previously generated in the interaction, we want to
    extract the subsequences that represent subtrees that we potentially want to copy.
    """
    action_subsequence_candidates: List[List[str]] = []

    for candidate_node_type in candidate_node_types:
        for index, action in enumerate(action_sequence):
            if action.split(' -> ')[0] == candidate_node_type:
                grammar_state = GrammarStatelet([candidate_node_type],
                                                world.valid_actions,
                                                AtisSemanticParser.is_nonterminal)
                action_subsequence_candidate = []
                for action in action_sequence[index:]:
                    grammar_state = grammar_state.take_action(action)
                    action_subsequence_candidate.append(action)
                    if grammar_state._nonterminal_stack == []:
                        break
                action_subsequence_candidates.append(action_subsequence_candidate)
    return action_subsequence_candidates

def add_copy_actions_to_target_sequence(action_subsequence_candidates: List[List[str]],
                                        target_sequence: List[str]):
    """
    We replace subsequences in the target sequence with copy actions. We replace subsequences
    in the target sequence greedily by replacing the sequences longest to shortest.
    """
    action_subsequence_candidates = sorted(action_subsequence_candidates, key=len, reverse=True)
    sql_subtrees = [action_sequence_to_sql(action_subsequence_candidate, root_nonterminal='condition')
                    for action_subsequence_candidate in action_subsequence_candidates]

    for action_subsequence_candidate in action_subsequence_candidates:
        matches_action_subsequence = lambda *subsequence: subsequence == tuple(action_subsequence_candidate)
        sql = action_sequence_to_sql(action_subsequence_candidate, root_nonterminal='condition')
        copy_action = format_action('condition', right_hand_side=sql,
                is_number=True)
        target_sequence = list(more_itertools.replace(iterable=target_sequence,
                                                      pred=matches_action_subsequence,
                                                      substitutes=[copy_action],
                                                      window_size=len(action_subsequence_candidate)))
    pprint(action_sequence_to_sql(target_sequence))
    return target_sequence



def add_copy_actions(action_subsequence_candidates: List[List[str]]):
    """
    We update the valid actions, the linking scores, and the entities here. 
    """
    pass

    


