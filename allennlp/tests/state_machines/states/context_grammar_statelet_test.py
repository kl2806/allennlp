# pylint: disable=no-self-use,invalid-name
import pytest
import torch
from numpy.testing import assert_almost_equal

from allennlp.common.testing import AllenNlpTestCase

from allennlp.state_machines.states import GrammarStatelet, ContextGrammarStatelet

def is_nonterminal(symbol: str) -> bool:
    if symbol == 'identity':
        return False
    if 'lambda ' in symbol:
        return False
    if symbol in {'x', 'y', 'z'}:
        return False
    return True

class TestContextGrammarStatelet(AllenNlpTestCase):
    def test_context_grammar_state_use_nonterminal(self):
        state = ContextGrammarStatelet(['s'], {}, {}, {}, is_nonterminal)
        assert not state.is_finished()
        state = ContextGrammarStatelet([], {}, {}, {}, is_nonterminal)
        assert state.is_finished()

    def test_get_valid_actions_uses_top_of_stack(self):
        s_actions = object()
        t_actions = object()
        e_actions = object()
        state = LambdaGrammarStatelet(['s'], {}, {'s': s_actions, 't': t_actions}, {}, is_nonterminal)
        assert state.get_valid_actions() == s_actions
        state = LambdaGrammarStatelet(['t'], {}, {'s': s_actions, 't': t_actions}, {}, is_nonterminal)
        assert state.get_valid_actions() == t_actions
        state = LambdaGrammarStatelet(['e'],
                                      {},
                                      {'s': s_actions, 't': t_actions, 'e': e_actions},
                                      {},
                                      is_nonterminal)
        assert state.get_valid_actions() == e_actions
