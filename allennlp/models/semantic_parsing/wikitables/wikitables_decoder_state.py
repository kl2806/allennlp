from typing import Dict, List, Tuple

import torch

from allennlp.semparse.worlds import WikiTablesWorld
from allennlp.data.fields.production_rule_field import ProductionRuleArray
from allennlp.nn.decoding import DecoderState, GrammarState, RnnState, ChecklistState


# This syntax is pretty weird and ugly, but it's necessary to make mypy happy with the API that
# we've defined.  We're using generics to make the type of `combine_states` come out right.  See
# the note in `nn.decoding.decoder_state.py` for a little more detail.
class WikiTablesDecoderState(DecoderState['WikiTablesDecoderState']):
    """
    Parameters
    ----------
    batch_indices : ``List[int]``
        Passed to super class; see docs there.
    action_history : ``List[List[int]]``
        Passed to super class; see docs there.
    score : ``List[torch.Tensor]``
        Passed to super class; see docs there.
    rnn_state : ``List[RnnState]``
        An ``RnnState`` for every group element.  This keeps track of the current decoder hidden
        state, the previous decoder output, the output from the encoder (for computing attentions),
        and other things that are typical seq2seq decoder state things.
    grammar_state : ``List[GrammarState]``
        This hold the current grammar state for each element of the group.  The ``GrammarState``
        keeps track of which actions are currently valid.
    possible_actions : ``List[List[ProductionRuleArray]]``
        The list of all possible actions that was passed to ``model.forward()``.  We need this so
        we can recover production strings, which we need to update grammar states.
    world : ``List[WikiTablesWorld]``, optional (default=None)
        The worlds corresponding to elements in the batch. We store them here because they're required
        for executing logical forms to determine costs while training, if we're learning to search.
        Otherwise, they're not required. Note that the worlds are batched, and they will be passed
        around unchanged during the decoding process.
    example_lisp_string : ``List[str]``, optional (default=None)
        The lisp strings that come from example files. They're also required for evaluating logical
        forms only if we're learning to search. These too are batched, and will be passed around
        unchanged.
    checklist_state : ``List[ChecklistState]``, optional (default=None)
        If you are using this state within a parser being trained for coverage, we need to store a
        ``ChecklistState`` which keeps track of the coverage information. Not needed if you are
        using a non-coverage based training algorithm.
    """
    def __init__(self,
                 batch_indices: List[int],
                 action_history: List[List[int]],
                 score: List[torch.Tensor],
                 rnn_state: List[RnnState],
                 grammar_state: List[GrammarState],
                 possible_actions: List[List[ProductionRuleArray]],
                 world: List[WikiTablesWorld] = None,
                 example_lisp_string: List[str] = None,
                 checklist_state: List[ChecklistState] = None,
                 debug_info: List = None) -> None:
        super(WikiTablesDecoderState, self).__init__(batch_indices, action_history, score)
        self.rnn_state = rnn_state
        self.grammar_state = grammar_state
        self.possible_actions = possible_actions
        self.world = world
        self.example_lisp_string = example_lisp_string
        # Converting None to a list of Nones of appropriate size to avoid checking for None in all
        # state operations.
        self.checklist_state = checklist_state or [None for _ in batch_indices]
        self.debug_info = debug_info

    def new_state_from_group_index(self,
                                   group_index: int,
                                   action: int,
                                   new_score: torch.Tensor,
                                   new_rnn_state: RnnState,
                                   decoder_debug_info: Dict[int, Tuple] = None,
                                   attention_weights: torch.Tensor = None) -> 'WikiTablesDecoderState':
        batch_index = self.batch_indices[group_index]
        new_action_history = self.action_history[group_index] + [action]
        production_rule = self.possible_actions[batch_index][action][0]
        new_grammar_state = self.grammar_state[group_index].take_action(production_rule)
        if self.checklist_state[0] is not None:
            new_checklist_state = [self.checklist_state[group_index].update(action)]
        else:
            new_checklist_state = None
        if self.debug_info is not None:
            considered_actions = []
            for i, log_probs, _, actions in decoder_debug_info[batch_index]:
                if i == group_index:
                    considered_actions = actions
                    probabilities = log_probs.exp().cpu()
            debug_info = {
                    'considered_actions': considered_actions,
                    'question_attention': attention_weights[group_index],
                    'probabilities': probabilities,
                    }
            new_debug_info = [self.debug_info[group_index] + [debug_info]]
        else:
            new_debug_info = None
        new_state = WikiTablesDecoderState(batch_indices=[batch_index],
                                           action_history=[new_action_history],
                                           score=[new_score],
                                           rnn_state=[new_rnn_state],
                                           grammar_state=[new_grammar_state],
                                           possible_actions=self.possible_actions,
                                           world=self.world,
                                           example_lisp_string=self.example_lisp_string,
                                           checklist_state=new_checklist_state,
                                           debug_info=new_debug_info)
        return new_state

    def print_action_history(self, group_index: int = None) -> None:
        scores = self.score if group_index is None else [self.score[group_index]]
        batch_indices = self.batch_indices if group_index is None else [self.batch_indices[group_index]]
        histories = self.action_history if group_index is None else [self.action_history[group_index]]
        for score, batch_index, action_history in zip(scores, batch_indices, histories):
            print('  ', score.detach().cpu().numpy()[0],
                  [self.possible_actions[batch_index][action][0] for action in action_history])

    def get_valid_actions(self) -> List[Dict[str, Tuple[torch.Tensor, torch.Tensor, List[int]]]]:
        """
        Returns a list of valid actions for each element of the group.
        """
        return [state.get_valid_actions() for state in self.grammar_state]

    def is_finished(self) -> bool:
        if len(self.batch_indices) != 1:
            raise RuntimeError("is_finished() is only defined with a group_size of 1")
        return self.grammar_state[0].is_finished()

    @classmethod
    def combine_states(cls, states: List['WikiTablesDecoderState']) -> 'WikiTablesDecoderState':
        batch_indices = [batch_index for state in states for batch_index in state.batch_indices]
        action_histories = [action_history for state in states for action_history in state.action_history]
        scores = [score for state in states for score in state.score]
        rnn_states = [rnn_state for state in states for rnn_state in state.rnn_state]
        grammar_states = [grammar_state for state in states for grammar_state in state.grammar_state]
        checklist_states = [checklist_state for state in states for checklist_state in state.checklist_state]
        if states[0].debug_info is not None:
            debug_info = [debug_info for state in states for debug_info in state.debug_info]
        else:
            debug_info = None
        return WikiTablesDecoderState(batch_indices=batch_indices,
                                      action_history=action_histories,
                                      score=scores,
                                      rnn_state=rnn_states,
                                      grammar_state=grammar_states,
                                      possible_actions=states[0].possible_actions,
                                      world=states[0].world,
                                      example_lisp_string=states[0].example_lisp_string,
                                      checklist_state=checklist_states,
                                      debug_info=debug_info)
