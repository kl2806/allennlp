from typing import Callable, Dict, Generic, List, TypeVar

from allennlp.nn import util

ActionRepresentation = TypeVar('ActionRepresentation')  # pylint: disable=invalid-name

from allennlp.state_machines.states import GrammarStatelet

class ContextGrammarStatelet(GrammarStatelet):
    """
    A ``GrammarStatelet`` keeps track of the currently valid actions at every step of decoding.

    This class is relatively simple: we have a non-terminal stack which tracks which non-terminals
    we still need to expand.  At every timestep of decoding, we take an action that pops something
    off of the non-terminal stack, and possibly pushes more things on.  The grammar state is
    "finished" when the non-terminal stack is empty.

    At any point during decoding, you can query this object to get a representation of all of the
    valid actions in the current state.  The representation is something that you provide when
    constructing the initial state, in whatever form you want, and we just hold on to it for you
    and return it when you ask.  Putting this in here is purely for convenience, to group together
    pieces of state that are related to taking actions - if you want to handle the action
    representations outside of this class, that would work just fine too.

    Parameters
    ----------
    nonterminal_stack : ``List[str]``
        Holds the list of non-terminals that still need to be expanded.  This starts out as
        [START_SYMBOL], and decoding ends when this is empty.  Every time we take an action, we
        update the non-terminal stack and the context-dependent valid actions, and we use what's on
        the stack to decide which actions are valid in the current state.
    valid_actions : ``Dict[str, ActionRepresentation]``
        A mapping from non-terminals (represented as strings) to all valid expansions of that
        non-terminal.  The class that constructs this object can pick how it wants the actions to
        be represented.
    is_nonterminal : ``Callable[[str], bool]``
        A function that is used to determine whether each piece of the RHS of the action string is
        a non-terminal that needs to be added to the non-terminal stack.  You can use
        ``type_declaraction.is_nonterminal`` here, or write your own function if that one doesn't
        work for your domain.
    """
    def __init__(self,
                 nonterminal_stack: List[str],
                 valid_actions: Dict[str, ActionRepresentation],
                 is_nonterminal: Callable[[str], bool],
                 context_actions: Dict[str, Tuple[torch.Tensor, torch.Tensor, int]]) -> None:
        super().__init__()
    
    def take_action(self, production_rule: str) -> 'ContextGrammarStatelet':
        """
        Takes an action in the current grammar state, returning a new grammar state with whatever
        updates are necessary.  The production rule is assumed to be formatted as "LHS -> RHS".

        This will update the non-terminal stack.  Updating the non-terminal stack involves popping
        the non-terminal that was expanded off of the stack, then pushing on any non-terminals in
        the production rule back on the stack.

        For example, if our current ``nonterminal_stack`` is ``["r", "<e,r>", "d"]``, and
        ``action`` is ``d -> [<e,d>, e]``, the resulting stack will be ``["r", "<e,r>", "e",
        "<e,d>"]``.

        If ``self._reverse_productions`` is set to ``False`` then we push the non-terminals on in
        in their given order, which means that the first non-terminal in the production rule gets
        popped off the stack `last`.
        """
        left_side, right_side = production_rule.split(' -> ')
        assert self._nonterminal_stack[-1] == left_side, (f"Tried to expand {self._nonterminal_stack[-1]}"
                                                          f"but got rule {left_side} -> {right_side}")

        new_stack = self._nonterminal_stack[:-1]
        
        if left_side == "PropertyString":
            new_production_rule = f"PropertyType -> {right_side}"
            new_action = self._context_actions[new_production_rule]
            
            input_tensor, output_tensor, action_ids = actions['global']
            new_inputs = [input_tensor] + new_action[0] 
            input_tensor = torch.cat(new_inputs, dim=0)
            new_outputs = [output_tensor] + new_action[1] 
            output_tensor = torch.cat(new_outputs, dim=0)
            new_action_ids = action_ids + new_action[2] 
            self._valid_actions['global'] = (input_tensor, output_tensor, new_action_ids) 
            
        productions = self._get_productions_from_string(right_side)
        if self._reverse_productions:
            productions = list(reversed(productions))

        for production in productions:
            if self._is_nonterminal(production):
                new_stack.append(production)

        return ContextGrammarStatelet(nonterminal_stack=new_stack,
                               valid_actions=self._valid_actions,
                               is_nonterminal=self._is_nonterminal,
                               reverse_productions=self._reverse_productions)


