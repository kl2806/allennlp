import numpy as np
from typing import List, Dict, Tuple, Set, Callable
import more_itertools
from copy import copy
import numpy
from nltk import ngrams, bigrams

from parsimonious.grammar import Grammar
from parsimonious.expressions import Expression, OneOf, Sequence, Literal
from parsimonious.exceptions import ParseError 

from allennlp.semparse.contexts.atis_tables import * # pylint: disable=wildcard-import,unused-wildcard-import
from allennlp.semparse.contexts.atis_sql_table_context import AtisSqlTableContext, KEYWORDS, NUMERIC_NONTERMINALS
from allennlp.semparse.contexts.atis_anonymization_utils import anonymize_strings_list, \
        get_strings_from_and_anonymize_utterance, anonymize_valid_actions, anonymize_action_sequence, \
        AnonymizedToken, deanonymize_action_sequence
from allennlp.semparse.contexts.sql_context_utils import SqlVisitor, format_action, initialize_valid_actions, \
        action_sequence_to_sql

from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from pprint import pprint
from copy import deepcopy
import collections

def get_strings_from_utterance(tokenized_utterance: List[Token]) -> Dict[str, List[int]]:
    """
    Based on the current utterance, return a dictionary where the keys are the strings in
    the database that map to lists of the token indices that they are linked to.
    """
    string_linking_scores: Dict[str, List[int]] = defaultdict(list)

    for index, token in enumerate(tokenized_utterance):
        for string, _ in ATIS_TRIGGER_DICT.get(token.text.lower(), []):
            string_linking_scores[string].append(index)

    token_bigrams = bigrams([token.text for token in tokenized_utterance])
    for index, token_bigram in enumerate(token_bigrams):
        for string, _ in ATIS_TRIGGER_DICT.get(' '.join(token_bigram).lower(), []):
            string_linking_scores[string].extend([index,
                                                  index + 1])

    trigrams = ngrams([token.text for token in tokenized_utterance], 3)
    for index, trigram in enumerate(trigrams):
        for string, _ in ATIS_TRIGGER_DICT.get(' '.join(trigram).lower(), []):
            string_linking_scores[string].extend([index,
                                                  index + 1,
                                                  index + 2])
    return string_linking_scores

class AtisWorld():
    """
    World representation for the Atis SQL domain. This class has a ``SqlTableContext`` which holds the base
    grammar, it then augments this grammar by constraining each column to the values that are allowed in it.

    Parameters
    ----------
    utterances: ``List[str]``
        A list of utterances in the interaction, the last element in this list is the
        current utterance that we are interested in.
    tokenizer: ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this tokenizer to tokenize the utterances.
    anonymize_entities: ``bool``, optional
        If is ``True``, then the entities will be replaced with special tokens with their types.
    """

    database_file = "https://s3-us-west-2.amazonaws.com/allennlp/datasets/atis/atis.db"
    sql_table_context = None

    def __init__(self,
                 utterances: List[str],
                 tokenizer: Tokenizer = None,
                 anonymize_entities: bool = True,
                 previous_action_sequences: List[str] = None,
                 linking_weight: int = 1) -> None:
        if AtisWorld.sql_table_context is None:
            AtisWorld.sql_table_context = AtisSqlTableContext(ALL_TABLES,
                                                              TABLES_WITH_STRINGS,
                                                              AtisWorld.database_file)
        self.utterances: List[str] = utterances
        self.tokenizer = tokenizer if tokenizer else WordTokenizer()
        self.anonymize_entities = anonymize_entities
        self.previous_action_sequences = previous_action_sequences
        self.linking_weight = linking_weight

        self.tokenized_utterances = [self.tokenizer.tokenize(utterance)
                                     for utterance in self.utterances]
        self.dates = self._get_dates()
        self.linked_entities, self.anonymized_tokenized_utterance, self.anonymized_tokens, self.anonymized_nonterminals= \
                self._get_linked_entities()

        entities, linking_scores = self._flatten_entities()
        # This has shape (num_entities, num_utterance_tokens).
        self.linking_scores: numpy.ndarray = linking_scores

        self.entities: List[str] = entities
        self.grammar: Grammar = self._update_grammar()
        self.valid_actions = initialize_valid_actions(self.grammar,
                                                      KEYWORDS)

        if self.anonymized_tokens:
            self.valid_actions = anonymize_valid_actions(self.valid_actions,
                                                         self.anonymized_tokens,
                                                         self.anonymized_nonterminals)
        if self.previous_action_sequences:
            self.action_subsequence_candidates = self.get_action_subsequence_candidates(self.previous_action_sequences[-1],
                                                                                        ['condition'])
            self.action_subsequence_candidates = self.filter_action_subsequence_candidates()
            self.previous_subsequence_candidates = self.get_all_previous_action_subsequences()
            self.action_subsequence_candidate_ages = \
                {create_copy_action(create_macro_action_sequence(action_subsequence_candidate))[0]: age
                                for action_subsequence_candidate, age in 
                                zip(self.action_subsequence_candidates,
                                    self.get_subsequence_candidate_ages())}


            # Map from the action to the anonymized sequence
            self.copy_actions = {create_copy_action(create_macro_action_sequence(action_subsequence_candidate))[0]: anonymize_action_sequence(action_subsequence_candidate,
                                                                                           self.anonymized_tokens,
                                                                                           self.anonymized_nonterminals)
                                for action_subsequence_candidate in self.action_subsequence_candidates}
            self.add_copy_actions(self.copy_actions)
        else:
            self.action_subsequence_candidates = []
            self.copy_actions = {}

    def _update_grammar(self):
        """
        We create a new ``Grammar`` object from the one in ``AtisSqlTableContext``, that also
        has the new entities that are extracted from the utterance. Stitching together the expressions
        to form the grammar is a little tedious here, but it is worth it because we don't have to create
        a new grammar from scratch. Creating a new grammar is expensive because we have many production
        rules that have all database values in the column on the right hand side. We update the expressions
        bottom up, since the higher level expressions may refer to the lower level ones. For example, the
        ternary expression will refer to the start and end times.
        """

        # This will give us a shallow copy. We have to be careful here because the ``Grammar`` object
        # contains ``Expression`` objects that have tuples containing the members of that expression.
        # We have to create new sub-expression objects so that original grammar is not mutated.
        new_grammar = copy(AtisWorld.sql_table_context.grammar)

        for numeric_nonterminal in NUMERIC_NONTERMINALS:
            self._add_numeric_nonterminal_to_grammar(numeric_nonterminal, new_grammar)
        self._update_expression_reference(new_grammar, 'pos_value', 'number')

        ternary_expressions = [self._get_sequence_with_spacing(new_grammar,
                                                               [new_grammar['col_ref'],
                                                                Literal('BETWEEN'),
                                                                new_grammar['time_range_start'],
                                                                Literal(f'AND'),
                                                                new_grammar['time_range_end']]),
                               self._get_sequence_with_spacing(new_grammar,
                                                               [new_grammar['col_ref'],
                                                                Literal('NOT'),
                                                                Literal('BETWEEN'),
                                                                new_grammar['time_range_start'],
                                                                Literal(f'AND'),
                                                                new_grammar['time_range_end']]),
                               self._get_sequence_with_spacing(new_grammar,
                                                               [new_grammar['col_ref'],
                                                                Literal('not'),
                                                                Literal('BETWEEN'),
                                                                new_grammar['time_range_start'],
                                                                Literal(f'AND'),
                                                                new_grammar['time_range_end']])]

        new_grammar['ternaryexpr'] = OneOf(*ternary_expressions, name='ternaryexpr')
        self._update_expression_reference(new_grammar, 'condition', 'ternaryexpr')

        new_binary_expressions = []

        fare_round_trip_cost_expression = \
                    self._get_sequence_with_spacing(new_grammar,
                                                    [Literal('fare'),
                                                     Literal('.'),
                                                     Literal('round_trip_cost'),
                                                     new_grammar['binaryop'],
                                                     new_grammar['fare_round_trip_cost']])
        new_binary_expressions.append(fare_round_trip_cost_expression)

        fare_one_direction_cost_expression = \
                    self._get_sequence_with_spacing(new_grammar,
                                                    [Literal('fare'),
                                                     Literal('.'),
                                                     Literal('one_direction_cost'),
                                                     new_grammar['binaryop'],
                                                     new_grammar['fare_one_direction_cost']])

        new_binary_expressions.append(fare_one_direction_cost_expression)

        flight_arrival_start = self._get_sequence_with_spacing(new_grammar,
                                                               [Literal('flight'),
                                                                Literal('.'),
                                                                Literal('arrival_time'),
                                                               new_grammar['binaryop'],
                                                               new_grammar['time_range_start']])
        flight_arrival_end = self._get_sequence_with_spacing(new_grammar,
                                                               [Literal('flight'),
                                                                Literal('.'),
                                                                Literal('arrival_time'),
                                                               new_grammar['binaryop'],
                                                               new_grammar['time_range_end']])
        flight_departure_start = self._get_sequence_with_spacing(new_grammar,
                                                                [Literal('flight'),
                                                                Literal('.'),
                                                                Literal('departure_time'),
                                                               new_grammar['binaryop'],
                                                               new_grammar['time_range_start']])

        flight_departure_end = self._get_sequence_with_spacing(new_grammar,
                                                               [Literal('flight'),
                                                                Literal('.'),
                                                                Literal('departure_time'),
                                                               new_grammar['binaryop'],
                                                               new_grammar['time_range_end']])

        new_binary_expressions.extend([flight_arrival_start,
                                       flight_arrival_end,
                                       flight_departure_start,
                                       flight_departure_end])

        flight_number_expression = \
                    self._get_sequence_with_spacing(new_grammar,
                                                    [Literal('flight'),
                                                     Literal('.'),
                                                     Literal('flight_number'),
                                                     new_grammar['binaryop'],
                                                     new_grammar['flight_number']])
        new_binary_expressions.append(flight_number_expression)

        if self.dates:
            year_binary_expression = self._get_sequence_with_spacing(new_grammar,
                                                                     [Literal('date_day'),
                                                                      Literal('.'),
                                                                      Literal('year'),
                                                                      new_grammar['binaryop'],
                                                                      new_grammar['year_number']])
            month_binary_expression = self._get_sequence_with_spacing(new_grammar,
                                                                      [Literal('date_day'),
                                                                       Literal('.'),
                                                                       Literal('month_number'),
                                                                       new_grammar['binaryop'],
                                                                       new_grammar['month_number']])
            day_binary_expression = self._get_sequence_with_spacing(new_grammar,
                                                                    [Literal('date_day'),
                                                                     Literal('.'),
                                                                     Literal('day_number'),
                                                                     new_grammar['binaryop'],
                                                                     new_grammar['day_number']])
            
            new_binary_expressions.extend([year_binary_expression,
                                           month_binary_expression,
                                           day_binary_expression])

        new_binary_expressions = new_binary_expressions + list(new_grammar['biexpr'].members)
        new_grammar['biexpr'] = OneOf(*new_binary_expressions, name='biexpr')
        self._update_expression_reference(new_grammar, 'condition', 'biexpr')
        return new_grammar

    def _get_numeric_database_values(self,
                                     nonterminal: str) -> List[str]:
        return sorted([value[1] for key, value in self.linked_entities['number'].items()
                       if value[0] == nonterminal], reverse=True)

    def _add_numeric_nonterminal_to_grammar(self,
                                            nonterminal: str,
                                            new_grammar: Grammar) -> None:
        numbers = self._get_numeric_database_values(nonterminal)
        if self.anonymized_tokens:
            anonymized_value_to_database_value = {f'{anonymized_token.entity_type.name}_{entity_counter}': anonymized_token.sql_value
                                                  for anonymized_token, entity_counter in self.anonymized_tokens.items()}
            numbers = sorted([anonymized_value_to_database_value.get(number, number) for number in numbers], reverse=True)

        number_literals = [Literal(number) for number in numbers]
        if number_literals:
            new_grammar[nonterminal] = OneOf(*number_literals, name=nonterminal)

    def _update_expression_reference(self, # pylint: disable=no-self-use
                                     grammar: Grammar,
                                     parent_expression_nonterminal: str,
                                     child_expression_nonterminal: str) -> None:
        """
        When we add a new expression, there may be other expressions that refer to
        it, and we need to update those to point to the new expression.
        """
        grammar[parent_expression_nonterminal].members = \
                [member if member.name != child_expression_nonterminal
                 else grammar[child_expression_nonterminal]
                 for member in grammar[parent_expression_nonterminal].members]

    def _get_sequence_with_spacing(self, # pylint: disable=no-self-use
                                   new_grammar,
                                   expressions: List[Expression],
                                   name: str = '') -> Sequence:
        """
        This is a helper method for generating sequences, since we often want a list of expressions
        with whitespaces between them.
        """
        expressions = [subexpression
                       for expression in expressions
                       for subexpression in (expression, new_grammar['ws'])]
        return Sequence(*expressions, name=name)

    def get_valid_actions(self) -> Dict[str, List[str]]:
        return self.valid_actions

    def add_dates_to_number_linking_scores(self,
                                           number_linking_scores: Dict[str, Tuple[str, str, List[int]]],
                                           current_tokenized_utterance: List[Token],
                                           anonymized_tokens: Dict[AnonymizedToken, int],
                                           string_linking_dict) -> None:

        anonymized_counter: Dict[EntityType, int] = defaultdict(int)
        month_reverse_lookup = {str(number): string for string, number in MONTH_NUMBERS.items()}
        day_reverse_lookup = {str(number) : string for string, number in DAY_NUMBERS.items()}

        if self.dates:
            for date in self.dates:
                anonymized_token_text = None
                # Add the year linking score
                entity_linking = [0 for token in current_tokenized_utterance]

                for token_index, token in enumerate(current_tokenized_utterance):
                    if token.text == str(date.year):
                        entity_type = EntityType.YEAR
                        anonymized_token = AnonymizedToken(sql_value=str(date.year), entity_type=entity_type)
                        if anonymized_token in anonymized_tokens:
                            anonymized_token_text = f'{entity_type.name}_{str(anonymized_tokens[anonymized_token])}'
                        else:
                            anonymized_token_text = f'{entity_type.name}_{str(anonymized_counter[entity_type])}'
                            anonymized_tokens[anonymized_token] = anonymized_counter[entity_type]
                            anonymized_counter[entity_type] += 1

                        current_tokenized_utterance[token_index] = Token(text=anonymized_token_text)
                        entity_linking[token_index] = self.linking_weight
                if anonymized_token_text:
                    action = format_action(nonterminal='year_number',
                                       right_hand_side=str(anonymized_token_text),
                                       is_number=True,
                                       keywords_to_uppercase=KEYWORDS)

                    number_linking_scores[action] = ('year_number', str(anonymized_token_text), entity_linking)
                else:
                    action = format_action(nonterminal='year_number',
                                       right_hand_side=str(date.year),
                                       is_number=True,
                                       keywords_to_uppercase=KEYWORDS)
                    number_linking_scores[action] = ('year_number', str(date.year), entity_linking)


                anonymized_token_text = None
                # Add the month linking score
                entity_linking = [0 for token in current_tokenized_utterance]
                for token_index, token in enumerate(current_tokenized_utterance):
                    if token.text == month_reverse_lookup[str(date.month)]:
                        entity_type = EntityType.MONTH
                        anonymized_token = AnonymizedToken(sql_value=str(date.month), entity_type=entity_type)
                        if anonymized_token in anonymized_tokens:
                            anonymized_token_text = f'{entity_type.name}_{str(anonymized_tokens[anonymized_token])}'
                        else:
                            anonymized_token_text = f'{entity_type.name}_{str(anonymized_counter[entity_type])}'
                            anonymized_tokens[anonymized_token] = anonymized_counter[entity_type]
                            anonymized_counter[entity_type] += 1

                        current_tokenized_utterance[token_index] = Token(text=anonymized_token_text)
                        entity_linking[token_index] = self.linking_weight 

                if anonymized_token_text:
                    action = format_action(nonterminal='month_number',
                                           right_hand_side=str(anonymized_token_text),
                                           is_number=True,
                                           keywords_to_uppercase=KEYWORDS)

                    # number_linking_scores[action] = ('month_number', str(date.month), entity_linking)
                    number_linking_scores[action] = ('month_number', str(anonymized_token_text), entity_linking)
                else:
                     action = format_action(nonterminal='month_number',
                                           right_hand_side=str(date.month),
                                           is_number=True,
                                           keywords_to_uppercase=KEYWORDS)

                     # number_linking_scores[action] = ('month_number', str(date.month), entity_linking)
                     number_linking_scores[action] = ('month_number', str(date.month), entity_linking)


                anonymized_token_text = None
                # Add the day number linking score
                # TODO add anonymization here
                entity_linking = [0 for token in current_tokenized_utterance]
                for token_index, token in enumerate(current_tokenized_utterance):
                    if token.text == day_reverse_lookup[str(date.day)]:
                        entity_type = EntityType.DAY
                        anonymized_token = AnonymizedToken(sql_value=str(date.day), entity_type=entity_type)
                        if anonymized_token in anonymized_tokens:
                            anonymized_token_text = f'{entity_type.name}_{str(anonymized_tokens[anonymized_token])}'
                        else:
                            anonymized_token_text = f'{entity_type.name}_{str(anonymized_counter[entity_type])}'
                            anonymized_tokens[anonymized_token] = anonymized_counter[entity_type]
                            anonymized_counter[entity_type] += 1

                        current_tokenized_utterance[token_index] = Token(text=anonymized_token_text)
                        entity_linking[token_index] = self.linking_weight 

                        # entity_linking[token_index] = self.linking_weight

                for bigram_index, bigram in enumerate(bigrams([token.text
                                                               for token in current_tokenized_utterance])):
                    if ' '.join(bigram) == day_reverse_lookup[str(date.day)]:
                        entity_type = EntityType.DAY
                        anonymized_token = AnonymizedToken(sql_value=str(date.day), entity_type=entity_type)

                        if anonymized_token in anonymized_tokens:
                            anonymized_token_text = f'{entity_type.name}_{str(anonymized_tokens[anonymized_token])}'
                        else:
                            anonymized_token_text = f'{entity_type.name}_{str(anonymized_counter[entity_type])}'
                            anonymized_tokens[anonymized_token] = anonymized_counter[entity_type]
                            anonymized_counter[entity_type] += 1

                        current_tokenized_utterance[bigram_index] = Token(text=anonymized_token_text)
                        current_tokenized_utterance[bigram_index + 1] = Token(text=anonymized_token_text)

                        entity_linking[bigram_index] = self.linking_weight
                        entity_linking[bigram_index + 1] = self.linking_weight

                if anonymized_token_text:
                    action = format_action(nonterminal='day_number',
                                           right_hand_side=str(anonymized_token_text),
                                           is_number=True,
                                           keywords_to_uppercase=KEYWORDS)

                    number_linking_scores[action] = ('day_number', str(anonymized_token_text), entity_linking)
                else:
                    action = format_action(nonterminal='day_number',
                                       right_hand_side=str(date.day),
                                       is_number=True,
                                       keywords_to_uppercase=KEYWORDS)
                    number_linking_scores[action] = ('day_number', str(date.day), entity_linking)

    def add_to_number_linking_scores(self,
                                   all_numbers: Set[str],
                                     number_linking_scores: Dict[str, Tuple[str, str, List[int]]],
                                     get_number_linking_dict: Callable[[str, List[Token], List[AnonymizedToken]],
                                                                       Dict[str, List[int]]],
                                     current_tokenized_utterance: List[Token],
                                     nonterminals: List[str],
                                     anonymized_tokens,
                                     anonymized_counter) -> None:
        """
        This is a helper method for adding different types of numbers (eg. starting time ranges) as entities.
        We first go through all utterances in the interaction and find the numbers of a certain type and add
        them to the set ``all_numbers``, which is initialized with default values. We want to add all numbers
        that occur in the interaction, and not just the current turn because the query could contain numbers
        that were triggered before the current turn. For each entity, we then check if it is triggered by tokens
        in the current utterance and construct the linking score.
        """
        number_linking_dict: Dict[str, List[int]] = {}
        
        number_linking_dict, anonymized_tokenized_utterance = get_number_linking_dict(self.utterances[-1],
                                                                                      current_tokenized_utterance,
                                                                                      anonymized_tokens,
                                                                                      anonymized_counter)
        all_numbers.update(number_linking_dict.keys())

        '''
        for utterance, tokenized_utterance in zip(self.utterances, self.tokenized_utterances):
            number_linking_dict, anonymized_tokenized_utterance = get_number_linking_dict(utterance,
                                                                                          tokenized_utterance,
                                                                                          anonymized_tokens,
                                                                                          anonymized_counter)
            all_numbers.update(number_linking_dict.keys())
        '''

        all_numbers_list: List[str] = sorted(all_numbers, reverse=True)
        for number in all_numbers_list:
            entity_linking = [0 for token in current_tokenized_utterance]
            # ``number_linking_dict`` is for the last utterance here. If the number was triggered
            # before the last utterance, then it will have linking scores of 0's.
            for token_index in number_linking_dict.get(number, []):
                if token_index < len(entity_linking):
                    entity_linking[token_index] = self.linking_weight
            for nonterminal in nonterminals:
                action = format_action(nonterminal, number, is_number=True, keywords_to_uppercase=KEYWORDS)
                number_linking_scores[action] = (nonterminal, number, entity_linking)


    def _get_linked_entities(self) -> Tuple[Dict[str, Dict[str, Tuple[str, str, List[int]]]],
                                            List[Token],
                                            Dict[AnonymizedToken, int],
                                            Dict[str, AnonymizedToken]]:
        """
        This method gets entities from the current utterance finds which tokens they are linked to.
        The entities are divided into two main groups, ``numbers`` and ``strings``. We rely on these
        entities later for updating the valid actions and the grammar.
        """
        current_tokenized_utterance = [] if not self.tokenized_utterances \
                else self.tokenized_utterances[-1]

        # We generate a dictionary where the key is the type eg. ``number`` or ``string``.
        # The value is another dictionary where the key is the action and the value is a tuple
        # of the nonterminal, the string value and the linking score.
        entity_linking_scores: Dict[str, Dict[str, Tuple[str, str, List[int]]]] = {}

        number_linking_scores: Dict[str, Tuple[str, str, List[int]]] = {}
        string_linking_scores: Dict[str, Tuple[str, str, List[int]]] = {}

        # Add string linking dict.
        string_linking_dict: Dict[str, List[int]] = {}

        anonymized_tokens = []
        anonymized_nonterminals = None
        if self.anonymize_entities:
            string_linking_dict, current_tokenized_utterance, anonymized_tokens \
                    = get_strings_from_and_anonymize_utterance(current_tokenized_utterance)
            anonymized_nonterminals = {nonterminal: anonymized_token
                                       for anonymized_token in anonymized_tokens
                                       for nonterminal in ENTITY_TYPE_TO_NONTERMINALS[anonymized_token.entity_type]}
        else:
            string_linking_dict = get_strings_from_utterance(current_tokenized_utterance)

        strings_list = AtisWorld.sql_table_context.strings_list
        if anonymized_tokens:
            strings_list = anonymize_strings_list(strings_list,
                                                  anonymized_tokens,
                                                  anonymized_nonterminals)

        
        # Get time range start
        anonymized_counter = defaultdict(int)
        self.add_to_number_linking_scores({'0'},
                                          number_linking_scores,
                                          get_time_range_start_from_utterance,
                                          current_tokenized_utterance,
                                          ['time_range_start'],
                                          anonymized_tokens,
                                          anonymized_counter)

        self.add_to_number_linking_scores({'1200'},
                                          number_linking_scores,
                                          get_time_range_end_from_utterance,
                                          current_tokenized_utterance,
                                          ['time_range_end'],
                                          anonymized_tokens,
                                          anonymized_counter)

        self.add_to_number_linking_scores({'0', '1', '60', '41'},
                                          number_linking_scores,
                                          get_numbers_from_utterance,
                                          current_tokenized_utterance,
                                          ['number'],
                                          anonymized_tokens,
                                          anonymized_counter)

        self.add_to_number_linking_scores({'0'},
                                          number_linking_scores,
                                          get_costs_from_utterance,
                                          current_tokenized_utterance,
                                          ['fare_round_trip_cost', 'fare_one_direction_cost'],
                                          anonymized_tokens,
                                          anonymized_counter)

        self.add_to_number_linking_scores({'0'},
                                          number_linking_scores,
                                          get_flight_numbers_from_utterance,
                                          current_tokenized_utterance,
                                          ['flight_number'],
                                          anonymized_tokens,
                                          anonymized_counter)

        self.add_dates_to_number_linking_scores(number_linking_scores,
                                                current_tokenized_utterance,
                                                anonymized_tokens,
                                                string_linking_dict)

        if anonymized_tokens:
            anonymized_nonterminals = {nonterminal: anonymized_token
                                       for anonymized_token in anonymized_tokens
                                       for nonterminal in ENTITY_TYPE_TO_NONTERMINALS[anonymized_token.entity_type]}
        
        entity_linking_scores['number'] = number_linking_scores
        entity_linking_scores['string'] = string_linking_scores

        # We construct the linking scores for strings from the ``string_linking_dict`` here.
        for string in strings_list:
            entity_linking = [0 for token in current_tokenized_utterance]
            # string_linking_dict has the strings and linking scores from the last utterance.
            # If the string is not in the last utterance, then the linking scores will be all 0.
            for token_index in string_linking_dict.get(string[1], []):
                entity_linking[token_index] = self.linking_weight
            action = string[0]
            string_linking_scores[action] = (action.split(' -> ')[0], string[1], entity_linking)

        return entity_linking_scores, current_tokenized_utterance, anonymized_tokens, anonymized_nonterminals

    def _get_dates(self):
        dates = []
        for tokenized_utterance in self.tokenized_utterances:
            dates.extend(get_date_from_utterance(tokenized_utterance))
        return dates

    def _ignore_dates(self, query: str):
        tokens = query.split(' ')
        year_indices = [index for index, token in enumerate(tokens) if token.endswith('year')]
        month_indices = [index for index, token in enumerate(tokens) if token.endswith('month_number')]
        day_indices = [index for index, token in enumerate(tokens) if token.endswith('day_number')]

        if self.dates:
            for token_index, token in enumerate(tokens):
                if token_index - 2 in year_indices and token.isdigit():
                    tokens[token_index] = str(self.dates[0].year)
                if token_index - 2 in month_indices and token.isdigit():
                    tokens[token_index] = str(self.dates[0].month)
                if token_index - 2 in day_indices and token.isdigit():
                    tokens[token_index] = str(self.dates[0].day)
        return ' '.join(tokens)

    def get_action_sequences(self,
                            query: str) -> List[str]:
        """
        Get a list of valid action sequences
        """
        sql_visitor = SqlVisitor(self.grammar, keywords_to_uppercase=KEYWORDS)
        action_sequences = []

        if query:
            query = self._ignore_dates(query)
            action_sequence = sql_visitor.parse(query)

            if self.anonymized_tokens:
                action_sequence = anonymize_action_sequence(action_sequence,
                                                            self.anonymized_tokens,
                                                            self.anonymized_nonterminals)
            action_sequences.append(deepcopy(action_sequence))
            if self.action_subsequence_candidates:
                action_sequence, replaced_action_subsequences = \
                    add_copy_actions_to_target_sequence(self.action_subsequence_candidates,
                                                        action_sequence)
                action_sequences[0] = action_sequence
            return action_sequences
        return []

    def all_possible_actions(self) -> List[str]:
        """
        Return a sorted list of strings representing all possible actions
        of the form: nonterminal -> [right_hand_side]
        """
        all_actions = set()
        for _, action_list in self.valid_actions.items():
            for action in action_list:
                all_actions.add(action)
        return sorted(all_actions)

    def _flatten_entities(self) -> Tuple[List[str], numpy.ndarray]:
        """
        When we first get the entities and the linking scores in ``_get_linked_entities``
        we represent as dictionaries for easier updates to the grammar and valid actions.
        In this method, we flatten them for the model so that the entities are represented as
        a list, and the linking scores are a 2D numpy array of shape (num_entities, num_utterance_tokens).
        """
        entities = []
        linking_scores = []
        for entity in sorted(self.linked_entities['number']):
            entities.append(entity)
            linking_scores.append(self.linked_entities['number'][entity][2])

        for entity in sorted(self.linked_entities['string']):
            entities.append(entity)
            linking_scores.append(self.linked_entities['string'][entity][2])

        return entities, numpy.array(linking_scores)

    def get_action_subsequence_candidates(self,
                                          previous_action_sequence: str,
                                          candidate_node_types: List[str]) -> List[str]:

        """
        Given a sequence of actions previously generated in the interaction, we want to
        extract the subsequences that represent subtrees that we potentially want to copy.
        """
        from allennlp.state_machines.states import GrammarStatelet
        from allennlp.models.semantic_parsing.atis.atis_semantic_parser import AtisSemanticParser
        action_subsequence_candidates: List[List[str]] = []
        
        # Currently the only node type we can extracting is a ``condition`` subtree in the grammar.
        for candidate_node_type in candidate_node_types:
            for index, action in enumerate(previous_action_sequence):
                if action.split(' -> ')[0] == candidate_node_type:
                    grammar_state = GrammarStatelet([candidate_node_type],
                                                    self.valid_actions,
                                                    AtisSemanticParser.is_nonterminal)
                    action_subsequence_candidate = []
                    for action in previous_action_sequence[index:]:
                        grammar_state = grammar_state.take_action(action)
                        action_subsequence_candidate.append(action)

                        # If the nonterminal stack is empty then that means we have
                        # have finished a subtree.
                        if grammar_state._nonterminal_stack == []:
                            break
                    action_subsequence_candidates.append(action_subsequence_candidate)
        return action_subsequence_candidates

    def filter_action_subsequence_candidates(self):
        """
        After generating a set of potential candidates, we want to filter out the ones the ones that contain
        that entities mentioned in the current utterance.
        """
        last_tokenized_utterance = [token.text for token in reversed(self.tokenized_utterances[-1])]
        length_of_last_utterance = last_tokenized_utterance.index('@@EOU@@')
        action_subsequence_candidates = []

        for action_subsequence_candidate in self.action_subsequence_candidates:
            anonymized_subsequence_candidate = anonymize_action_sequence(action_sequence=deepcopy(action_subsequence_candidate),
                                                                         anonymized_tokens=self.anonymized_tokens, 
                                                                         anonymized_nonterminals=self.anonymized_nonterminals)

            num_entities_mentioned_in_last_utterance = 0
            for action in anonymized_subsequence_candidate:
                if self.linked_entities['string'].get(action):
                    num_entities_mentioned_in_last_utterance += \
                        sum(self.linked_entities['string'].get(action)[2][-length_of_last_utterance:])

                if self.linked_entities['number'].get(action):
                    num_entities_mentioned_in_last_utterance += \
                        sum(self.linked_entities['number'].get(action)[2][-length_of_last_utterance:])


            if num_entities_mentioned_in_last_utterance == 0:
                action_subsequence_candidates.append(action_subsequence_candidate)

        return action_subsequence_candidates

    def get_all_previous_action_subsequences(self) -> List[List[str]]:
        previous_action_subsequences = []
        for previous_action_sequence in self.previous_action_sequences:
            action_subsequences = self.get_action_subsequence_candidates(previous_action_sequence,
                                                                         ['condition'])
            previous_action_subsequences.append(action_subsequences)
        return previous_action_subsequences
    
    def get_subsequence_candidate_ages(self) -> List[int]:
        """
        Get the relative age of a subsequence. In other words, how many turns ago the subsequence first appeared. 
    
        """
        subsequence_ages = [0 for action_subsequence_candidate in self.action_subsequence_candidates]
        for age, previous_action_subsequences in enumerate(reversed(self.previous_subsequence_candidates)):
            for subsequence_index, action_subsequence_candidate in enumerate(self.action_subsequence_candidates):
                if action_subsequence_candidate in previous_action_subsequences:
                    subsequence_ages[subsequence_index] = age + 1
        return subsequence_ages

    def get_copy_action_linking_scores(self, replaced_action_subsequences: List[str]) -> List[List[int]]:
        subsequence_linking_scores = []
        for replaced_action_subsequence in replaced_action_subsequences:
            subsequence_linking_score = [0 for token in self.anonymized_tokenized_utterance]

            anonymized_replaced_subsequence = anonymize_action_sequence(action_sequence=replaced_action_subsequence,
                                                                        anonymized_tokens=self.anonymized_tokens, 
                                                                        anonymized_nonterminals=self.anonymized_nonterminals)

            for action in replaced_action_subsequence:
                entity_linking_score = self.linked_entities['string'].get(action)
                if entity_linking_score:
                    subsequence_linking_score = [subsequence_score or entity_score
                                     for subsequence_score, entity_score in
                                     zip(subsequence_linking_score,
                                         entity_linking_score[2])]
            subsequence_linking_scores.append(subsequence_linking_score)
        return subsequence_linking_scores

    def add_copy_actions(self, copy_actions):
        self.valid_actions['condition'].extend(copy_actions.keys())

    

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return all([self.valid_actions == other.valid_actions,
                        numpy.array_equal(self.linking_scores, other.linking_scores),
                        self.utterances == other.utterances])
        return False

def create_copy_action(action_subsequence_candidate : List[List[str]]):
    right_hand_side = action_sequence_to_sql(action_subsequence_candidate[0], root_nonterminal='condition').split()
    for index, token in enumerate(right_hand_side):
        if token[:-5] in NONTERMINAL_TO_ENTITY_TYPE and token.endswith("_slot"):
            right_hand_side[index] = f'", {token[:-5]}, "'

    return [format_action(nonterminal='condition',
                         right_hand_side=' '.join(right_hand_side),
                         is_number=True)] + action_subsequence_candidate[1:]

def create_macro_action_sequence(action_subsequence_candidate):
    action_subsequence_candidate = deepcopy(action_subsequence_candidate)
    linked_actions = []
    for action_index, action in enumerate(action_subsequence_candidate):
        if not is_global_rule(action):
            linked_actions.append(action)
            nonterminal = action.split(' -> ')[0]
            action_subsequence_candidate[action_index] = f'{nonterminal} -> [{nonterminal}_slot]'
    return [action_subsequence_candidate] + linked_actions

def add_copy_actions_to_target_sequence(action_subsequence_candidates: List[List[str]],
                                        target_sequence: List[str]):
    """
    We replace subsequences in the target sequence with copy actions. We replace subsequences
    in the target sequence greedily by replacing the sequences longest to shortest.
    """
    action_subsequence_candidates = sorted(action_subsequence_candidates, key=len, reverse=True)
    replaced_action_subsequences = []

    for action_subsequence_candidate in action_subsequence_candidates:
        matches_action_subsequence = lambda *subsequence: subsequence == tuple(action_subsequence_candidate)
        copy_action = create_copy_action(create_macro_action_sequence(action_subsequence_candidate))
        new_target_sequence = list(more_itertools.replace(iterable=target_sequence,
                                   pred=matches_action_subsequence,
                                   substitutes=[copy_action],
                                   window_size=len(action_subsequence_candidate)))

        new_target_sequence = flatten_target_sequence(new_target_sequence)
        
        # If the target sequence is different, then it means the subtree was found in the target action sequence.
        if target_sequence != new_target_sequence:
            replaced_action_subsequences.append(action_subsequence_candidate)
            target_sequence = new_target_sequence

    return target_sequence, replaced_action_subsequences

def flatten_target_sequence(target_sequence):
    flat_target_sequence = []
    for action in target_sequence:
        if type(action) == str: 
            flat_target_sequence.append(action)
        else:
            flat_target_sequence.extend(action)
    return flat_target_sequence

def is_global_rule(production_rule: str) -> bool:
    nonterminal, _ = production_rule.split(' ->')
    if nonterminal in NUMERIC_NONTERMINALS:
        return False
    if nonterminal.endswith('string'):
        return False
    return True

