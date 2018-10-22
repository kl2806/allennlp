from typing import List, Dict, Tuple, Set, Callable
from copy import copy
import numpy
from nltk import ngrams, bigrams

from parsimonious.grammar import Grammar
from parsimonious.expressions import Expression, OneOf, Sequence, Literal

from allennlp.semparse.contexts.atis_tables import * # pylint: disable=wildcard-import,unused-wildcard-import
from allennlp.semparse.contexts.atis_sql_table_context import AtisSqlTableContext, KEYWORDS, NUMERIC_NONTERMINALS
from allennlp.semparse.contexts.sql_context_utils import SqlVisitor, format_action, initialize_valid_actions

from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer

from pprint import pprint
import collections

AnonymizedToken = collections.namedtuple('AnonymizedToken', 'sql_value anonymized_token nonterminal')

def get_strings_from_utterance(tokenized_utterance: List[Token]) -> Dict[str, List[int]]:
    """
    Based on the current utterance, return a dictionary where the keys are the strings in
    the database that map to lists of the token indices that they are linked to.
    """
    # Initialize a counter for the different types that we encounter
    anonymized_counter = defaultdict(int)
    # Initialize a list of entities that we will use to anonymize, and deanonymize the query
    anonymized_tokens = {}

    string_linking_scores: Dict[str, List[int]] = defaultdict(list)

    """
    trigrams = ngrams([token.text for token in tokenized_utterance], 3)
    for index, trigram in enumerate(trigrams):
        if trigram[0] == 'st':
            natural_language_key = f'st. {trigram[2]}'.lower()
        else:
            natural_language_key = ' '.join(trigram).lower()
        for string in ATIS_TRIGGER_DICT.get(natural_language_key, []):
            anonymized_token = f'{string[1]}_{str(anonymized_counter[string[1]])}'
            tokenized_utterance = tokenized_utterance[:index] + [Token(text=anonymized_token)] + tokenized_utterance[index+3:]
            anonymized_counter[string[1]] += 1
            anonymized_tokens.append(AnonymizedToken(sql_value=string[0], anonymized_token=anonymized_token, nonterminal=string[2]))
            string_linking_scores[anonymized_token].extend([index])
            '''
            string_linking_scores[string].extend([index,
                                                  index + 1,
                                                  index + 2])
            '''
    """
    token_trigrams = ngrams([token.text for token in tokenized_utterance], 3)
    matched_trigrams= 0
    for index, trigram in enumerate(token_trigrams):
        if trigram[0] == 'st':
            natural_language_key = f'st. {trigram[2]}'.lower()
        else:
            natural_language_key = ' '.join(trigram).lower()
        
        for string in ATIS_TRIGGER_DICT.get(natural_language_key, []):
            anonymized_token = AnonymizedToken(sql_value=string[0],
                                               anonymized_token=string[1],
                                               nonterminal=string[2]) 
            if anonymized_token in anonymized_tokens:
                # If we have already seen this entity once, then don't create a new anonymized token, we just replace
                # it in the anonymized utterance.
                anonymized_token_text = f'{anonymized_token.anonymized_token}_{anonymized_tokens[anonymized_token]}'
            else:
                anonymized_token_text = f'{string[1]}_{str(anonymized_counter[string[1]])}'
                anonymized_tokens[anonymized_token] = anonymized_counter[string[1]]
                anonymized_counter[string[1]] += 1

            tokenized_utterance = tokenized_utterance[:index - matched_trigrams * 2] + \
                                  [Token(text=anonymized_token_text)] + \
                                  tokenized_utterance[index - matched_trigrams* 2 + 3:]
            '''
            string_linking_scores[string].extend([index,
                                                  index + 1])
            '''
            string_linking_scores[anonymized_token_text].extend([index - matched_trigrams * 2])
        if ' '.join(trigram).lower() in ATIS_TRIGGER_DICT:
            matched_trigrams += 1


    token_bigrams = bigrams([token.text for token in tokenized_utterance])
    matched_bigrams = 0
    for index, bigram in enumerate(token_bigrams):
        for string in ATIS_TRIGGER_DICT.get(' '.join(bigram).lower(), []):
            anonymized_token = AnonymizedToken(sql_value=string[0],
                                               anonymized_token=string[1],
                                               nonterminal=string[2]) 
            if anonymized_token in anonymized_tokens:
                # If we have already seen this entity once, then don't create a new anonymized token, we just replace
                # it in the anonymized utterance.
                anonymized_token_text = f'{anonymized_token.anonymized_token}_{anonymized_tokens[anonymized_token]}'
            else:
                anonymized_token_text = f'{string[1]}_{str(anonymized_counter[string[1]])}'
                anonymized_tokens[anonymized_token] = anonymized_counter[string[1]]
                anonymized_counter[string[1]] += 1

            tokenized_utterance = tokenized_utterance[:index - matched_bigrams] + \
                                  [Token(text=anonymized_token_text)] + \
                                  tokenized_utterance[index - matched_bigrams + 2:]
            '''
            string_linking_scores[string].extend([index,
                                                  index + 1])
            '''
            update_linking_scores(string_linking_scores, index - matched_bigrams)
            string_linking_scores[anonymized_token_text].extend([index - matched_bigrams])
        if ' '.join(bigram).lower() in ATIS_TRIGGER_DICT:
            matched_bigrams += 1

    for index, token in enumerate(tokenized_utterance):
        for string in ATIS_TRIGGER_DICT.get(token.text.lower(), []):
            # If we are anonymizing, then replace the token

            anonymized_token = f'{string[1]}_{str(anonymized_counter[string[1]])}'
            tokenized_utterance[index] = Token(text=anonymized_token)
            anonymized_tokens[AnonymizedToken(sql_value=string[0], anonymized_token=string[1], nonterminal=string[2])] = anonymized_counter[string[1]]
            string_linking_scores[anonymized_token].append(index)
            # string_linking_scores[string].append(index)
        if token.text.lower() in ATIS_TRIGGER_DICT:
            anonymized_counter[string[1]] += 1
    
    return string_linking_scores, tokenized_utterance, anonymized_tokens

def update_linking_scores(string_linking_scores, current_index):
    for string, indices in string_linking_scores.items():
        for index, index_value in enumerate(indices):
            if index_value > current_index:
                indices[index] -= 1 

def deanonymize_action_sequence(anonymized_action_sequence: List[str],
                                anonymized_tokens: Dict[AnonymizedToken, int]):
    anonymized_token_to_database_value = {(anonymized_token.nonterminal,
                                           f'{anonymized_token.anonymized_token}_{entity_counter}') : anonymized_token
                                            for anonymized_token, entity_counter in anonymized_tokens.items()}
    for index, anonymized_action in enumerate(anonymized_action_sequence):
        anonymized_token = anonymized_token_to_database_value.get((anonymized_action.split(' -> ')[0],
                                                                   anonymized_action.split(' -> ')[1][2:-2]))
        if anonymized_token:
            anonymized_action_sequence[index] = f'{anonymized_token.nonterminal} -> ["\'{anonymized_token.sql_value}\'"]'

    return anonymized_action_sequence


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
    """

    database_file = "https://s3-us-west-2.amazonaws.com/allennlp/datasets/atis/atis.db"
    sql_table_context = None

    def __init__(self,
                 utterances: List[str],
                 tokenizer: Tokenizer = None) -> None:
        if AtisWorld.sql_table_context is None:
            AtisWorld.sql_table_context = AtisSqlTableContext(ALL_TABLES,
                                                              TABLES_WITH_STRINGS,
                                                              AtisWorld.database_file)
        self.utterances: List[str] = utterances
        self.tokenizer = tokenizer if tokenizer else WordTokenizer()
        self.tokenized_utterances = [self.tokenizer.tokenize(utterance) for utterance in self.utterances]
        self.dates = self._get_dates()
        self.linked_entities, self.anonymized_tokenized_utterance, self.anonymized_tokens = self._get_linked_entities()

        entities, linking_scores = self._flatten_entities()
        # This has shape (num_entities, num_utterance_tokens).
        self.linking_scores: numpy.ndarray = linking_scores
        self.entities: List[str] = entities
        self.grammar: Grammar = self._update_grammar()
        self.valid_actions = initialize_valid_actions(self.grammar,
                                                      KEYWORDS)
        if self.anonymized_tokens:
            self.valid_actions = self._anonymize_valid_actions()


    def _anonymize_valid_actions(self):
        nonterminals_with_anonymized_tokens = {anonymized_token.nonterminal for anonymized_token in self.anonymized_tokens}
        for nonterminal in nonterminals_with_anonymized_tokens:
            anonymized_actions = []
            for anonymized_token, entity_counter in self.anonymized_tokens.items():
                if anonymized_token.nonterminal == nonterminal:
                    anonymized_actions.append(format_action(nonterminal=nonterminal,
                                                            right_hand_side=f'{anonymized_token.anonymized_token}_{entity_counter}',
                                                            is_number=True,
                                                            keywords_to_uppercase=KEYWORDS))
            self.valid_actions[nonterminal] = anonymized_actions
        return self.valid_actions


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
                                           current_tokenized_utterance: List[Token]) -> None:

        month_reverse_lookup = {str(number): string for string, number in MONTH_NUMBERS.items()}
        day_reverse_lookup = {str(number) : string for string, number in DAY_NUMBERS.items()}

        if self.dates:
            for date in self.dates:
                # Add the year linking score
                entity_linking = [0 for token in current_tokenized_utterance]
                for token_index, token in enumerate(current_tokenized_utterance):
                    if token.text == str(date.year):
                        entity_linking[token_index] = 1
                action = format_action(nonterminal='year_number',
                                       right_hand_side=str(date.year),
                                       is_number=True,
                                       keywords_to_uppercase=KEYWORDS)
                number_linking_scores[action] = ('year_number', str(date.year), entity_linking)


                entity_linking = [0 for token in current_tokenized_utterance]
                for token_index, token in enumerate(current_tokenized_utterance):
                    if token.text == month_reverse_lookup[str(date.month)]:
                        entity_linking[token_index] = 1
                action = format_action(nonterminal='month_number',
                                       right_hand_side=str(date.month),
                                       is_number=True,
                                       keywords_to_uppercase=KEYWORDS)

                number_linking_scores[action] = ('month_number', str(date.month), entity_linking)

                entity_linking = [0 for token in current_tokenized_utterance]
                for token_index, token in enumerate(current_tokenized_utterance):
                    if token.text == day_reverse_lookup[str(date.day)]:
                        entity_linking[token_index] = 1
                for bigram_index, bigram in enumerate(bigrams([token.text
                                                               for token in current_tokenized_utterance])):
                    if ' '.join(bigram) == day_reverse_lookup[str(date.day)]:
                        entity_linking[bigram_index] = 1
                        entity_linking[bigram_index + 1] = 1
                action = format_action(nonterminal='day_number',
                                       right_hand_side=str(date.day),
                                       is_number=True,
                                       keywords_to_uppercase=KEYWORDS)
                number_linking_scores[action] = ('day_number', str(date.day), entity_linking)

    def add_to_number_linking_scores(self,
                                     all_numbers: Set[str],
                                     number_linking_scores: Dict[str, Tuple[str, str, List[int]]],
                                     get_number_linking_dict: Callable[[str, List[Token]],
                                                                       Dict[str, List[int]]],
                                     current_tokenized_utterance: List[Token],
                                     nonterminal: str) -> None:
        """
        This is a helper method for adding different types of numbers (eg. starting time ranges) as entities.
        We first go through all utterances in the interaction and find the numbers of a certain type and add
        them to the set ``all_numbers``, which is initialized with default values. We want to add all numbers
        that occur in the interaction, and not just the current turn because the query could contain numbers
        that were triggered before the current turn. For each entity, we then check if it is triggered by tokens
        in the current utterance and construct the linking score.
        """
        number_linking_dict: Dict[str, List[int]] = {}
        for utterance, tokenized_utterance in zip(self.utterances, self.tokenized_utterances):
            number_linking_dict = get_number_linking_dict(utterance, tokenized_utterance)
            all_numbers.update(number_linking_dict.keys())
        all_numbers_list: List[str] = sorted(all_numbers, reverse=True)
        for number in all_numbers_list:
            entity_linking = [0 for token in current_tokenized_utterance]
            # ``number_linking_dict`` is for the last utterance here. If the number was triggered
            # before the last utterance, then it will have linking scores of 0's.
            for token_index in number_linking_dict.get(number, []):
                if token_index < len(entity_linking):
                    entity_linking[token_index] = 1
            action = format_action(nonterminal, number, is_number=True, keywords_to_uppercase=KEYWORDS)
            number_linking_scores[action] = (nonterminal, number, entity_linking)


    def _get_linked_entities(self) -> Dict[str, Dict[str, Tuple[str, str, List[int]]]]:
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
        for tokenized_utterance in self.tokenized_utterances:
            string_linking_dict, anonymized_tokenized_utterance, anonymized_tokens \
                    = get_strings_from_utterance(tokenized_utterance)
        strings_list = AtisWorld.sql_table_context.strings_list
        # strings_list.append(('flight_airline_code_string -> ["\'EA\'"]', 'EA'))
        # strings_list.append(('airline_airline_name_string-> ["\'EA\'"]', 'EA'))
        # We construct the linking scores for strings from the ``string_linking_dict`` here.
        strings_list = self.anonymize_strings_list(strings_list, anonymized_tokens) 
        for string in strings_list:
            entity_linking = [0 for token in anonymized_tokenized_utterance]
            # string_linking_dict has the strings and linking scores from the last utterance.
            # If the string is not in the last utterance, then the linking scores will be all 0.
            for token_index in string_linking_dict.get(string[1], []):
                entity_linking[token_index] = 1
            action = string[0]
            string_linking_scores[action] = (action.split(' -> ')[0], string[1], entity_linking)

        # Get time range start
        self.add_to_number_linking_scores({'0'},
                                          number_linking_scores,
                                          get_time_range_start_from_utterance,
                                          anonymized_tokenized_utterance, # add an option for unanonymized
                                          'time_range_start')

        self.add_to_number_linking_scores({'1200'},
                                          number_linking_scores,
                                          get_time_range_end_from_utterance,
                                          anonymized_tokenized_utterance,
                                          'time_range_end')

        self.add_to_number_linking_scores({'0', '1', '60', '41'},
                                          number_linking_scores,
                                          get_numbers_from_utterance,
                                          anonymized_tokenized_utterance,
                                          'number')

        self.add_to_number_linking_scores({'0'},
                                          number_linking_scores,
                                          get_costs_from_utterance,
                                          anonymized_tokenized_utterance,
                                          'fare_round_trip_cost')

        self.add_to_number_linking_scores({'0'},
                                          number_linking_scores,
                                          get_costs_from_utterance,
                                          anonymized_tokenized_utterance,
                                          'fare_one_direction_cost')

        self.add_to_number_linking_scores({'0'},
                                          number_linking_scores,
                                          get_flight_numbers_from_utterance,
                                          anonymized_tokenized_utterance,
                                          'flight_number')

        self.add_dates_to_number_linking_scores(number_linking_scores,
                                                anonymized_tokenized_utterance)



        entity_linking_scores['number'] = number_linking_scores
        entity_linking_scores['string'] = string_linking_scores

        return entity_linking_scores, anonymized_tokenized_utterance, anonymized_tokens
    
    def anonymize_strings_list(self, strings_list: List[Tuple[str, str]], anonymized_tokens: List[AnonymizedToken]):
        # We collapse the entities here, for example, if we anonymized city, then we replace an like  
        # city_string -> 'BOSTON' with city_string -> CITY_0. 

        # Get a set of nonterminals that have anonymized tokens
        nonterminals_with_anonymized_tokens = {anonymized_token.nonterminal for anonymized_token in anonymized_tokens}

        # Filter them out from the strings list
        strings_list = [string for string in strings_list if string[0].split(' -> ')[0] not in nonterminals_with_anonymized_tokens]
        
        # Add in the new nonterminals
        for anonymized_token, entity_counter in anonymized_tokens.items():
            strings_list.append((f'{anonymized_token.nonterminal} -> ["{anonymized_token.anonymized_token}_{entity_counter}"]',
                                f'{anonymized_token.anonymized_token}_{entity_counter}'))
        
        return sorted(strings_list, key=lambda string: string[0])


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

    def get_action_sequence(self, query: str) -> List[str]:
        query = self._ignore_dates(query)
        sql_visitor = SqlVisitor(self.grammar, keywords_to_uppercase=KEYWORDS)
        if query:
            action_sequence = sql_visitor.parse(query)
            if self.anonymized_tokens:
                action_sequence = self.anonymize_action_sequence(action_sequence)
            return action_sequence
            # If we are anonymizing, then we collapse some of the actions here
        return []

    def anonymize_action_sequence(self, action_sequence):
        anonymized_nonterminals = {anonymized_token.nonterminal : anonymized_token for anonymized_token in self.anonymized_tokens}
        action_to_anonymized_action = {f'{anonymized_token.nonterminal} -> ["\'{anonymized_token.sql_value}\'"]':
                                       f'{anonymized_token.nonterminal} -> ["{anonymized_token.anonymized_token}_{entity_counter}"]'
                                     for anonymized_token, entity_counter in self.anonymized_tokens.items()}

        for index, action in enumerate(action_sequence):
            nonterminal = action.split(' -> ')[0] 
            if nonterminal in anonymized_nonterminals:
                if action in action_to_anonymized_action:
                    action_sequence[index] = action_to_anonymized_action[action] 
                else:
                    anonymized_token = anonymized_nonterminals[nonterminal]
                    action_sequence[index] = f'{anonymized_token.nonterminal} -> ["{anonymized_token.anonymized_token}_0"]'
                    
            '''
            for anonymized_token, entity_counter in self.anonymized_tokens.items():
                if anonymized_token.nonterminal == action.split(' -> ')[0] and \
                   anonymized_token.sql_value == action.split(' -> ')[1][3:-3]:
                    action_sequence[index] = f'{anonymized_token.nonterminal} -> ["{anonymized_token.anonymized_token}_{entity_counter}"]'
            '''
        return action_sequence

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

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            return all([self.valid_actions == other.valid_actions,
                        numpy.array_equal(self.linking_scores, other.linking_scores),
                        self.utterances == other.utterances])
        return False
