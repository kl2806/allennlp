from typing import List, Dict, Tuple, NamedTuple
from collections import defaultdict

from nltk import ngrams
from allennlp.data.tokenizers import Token
from allennlp.semparse.contexts.atis_sql_table_context import KEYWORDS
from allennlp.semparse.contexts.atis_tables import * #pylint: disable=wildcard-import,unused-wildcard-import
from allennlp.semparse.contexts.sql_context_utils import format_action

class AnonymizedToken(NamedTuple):
    sql_value: str
    entity_type: EntityType

def anonymize_action_sequence(action_sequence: List[str], anonymized_tokens: Dict[AnonymizedToken, int]):
    # Get all the nonterminals that were anonymized.
    anonymized_nonterminals = {nonterminal: anonymized_token
                               for anonymized_token in anonymized_tokens
                               for nonterminal in ENTITY_TYPE_TO_NONTERMINALS[anonymized_token.entity_type]}

    action_to_anonymized_action = {f'{nonterminal} -> ["\'{anonymized_token.sql_value}\'"]':
                                   f'{nonterminal} -> ["{anonymized_token.entity_type.name}_{entity_counter}"]'
                                   for anonymized_token, entity_counter in anonymized_tokens.items()
                                   for nonterminal in ENTITY_TYPE_TO_NONTERMINALS[anonymized_token.entity_type]}

    for index, action in enumerate(action_sequence):
        nonterminal = action.split(' -> ')[0]
        if nonterminal in anonymized_nonterminals:
            if action in action_to_anonymized_action:
                action_sequence[index] = action_to_anonymized_action[action]
            else:
                anonymized_token = anonymized_nonterminals[nonterminal]
                action_sequence[index] = f'{nonterminal} -> ["{anonymized_token.entity_type.name}_0"]'

    return action_sequence

def anonymize_valid_actions(valid_actions, anonymized_tokens):
    """
    After an action is anonymized, we need to collapse the valid actions of the nonterminal
    to just the valid anonymized tokens.
    """
    nonterminals_with_anonymized_tokens = {nonterminal: anonymized_token
                                           for anonymized_token in anonymized_tokens
                                           for nonterminal in
                                           ENTITY_TYPE_TO_NONTERMINALS[anonymized_token.entity_type]}

    for nonterminal in nonterminals_with_anonymized_tokens:
        anonymized_actions = []
        for anonymized_token, entity_counter in anonymized_tokens.items():
            if nonterminal in ENTITY_TYPE_TO_NONTERMINALS[anonymized_token.entity_type]:
                anonymized_actions.append(format_action(nonterminal=nonterminal,
                                                        right_hand_side= \
                                                                (f'{anonymized_token.entity_type.name}_'
                                                                 f'{entity_counter}'),
                                                        is_number=True,
                                                        keywords_to_uppercase=KEYWORDS))
        valid_actions[nonterminal] = anonymized_actions
    return valid_actions

def anonymize_strings_list(strings_list: List[Tuple[str, str]], anonymized_tokens: Dict[AnonymizedToken, int]):
    # We collapse the entities here, for example, if we anonymized city, then we replace an like
    # city_string -> 'BOSTON' with city_string -> CITY_0.

    # Get a set of nonterminals that have anonymized tokens
    nonterminals_with_anonymized_tokens = {nonterminal: anonymized_token
                                           for anonymized_token in anonymized_tokens
                                           for nonterminal in
                                           ENTITY_TYPE_TO_NONTERMINALS[anonymized_token.entity_type]}


    # Filter them out from the strings list
    strings_list = [string for string in strings_list
                    if string[0].split(' -> ')[0] not in nonterminals_with_anonymized_tokens]

    # Add in the new nonterminals
    for anonymized_token, entity_counter in anonymized_tokens.items():
        for nonterminal in ENTITY_TYPE_TO_NONTERMINALS[anonymized_token.entity_type]:
            strings_list.append((f'{nonterminal} -> ["{anonymized_token.entity_type.name}_{entity_counter}"]',
                                 f'{anonymized_token.entity_type.name}_{entity_counter}'))

    return sorted(strings_list, key=lambda string: string[0])

def deanonymize_action_sequence(anonymized_action_sequence: List[str],
                                anonymized_tokens: Dict[AnonymizedToken, int]):
    anonymized_token_to_query_value = {(nonterminal,
                                        f'{anonymized_token.entity_type.name}_{entity_counter}'): \
                                            anonymized_token
                                       for anonymized_token, entity_counter in anonymized_tokens.items()
                                       for nonterminal
                                       in ENTITY_TYPE_TO_NONTERMINALS[anonymized_token.entity_type]}
    for index, anonymized_action in enumerate(anonymized_action_sequence):
        anonymized_token = anonymized_token_to_query_value.get((anonymized_action.split(' -> ')[0],
                                                                anonymized_action.split(' -> ')[1][2:-2]))
        if anonymized_token:
            anonymized_action_sequence[index] = \
                    f'{anonymized_action.split(" -> ")[0]} -> ["\'{anonymized_token.sql_value}\'"]'
    return anonymized_action_sequence

def get_strings_for_ngram_triggers(ngram_n: int,
                                   tokenized_utterance: List[Token],
                                   anonymized_counter: Dict[EntityType, int],
                                   anonymized_tokens: Dict[AnonymizedToken, int],
                                   string_linking_scores: Dict[str, List[int]]):
    token_ngrams = ngrams([token.text for token in tokenized_utterance], ngram_n)
    matched_ngrams = 0
    for index, ngram in enumerate(token_ngrams):
        for database_value, entity_type in ATIS_TRIGGER_DICT.get(' '.join(ngram).lower(), []):
            anonymized_token = AnonymizedToken(sql_value=database_value,
                                               entity_type=entity_type)
             # If we have seen this token before
            if anonymized_token in anonymized_tokens:
                anonymized_token_text = f'{entity_type.name}_{str(anonymized_tokens[anonymized_token])}'
            else:
                anonymized_token_text = f'{entity_type.name}_{str(anonymized_counter[entity_type])}'
                anonymized_tokens[anonymized_token] = anonymized_counter[entity_type]
                anonymized_counter[entity_type] += 1
        if ' '.join(ngram).lower() in ATIS_TRIGGER_DICT:
            if ngram_n == 2:
                update_linking_scores(string_linking_scores, index - matched_ngrams)
            offset = ngram_n - 1
            tokenized_utterance = tokenized_utterance[:index - matched_ngrams * offset] + \
                                  [Token(text=anonymized_token_text)] + \
                                  tokenized_utterance[index - matched_ngrams* offset + ngram_n:]
            string_linking_scores[anonymized_token_text].extend([index - matched_ngrams * offset])
            matched_ngrams += 1
    return tokenized_utterance

def get_strings_from_and_anonymize_utterance(tokenized_utterance: List[Token]) -> \
                                                      Tuple[Dict[str, List[int]],
                                                            List[Token],
                                                            Dict[AnonymizedToken, int]]:
    """
    Based on the current utterance, return a dictionary where the keys are the strings in
    the database that map to lists of the token indices that they are linked to.
    """
    # Initialize a counter for the different types that we encounter
    anonymized_counter: Dict[EntityType, int] = defaultdict(int)
    # Initialize a list of entities that we will use to anonymize, and deanonymize the query
    anonymized_tokens: Dict[AnonymizedToken, int] = {}
    # Initialize a dict of (sql_value, type) to counter
    string_linking_scores: Dict[str, List[int]] = defaultdict(list)

    for ngram_n in range(3, 0, -1):
        tokenized_utterance = get_strings_for_ngram_triggers(ngram_n,
                                                             tokenized_utterance,
                                                             anonymized_counter,
                                                             anonymized_tokens,
                                                             string_linking_scores)
    return string_linking_scores, tokenized_utterance, anonymized_tokens


def update_linking_scores(string_linking_scores, current_index):
    for _, indices in string_linking_scores.items():
        for index, index_value in enumerate(indices):
            if index_value > current_index:
                indices[index] -= 1
