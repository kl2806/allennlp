from typing import Dict, List, Any
import itertools
import json
import logging
import numpy
import re

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.common.util import JsonDict
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import ArrayField, Field, TextField, LabelField
from allennlp.data.fields import ListField, MetadataField, SequenceLabelField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.tokenizers.word_splitter import WordpieceWordSplitter

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("wordnet_links")
class WordnetLinksReader(DatasetReader):
    """

    """

    def __init__(self,
                 pretrained_model: str,
                 max_pieces: int = 512,
                 answer_only: bool = False,
                 syntax: str = None,
                 sample: int = -1) -> None:
        super().__init__()
        self._token_indexers = {'tokens': SingleIdTokenIndexer()}
        self._word_splitter = WordpieceWordSplitter(pretrained_model, do_lower_case=True)
        self._max_pieces = max_pieces
        self._sample = sample
        self._syntax = syntax
        self._answer_only = answer_only


    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        counter = self._sample
        debug = 5

        with open(file_path, 'r') as data_file:
            logger.info("Reading instances from jsonl dataset at: %s", file_path)
            for line in data_file:
                counter -= 1
                debug -= 1
                if counter == 0:
                    break
                item_json = json.loads(line.strip())

                if debug > 0:
                    logger.info(item_json)

                item_id = item_json["id"]
                args = item_json.get("args")
                link_name = item_json.get("link_name")
                definitions = item_json.get("definitions")
                examples = item_json.get("examples")
                is_correct = item_json.get("is_correct")
                yield self.text_to_instance(
                    item_id,
                    args,
                    link_name,
                    is_correct,
                    definitions,
                    examples,
                    debug)

    @overrides
    def text_to_instance(self,  # type: ignore
                         item_id: str,
                         args: List[str],
                         link_name: str,
                         is_correct: int,
                         definitions: List[str],
                         examples: List[List[str]],
                         debug: int = -1) -> Instance:

        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}

        assertion_tokens, segment_ids, arg_marks = self.bert_features(args, link_name, definitions, examples)
        assertion_field = TextField(assertion_tokens, self._token_indexers)
        segment_ids_field = SequenceLabelField(segment_ids, assertion_field)
        arg_marks_field = SequenceLabelField(arg_marks, assertion_field)

        if debug > 0:
            logger.info(f"assertion_tokens = {assertion_tokens}")
            logger.info(f"segment_ids = {segment_ids}")
            logger.info(f"arg_marks = {arg_marks}")

        fields['assertion'] = assertion_field
        fields['segment_ids'] = segment_ids_field
        fields['arg_marks'] = arg_marks_field

        if is_correct is not None:
            fields['label'] = LabelField(is_correct, skip_indexing=True)

        metadata = {
            "id": item_id,
            "args": args,
            "link_name": link_name,
            "is_correct": is_correct
        }

        fields["metadata"] = MetadataField(metadata)

        return Instance(fields)

    @staticmethod
    def _truncate_tokens(tokens_a, tokens_b, max_length):
        """
        Truncate a from the start and b from the end until total is less than max_length.
        At each step, truncate the longest one
        """
        while len(tokens_a) + len(tokens_b) > max_length:
            if len(tokens_a) > len(tokens_b):
                tokens_a.pop(0)
            else:
                tokens_b.pop()
        return tokens_a, tokens_b

    def bert_features(self, args: List[str],
                      link_name: str,
                      definitions: List[str],
                      examples: List[List[str]]):
        split_words = lambda x: self._word_splitter.split_words(x.replace("_", " "))
        cls_token = Token("[CLS]")
        sep_token = Token("[SEP]")
        args_tokens = [split_words(arg) for arg in args]
        link_name_tokens = split_words(link_name)
        arg_marks = [1] * len(args_tokens[0]) + [0] * len(link_name_tokens) + [2] * len(args_tokens[1])
        assertion_tokens = args_tokens[0] + link_name_tokens + args_tokens[1]
        context = "; ".join([" ; ".join(examples[0]), " ; ".join(examples[1]), " ; ".join(definitions)])
        context_tokens = self._word_splitter.split_words(context)
        context_tokens, assertion_tokens = self._truncate_tokens(context_tokens, assertion_tokens, self._max_pieces - 3)
        arg_marks = [0] * (len(context_tokens) +2) + arg_marks[:len(assertion_tokens)] + [0]
        tokens = [cls_token] + context_tokens + [sep_token] + assertion_tokens + [sep_token]
        segment_ids = list(itertools.repeat(0, len(context_tokens) + 2)) + \
                      list(itertools.repeat(1, len(assertion_tokens) + 1))
        return tokens, segment_ids, arg_marks
