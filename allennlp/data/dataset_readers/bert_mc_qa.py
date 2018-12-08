from typing import Dict, List, Any
import itertools
import json
import logging
import numpy

from overrides import overrides

from allennlp.common import Params
from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import ArrayField, Field, TextField, LabelField, ListField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WordTokenizer
from allennlp.data.tokenizers.word_splitter import WordpieceWordSplitter

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@DatasetReader.register("bert_mc_qa")
class BertMCQAReader(DatasetReader):
    """
    Reads a file from the AllenAI-V1-Feb2018 dataset in Json format.  This data is
    formatted as jsonl, one json-formatted instance per line.  An example of the json in the data is:

        {"id":"MCAS_2000_4_6",
        "question":{"stem":"Which technology was developed most recently?",
            "choices":[
                {"text":"cellular telephone","label":"A"},
                {"text":"television","label":"B"},
                {"text":"refrigerator","label":"C"},
                {"text":"airplane","label":"D"}
            ]},
        "answerKey":"A"
        }

    Parameters
    ----------
    tokenizer : ``Tokenizer``, optional (default=``WordTokenizer()``)
        We use this ``Tokenizer`` for both the premise and the hypothesis.  See :class:`Tokenizer`.
    token_indexers : ``Dict[str, TokenIndexer]``, optional (default=``{"tokens": SingleIdTokenIndexer()}``)
        We similarly use this for both the premise and the hypothesis.  See :class:`TokenIndexer`.
    """

    def __init__(self,
                 pretrained_model: str,
                 instance_per_choice: bool = True,
                 max_pieces: int = 512,
                 sample: int = -1,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super().__init__()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._word_splitter = WordpieceWordSplitter(pretrained_model, do_lower_case=True)
        self._max_pieces = max_pieces
        self._instance_per_choice = instance_per_choice
        self._sample = sample


    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        counter = self._sample

        with open(file_path, 'r') as data_file:
            logger.info("Reading QA instances from jsonl dataset at: %s", file_path)
            for line in data_file:
                counter -= 1
                if counter == 0:
                    break
                item_json = json.loads(line.strip())

                item_id = item_json["id"]
                question_text = item_json["question"]["stem"]

                choice_label_to_id = {}
                choice_text_list = []

                for choice_id, choice_item in enumerate(item_json["question"]["choices"]):
                    choice_label = choice_item["label"]
                    choice_label_to_id[choice_label] = choice_id

                    choice_text = choice_item["text"]

                    choice_text_list.append(choice_text)

                    is_correct = 1 if item_json['answerKey'] == choice_text else 0

                    if self._instance_per_choice:
                        yield self.text_to_instance_per_choice(
                            str(item_id)+'-'+str(choice_text),
                            question_text,
                            choice_text,
                            is_correct)


                if not self._instance_per_choice:
                    answer_id = choice_label_to_id[item_json["answerKey"]]

                    yield self.text_to_instance(item_id, question_text, choice_text_list, answer_id)


    def text_to_instance_per_choice(self,  # type: ignore
                         item_id: str,
                         question: str,
                         choice: str,
                         is_correct: int) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        qa_tokens, segment_ids = self.bert_features_from_qa(question, choice)

        fields['question'] = TextField(qa_tokens, self._token_indexers)
        fields['segment_ids'] = ArrayField(numpy.asarray(segment_ids))
        fields['label'] = LabelField(is_correct, skip_indexing=True)

        metadata = {
            "id": item_id,
            "question": question,
            "choice": choice,
            # "question_tokens": [x.text for x in question_tokens],
            # "choice_tokens_list": [[x.text for x in ct] for ct in choices_tokens_list],
        }

        fields["metadata"] = MetadataField(metadata)

        return Instance(fields)


    @overrides
    def text_to_instance(self,  # type: ignore
                         item_id: str,
                         question: str,
                         choice_list: List[str],
                         answer_id: int) -> Instance:
        # pylint: disable=arguments-differ
        fields: Dict[str, Field] = {}
        bert_inputs = [self.bert_features_from_qa(question, choice) for
                       choice in choice_list]

        qa_pair_fields = [TextField(qa_tokens, self._token_indexers) for qa_tokens, _ in bert_inputs]
        segment_ids_fields = [ArrayField(numpy.asarray(segment_ids)) for _, segment_ids in bert_inputs]


        fields['question'] = ListField(qa_pair_fields)
        fields['segment_ids'] = ListField(segment_ids_fields)
        fields['label'] = LabelField(answer_id, skip_indexing=True)

        metadata = {
            "id": item_id,
            "question_text": question,
            "choice_text_list": choice_list,
            # "question_tokens": [x.text for x in question_tokens],
            # "choice_tokens_list": [[x.text for x in ct] for ct in choices_tokens_list],
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

    def bert_features_from_qa(self, question, answer):
        cls_token = Token("[CLS]")
        sep_token = Token("[SEP]")
        question_tokens = self._word_splitter.split_words(question)
        choice_tokens = self._word_splitter.split_words(question)
        question_tokens, choice_tokens = self._truncate_tokens(question_tokens, choice_tokens, self._max_pieces - 3)
        tokens = [cls_token] + question_tokens + [sep_token] + choice_tokens + [sep_token]
        segment_ids = list(itertools.repeat(0, len(question_tokens) + 2)) + \
                      list(itertools.repeat(0, len(choice_tokens) + 1))
        return tokens, segment_ids
