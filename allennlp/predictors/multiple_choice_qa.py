from typing import cast, Tuple

from overrides import overrides

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.data.dataset_readers import BertMCQAReader
from allennlp.predictors.predictor import Predictor
from allennlp.semparse.contexts.quarel_utils import get_explanation, from_qr_spec_string
from allennlp.semparse.contexts.quarel_utils import words_from_entity_string, from_entity_cues_string


@Predictor.register('multiple-choice-qa')
class MultipleChoiceQAPredictor(Predictor):
    """
    Wrapper for the bert_mc_qa model.
    """
    def _my_json_to_instance(self, json_dict: JsonDict) -> Tuple[Instance, JsonDict]:
        """
        """

        # Make a cast here to satisfy mypy
        dataset_reader = cast(BertMCQAReader, self._dataset_reader)

        question_raw = json_dict['question']
        if isinstance(question_raw, str):
            question_data = dataset_reader.split_mc_question(question_raw)
        else:
            question_data = question_raw
        question_text = question_data["stem"]
        choice_text_list = [choice['text'] for choice in question_data['choices']]
        choice_labels = [choice['label'] for choice in question_data['choices']]
        context = json_dict.get("para")
        choice_context_list = [choice.get('para') for choice in question_data['choices']]

        instance = dataset_reader.text_to_instance(
            "NA",
            question_text,
            choice_text_list,
            context=context,
            choice_context_list=choice_context_list
        )

        extra_info = {'question': question_raw,
                      'choice_labels': choice_labels}

        return instance, extra_info

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        instance, _ = self._my_json_to_instance(json_dict)
        return instance

    @overrides
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        instance, return_dict = self._my_json_to_instance(inputs)
        outputs = self._model.forward_on_instance(instance)

        answer_index = outputs['answer_index']
        answer = return_dict['choice_labels'][answer_index]
        score = max(outputs['label_probs'])

        outputs['answer'] = answer
        outputs['score'] = score
        outputs['question_tokens_list'] = instance.fields['metadata']['question_tokens_list']

        return_dict.update(outputs)

        return sanitize(return_dict)
