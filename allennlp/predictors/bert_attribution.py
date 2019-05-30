
from typing import cast, Tuple

from overrides import overrides

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import Instance
from allennlp.data.dataset_readers import BertMCQAReader
from allennlp.predictors.predictor import Predictor
import torch
from allennlp.data.dataset import Batch
from allennlp.data.fields import TextField, ListField
from allennlp.data.tokenizers import Token
import allennlp.nn.util as util
from tqdm import tqdm

class FakeBertEmbeddings(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_values = None
    def forward(self, input_ids, token_type_ids=None):
        return self.embedding_values

@Predictor.register('bert-mc-attribution')
class BertMCAttributionPredictor(Predictor):
    """
    Wrapper for the bert_mc_qa model.
    """
    def __init__(self, model, dataset_reader, grad_sample_count=100, baseline_type='cls_sep_mask'):
        super().__init__(model, dataset_reader)
        self._grad = None
        self._model.eval()
        self._real_embeddings = self._model._bert_model.embeddings
        self._fake_embeddings = FakeBertEmbeddings()
        self._fake_embeddings.register_backward_hook(self.collect_grad)
        self._model._bert_model.embeddings = self._fake_embeddings
        self.grad_sample_count = grad_sample_count
        self.baseline_type = baseline_type
        self._device = next(self._model.parameters()).device

    def collect_grad(self, embedding_module, grad_input, grad_output):
        self._grad = grad_output[0]

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
            answer_id=choice_labels.index(json_dict['answerKey']),
            context=context,
            choice_context_list=choice_context_list
        )

        extra_info = {
            'question': question_raw,
            'choice_labels': choice_labels,
            'question_tokens_list': instance.fields['metadata']['question_tokens_list']
        }

        return instance, extra_info
    
    def make_all_mask(self, question_fields):
        mask_token = Token('[MASK]')
        return ListField([
            TextField([mask_token]*len(question_field.tokens), question_field._token_indexers) \
                for question_field in question_fields.field_list
        ])
    
    def make_cls_sep_mask(self, question_fields):
        mask_token = Token('[MASK]')
        return ListField([
            TextField([(t if t.text in ('[CLS]', '[SEP]') else mask_token) for t in question_field.tokens], question_field._token_indexers) \
                for question_field in question_fields.field_list
        ])

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        instance, _ = self._my_json_to_instance(json_dict)
        return instance

    @overrides
    def predict_json(self, inputs: JsonDict) -> JsonDict:
        instance, return_dict = self._my_json_to_instance(inputs)
        instance_batch = Batch([instance])
        instance_batch.index_instances(self._model.vocab)
        instance_tensors = util.tensor_dict_to_device(instance_batch.as_tensor_dict(), self._device)

        real_embedding_values = self._real_embeddings(
            util.combine_initial_dims(instance_tensors['question']['tokens']),
            util.combine_initial_dims(instance_tensors['segment_ids'])
        ).clone().detach().requires_grad_(True)
        baseline_embedding_values = None
        if self.baseline_type == 'zeros':
            baseline_embedding_values = torch.zeros_like(real_embedding_values)
        else:
            instance2, _ = self._my_json_to_instance(inputs)
            if self.baseline_type == 'all_mask':
                instance2.fields['question'] = self.make_all_mask(instance2.fields['question'])
            elif self.baseline_type == 'cls_sep_mask':
                instance2.fields['question'] = self.make_cls_sep_mask(instance2.fields['question'])
            else:
                raise RuntimeError('Invalid baseline type: '+self.baseline_type)
            instance2_batch = Batch([instance2])
            instance2_batch.index_instances(self._model.vocab)
            instance2_tensors = util.tensor_dict_to_device(instance2_batch.as_tensor_dict(), self._device)
            baseline_embedding_values = self._real_embeddings(
                util.combine_initial_dims(instance2_tensors['question']['tokens']),
                util.combine_initial_dims(instance2_tensors['segment_ids'])
            ).clone().detach().requires_grad_(True)

        grad_total = torch.zeros_like(real_embedding_values)
        # get baseline output
        self._fake_embeddings.embedding_values = baseline_embedding_values
        baseline_outputs = self._model.forward(**instance_tensors)
        baseline_loss = baseline_outputs['loss'].item()
        del baseline_outputs
        final_loss = 0
        for i in tqdm(range(self.grad_sample_count)):
            embedding_value_diff = real_embedding_values - baseline_embedding_values
            interpolated_embedding_values = baseline_embedding_values + ((i+1)/self.grad_sample_count) * embedding_value_diff
            self._fake_embeddings.embedding_values = interpolated_embedding_values
            # forward
            outputs = self._model.forward(**instance_tensors)
            final_loss = outputs['loss'].item()
            # backward
            outputs['loss'].backward()
            grad_total = grad_total + self._grad
            # cleanup
            self._model.zero_grad()
            baseline_embedding_values.grad.zero_()
            real_embedding_values.grad.zero_()
            del outputs
        
        integrated_grads = embedding_value_diff * grad_total / self.grad_sample_count
        return_dict['integrated_grads'] = integrated_grads
        print('Baseline loss:', baseline_loss)
        print('Final loss:', final_loss)
        print('Sum of integrated gradients:', torch.sum(integrated_grads).item())
        print('Loss delta:', final_loss - baseline_loss)

        return sanitize(return_dict)
