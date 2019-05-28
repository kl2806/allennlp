
from allennlp.predictors.bert_attribution import BertMCAttributionPredictor
from allennlp.data.dataset_readers.bert_mc_qa import BertMCQAReader
import allennlp.models.bert_models
from allennlp.models.archival import load_archive
import sys
import logging
import json

def make_predictor(model_archive_path):
    a = load_archive(model_archive_path)
    reader_conf = a.config['dataset_reader']
    reader_conf.pop('type')
    reader = BertMCQAReader.from_params(reader_conf)
    return BertMCAttributionPredictor(a.model, reader)

def test(model_archive_path, data_path):
    logging.basicConfig(level=logging.WARNING)
    predictor = make_predictor(model_archive_path)
    with open(data_path) as f:
        data_lines = f.readlines()
    example = json.loads(data_lines[0])
    return predictor, predictor.predict_json(example)

if __name__ == "__main__":
    test(sys.argv[1], sys.argv[2])
