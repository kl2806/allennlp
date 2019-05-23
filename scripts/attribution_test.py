
from allennlp.predictors.bert_attribution import BertMCAttributionPredictor
from allennlp.data.dataset_readers.bert_mc_qa import BertMCQAReader
import allennlp.models.bert_models
from allennlp.models.archival import load_archive
import sys
import logging



if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    a = load_archive(sys.argv[1])
    reader = BertMCQAReader.from_params(a.config['dataset_reader'])
    predictor = BertMCAttributionPredictor(a.model, reader)
