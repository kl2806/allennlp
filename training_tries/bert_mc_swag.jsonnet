local train_size = 73546;
local batch_size = 4;
local grad_accumulate = 4;
local num_epochs = 3;
local bert_model = "bert-base-uncased"

{
  "dataset_reader": {
    "type": "bert_mc_qa",
    "sample": -1,
    "pretrained_model": bert_model,
    "max_pieces": 256
  },
  "vocabulary": {
    "type": "wordpiece",
    "pretrained_model": bert_model,
    "namespace": "tokens"
  },
  "train_data_path": "/inputs/swag-json/swag-train.jsonl",
  "validation_data_path": "/inputs/swag-json/swag-dev.jsonl",
  "model": {
    "type": "bert_mc_qa",
    "pretrained_model": bert_model

  },
  "iterator": {
    "type": "basic",
    "batch_size": batch_size
  },
  "trainer": {

    "optimizer": {
      "type": "bert_adam",
      "weight_decay_rate": 0.01,
      "parameter_groups": [[["bias", "gamma", "beta"], {"weight_decay_rate": 0}]],
      "lr": 2e-5
    },
    "learning_rate_scheduler": {
      "type": "slanted_triangular",
      "num_epochs": num_epochs,
      "cut_frac": 0.1,
      "num_steps_per_epoch": std.ceil(train_size / batch_size),
    },
    "validation_metric": "+accuracy",
    "num_serialized_models_to_keep": 2,
    "should_log_learning_rate": true,
    "grad_accumulate_epochs": grad_accumulate,
    "num_epochs": num_epochs,
    "cuda_device": 0
  }
}
