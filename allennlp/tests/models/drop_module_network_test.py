# pylint: disable=invalid-name,no-self-use,protected-access
from flaky import flaky

from allennlp.common.testing import ModelTestCase

class DropModuleNetworkTest(ModelTestCase):
    def setUp(self):
        super(DropModuleNetworkTest, self).setUp()
        self.set_up_model(str(self.FIXTURES_ROOT / "semantic_parsing" / "drop" / "drop.json"),
                          str(self.FIXTURES_ROOT / "data" / "drop.json"))

    @flaky
    def test_drop_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)


