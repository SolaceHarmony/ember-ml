import importlib

def test_sequential_import():
    module = importlib.import_module("ember_ml.nn.layers.sequential")
    assert hasattr(module, "Sequential")

