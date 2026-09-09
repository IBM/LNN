from lnn import Fact, Predicate
import torch

P = Predicate("P")
P.add_data({"x1": Fact.TRUE, "x2": Fact.FALSE})

_true = [1.0, 1.0]
_false = [0.0, 0.0]
world = [0.0, 1.0]


def test_known_grounding():
    assert torch.all(P.get_data("x1") == torch.tensor([_true]))


def test_multiple_known_groundings():
    assert torch.all(P.get_data("x1", "x2") == torch.tensor([_true, _false]))


def test_unknown_groundings():
    assert torch.all(P.get_data("x3") == torch.tensor([world]))


def test_known_and_unknown_groundings():
    assert torch.all(
        P.get_data("x3", "x1", "x2", "x4")
        == torch.tensor([world, _true, _false, world])
    )


if __name__ == "__main__":
    test_known_grounding()
    test_multiple_known_groundings()
    test_unknown_groundings()
    test_known_and_unknown_groundings()
