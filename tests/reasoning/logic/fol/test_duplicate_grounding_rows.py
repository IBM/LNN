import torch

from lnn import And, Direction, Model, Predicate, Variables


def test_duplicate_grounding_rows():
    x, y = Variables("x", "y")

    P = Predicate("P", arity=2)
    Q = Predicate("Q")

    And_0 = And(P(x, y), Q(y))

    model = Model()
    model.add_knowledge(P, Q, And_0)
    model.add_data(
        {
            P: {("x1", "y1"): (0.3, 1.0), ("x2", "y1"): (0.7, 0.9)},
            And_0: {("x1", "y1"): (0.3, 1.0), ("x2", "y1"): (0.7, 0.9)},
        }
    )
    model.infer(direction=Direction.DOWNWARD, index=1)

    assert torch.all(Q.get_data() == torch.Tensor([[0.8, 1.0]]))


if __name__ == "__main__":
    test_duplicate_grounding_rows()
