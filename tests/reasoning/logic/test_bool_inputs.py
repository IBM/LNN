from lnn import Fact, Predicate, Proposition


def test_true_proposition():
    P = Proposition("P")
    P.add_data(True)
    assert P.state() == Fact.TRUE


def test_false_proposition():
    P = Proposition("P")
    P.add_data(False)
    assert P.state() == Fact.FALSE


def test_predicate():
    P = Predicate("P")
    P.add_data({"x1": True, "x2": False})
    assert P.state() == {("x1",): Fact.TRUE, ("x2",): Fact.FALSE}


if __name__ == "__main__":
    test_true_proposition()
    test_false_proposition()
    test_predicate()
