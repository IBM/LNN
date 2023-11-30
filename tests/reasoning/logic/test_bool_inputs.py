from lnn import Fact, Predicate, Proposition, Model


def test_true_proposition():
    m = Model()
    P = Proposition("P", model=m)
    P.add_data(True)
    assert P.state() == Fact.TRUE


def test_false_proposition():
    m = Model()
    P = Proposition("P", model=m)
    P.add_data(False)
    assert P.state() == Fact.FALSE


def test_predicate():
    m = Model()
    P = Predicate("P", model=m)
    P.add_data({"x1": True, "x2": False})
    assert P.state() == {("x1",): Fact.TRUE, ("x2",): Fact.FALSE}


if __name__ == "__main__":
    test_true_proposition()
    test_false_proposition()
    test_predicate()
