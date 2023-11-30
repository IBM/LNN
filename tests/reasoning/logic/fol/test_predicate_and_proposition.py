from lnn import Model, Predicate, Fact, Variables, Proposition, And

x = Variables("x")


def test_true_proposition():
    # t-box
    model = Model()
    P = Proposition("P", model=model)
    Q = Predicate("Q", model=model)
    operator = And(P, Q(x))

    # a-box
    P.add_data(Fact.TRUE)
    q_data = {"0": Fact.TRUE, "1": Fact.FALSE, "2": Fact.UNKNOWN}
    Q.add_data(q_data)

    # ground truth
    operator_expected = q_data

    # evaluation
    operator.upward()
    assert P.state() == Fact.TRUE
    assert all([Q.state(g) is q_data[g] for g in q_data])
    assert len(Q.state()) == len(q_data)
    assert all([operator.state(g) is operator_expected[g] for g in operator_expected])
    assert len(operator.state()) == len(q_data)


def test_false_proposition():
    model = Model()
    P = Proposition("P", model=model)
    Q = Predicate("Q", model=model)
    operator = And(P, Q(x))

    P.add_data(Fact.FALSE)
    q_data = {"0": Fact.TRUE, "1": Fact.FALSE, "2": Fact.UNKNOWN}
    Q.add_data(q_data)

    operator_expected = {"0": Fact.FALSE, "1": Fact.FALSE, "2": Fact.FALSE}

    operator.upward()
    assert P.state() == Fact.FALSE
    assert all([Q.state(g) is q_data[g] for g in q_data])
    assert len(Q.state()) == len(q_data)
    assert all([operator.state(g) is operator_expected[g] for g in operator_expected])
    assert len(operator.state()) == len(operator_expected)


def test_multiple_propositions():
    model = Model()
    P = Proposition("P", model=model)
    S = Proposition("S", model=model)
    Q = Predicate("Q", model=model)
    operator = And(P, Q(x), S)

    P.add_data(Fact.TRUE)
    S.add_data(Fact.TRUE)
    q_data = {"0": Fact.TRUE, "1": Fact.FALSE, "2": Fact.UNKNOWN}
    Q.add_data(q_data)

    operator_expected = q_data

    operator.upward()
    assert P.state() == Fact.TRUE
    assert S.state() == Fact.TRUE
    assert all([Q.state(g) is q_data[g] for g in q_data])
    assert len(Q.state()) == len(q_data)
    assert all([operator.state(g) is operator_expected[g] for g in operator_expected])
    assert len(operator.state()) == len(operator_expected)


def test_multiple_predicates():
    model = Model()
    P = Proposition("P", model=model)
    S = Predicate("S", model=model)
    Q = Predicate("Q", model=model)
    operator = And(P, Q(x), S(x))

    data = {"0": Fact.TRUE, "1": Fact.FALSE, "2": Fact.UNKNOWN}

    P.add_data(Fact.TRUE)
    Q.add_data(data)
    S.add_data(data)

    operator_expected = {"0": Fact.TRUE, "1": Fact.FALSE, "2": Fact.UNKNOWN}

    operator.upward()
    assert P.state() == Fact.TRUE
    assert all([Q.state(g) is data[g] for g in data])
    assert len(Q.state()) == len(data)
    assert all([S.state(g) is data[g] for g in data])
    assert len(S.state()) == len(data)
    assert all([operator.state(g) is operator_expected[g] for g in operator_expected])
    assert len(operator.state()) == len(operator_expected)


def test_multiple_predicates_no_overlap():
    model = Model()
    P = Proposition("P", model=model)
    S = Predicate("S", model=model)
    Q = Predicate("Q", model=model)
    operatorS = And(P, Q(x), S(x))

    q_data = {"0": Fact.TRUE, "1": Fact.FALSE, "2": Fact.UNKNOWN}
    s_data = {"3": Fact.TRUE, "4": Fact.FALSE, "5": Fact.UNKNOWN}

    P.add_data(Fact.TRUE)
    Q.add_data(q_data)
    S.add_data(s_data)

    operator_expected = {
        "0": Fact.UNKNOWN,
        "1": Fact.FALSE,
        "2": Fact.UNKNOWN,
        "3": Fact.UNKNOWN,
        "4": Fact.FALSE,
        "5": Fact.UNKNOWN,
    }
    q_expected = {**q_data, "3": Fact.UNKNOWN, "4": Fact.UNKNOWN, "5": Fact.UNKNOWN}
    s_expected = {**s_data, "0": Fact.UNKNOWN, "1": Fact.UNKNOWN, "2": Fact.UNKNOWN}

    operatorS.upward()

    assert P.state() == Fact.TRUE
    assert all([Q.state(g) is q_expected[g] for g in q_expected])
    assert len(Q.state()) == len(q_expected)
    assert all([S.state(g) is s_expected[g] for g in s_expected])
    assert len(S.state()) == len(s_expected)
    assert all([operatorS.state(g) is operator_expected[g] for g in operator_expected])
    assert len(operatorS.state()) == len(operator_expected)


if __name__ == "__main__":
    test_true_proposition()
    test_false_proposition()
    test_multiple_propositions()
    test_multiple_predicates()
    test_multiple_predicates_no_overlap()
