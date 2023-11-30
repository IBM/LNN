from lnn import Propositions, Xor, Model, Fact


def test_xor():
    model = Model()
    P, Q, R, S = Propositions("P", "Q", "R", "S", model=model)
    xor = Xor(P, Q, R, S)
    model.add_data(
        {
            xor: Fact.TRUE,
            P: Fact.FALSE,
            Q: Fact.TRUE,
            R: Fact.TRUE,
        }
    )
    model.infer()
    assert xor.is_contradiction(), "Xor expects only 1 True input"

    model.flush()
    model.add_data(
        {
            P: Fact.FALSE,
            Q: Fact.FALSE,
            R: Fact.TRUE,
        }
    )
    model.infer()
    assert xor.state() == Fact.UNKNOWN, "Xor cannot tell if it is valid"

    model.flush()
    model.add_data(
        {
            P: Fact.FALSE,
            Q: Fact.FALSE,
            R: Fact.TRUE,
            S: Fact.FALSE,
        }
    )
    model.infer()
    assert xor.state() == Fact.TRUE, "Xor holds upward"

    model.flush()
    model.add_data(
        {
            xor: Fact.TRUE,
            P: Fact.FALSE,
            Q: Fact.FALSE,
            R: Fact.TRUE,
        }
    )
    model.infer()
    assert S.state() == Fact.FALSE, "Xor holds downward"


if __name__ == "__main__":
    test_xor()
