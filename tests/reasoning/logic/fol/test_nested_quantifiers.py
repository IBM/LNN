from lnn import And, Exists, Fact, Forall, Model, Predicate, Variables, World


def test_nested_quantifiers_extend_neuron_arity():
    x, y, z = Variables("x", "y", "z")

    model = Model()
    P = Predicate("P", arity=2, model=model)
    Q = Predicate("S", model=model)

    q1 = Forall(y, And(P(x, y), Q(z)))
    Forall(z, q1)

    model.add_data(
        {
            P: {("x1", "y1"): Fact.TRUE, ("x2", "y2"): Fact.TRUE},
            Q: {"z1": Fact.TRUE},
        }
    )
    model.infer()
    arity = sum([x.arity for x in q1.neurons])

    model.add_data({P: {("x2", "y1"): Fact.FALSE}})
    model.infer()

    assert sum([x.arity for x in q1.neurons]) > arity


def test_nested_axiom_forall_true_groundings():
    x, y, z = Variables("x", "y", "z")

    model = Model()
    P = Predicate("P", arity=2, model=model)
    Q = Predicate("Q", model=model)

    f = And(P(x, y), Q(z))
    q1 = Forall(y, f)
    q2 = Forall(z, q1, world=World.AXIOM)
    model.add_data(
        {
            P: {("x1", "y1"): Fact.TRUE, ("x2", "y2"): Fact.TRUE},
            Q: {"z1": Fact.TRUE},
        }
    )
    model.infer()

    assert all(x == Fact.TRUE for x in P.state().values())
    assert all(x == Fact.TRUE for x in Q.state().values())
    assert all(x == Fact.TRUE for x in f.state().values())
    assert all(x == Fact.TRUE for x in q1.state().values())
    assert all(x == Fact.TRUE for x in q2.state().values())


def test_nested_forall_true_groundings():
    x, y, z = Variables("x", "y", "z")

    model = Model()
    P = Predicate("P", arity=2, model=model)
    Q = Predicate("Q", model=model)

    f = And(P(x, y), Q(z))
    q1 = Forall(y, f)
    q2 = Forall(z, q1)

    model.add_data(
        {
            P: {("x1", "y1"): Fact.TRUE, ("x2", "y2"): Fact.TRUE},
            Q: {"z1": Fact.TRUE},
        }
    )
    model.infer()

    assert all(x == Fact.TRUE for x in P.state().values())
    assert all(x == Fact.TRUE for x in Q.state().values())
    assert all(x == Fact.TRUE for x in f.state().values())
    assert all(x == Fact.UNKNOWN for x in q1.state().values())
    assert all(x == Fact.UNKNOWN for x in q2.state().values())


def test_nested_axiom_forall_false_groundings():
    x, y, z = Variables("x", "y", "z")

    model = Model()
    P = Predicate("P", arity=2, model=model)
    Q = Predicate("Q", model=model)

    f = And(P(x, y), Q(z))
    q1 = Forall(y, f)
    q2 = Forall(z, q1, world=World.AXIOM)
    model.add_data(
        {
            P: {("x1", "y1"): Fact.FALSE, ("x2", "y2"): Fact.FALSE},
            Q: {"z1": Fact.FALSE},
        }
    )
    model.infer()

    assert all(x == Fact.FALSE for x in P.state().values())
    assert all(x == Fact.FALSE for x in Q.state().values())
    assert all(x == Fact.CONTRADICTION for x in f.state().values())
    assert all(x == Fact.CONTRADICTION for x in q1.state().values())
    assert all(x == Fact.CONTRADICTION for x in q2.state().values())


def test_nested_forall_false_groundings():
    x, y, z = Variables("x", "y", "z")

    model = Model()
    P = Predicate("P", arity=2, model=model)
    Q = Predicate("Q", model=model)

    f = And(P(x, y), Q(z))
    q1 = Forall(y, f)
    q2 = Forall(z, q1)

    model.add_data(
        {
            P: {("x1", "y1"): Fact.FALSE, ("x2", "y2"): Fact.FALSE},
            Q: {"z1": Fact.FALSE},
        }
    )
    model.infer()

    assert all(x == Fact.FALSE for x in P.state().values())
    assert all(x == Fact.FALSE for x in Q.state().values())
    assert all(x == Fact.FALSE for x in f.state().values())
    assert all(x == Fact.FALSE for x in q1.state().values())
    assert all(x == Fact.FALSE for x in q2.state().values())


def test_nested_axiom_forall_unknown_groundings():
    x, y, z = Variables("x", "y", "z")

    model = Model()
    P = Predicate("P", arity=2, model=model)
    Q = Predicate("Q", model=model)

    f = And(P(x, y), Q(z))
    q1 = Forall(y, f)
    q2 = Forall(z, q1, world=World.AXIOM)

    model.add_data(
        {
            P: {("x1", "y1"): Fact.UNKNOWN, ("x2", "y2"): Fact.UNKNOWN},
            Q: {"z1": Fact.UNKNOWN},
        }
    )
    model.infer()

    assert all(x == Fact.TRUE for x in P.state().values())
    assert all(x == Fact.TRUE for x in Q.state().values())
    assert all(x == Fact.TRUE for x in f.state().values())
    assert all(x == Fact.TRUE for x in q1.state().values())
    assert all(x == Fact.TRUE for x in q2.state().values())


def test_nested_forall_unknown_groundings():
    x, y, z = Variables("x", "y", "z")

    model = Model()
    P = Predicate("P", arity=2, model=model)
    Q = Predicate("Q", model=model)

    f = And(P(x, y), Q(z))
    q1 = Forall(y, f)
    q2 = Forall(z, q1)

    model.add_data(
        {
            P: {("x1", "y1"): Fact.UNKNOWN, ("x2", "y2"): Fact.UNKNOWN},
            Q: {"z1": Fact.UNKNOWN},
        }
    )
    model.infer()

    assert all(x == Fact.UNKNOWN for x in P.state().values())
    assert all(x == Fact.UNKNOWN for x in Q.state().values())
    assert all(x == Fact.UNKNOWN for x in f.state().values())
    assert all(x == Fact.UNKNOWN for x in q1.state().values())
    assert all(x == Fact.UNKNOWN for x in q2.state().values())


def test_nested_exists_true_groundings():
    x, y, z = Variables("x", "y", "z")

    model = Model()
    P = Predicate("P", arity=2, model=model)
    Q = Predicate("Q", model=model)

    f = And(P(x, y), Q(z))
    q1 = Exists(y, f)
    q2 = Exists(z, q1)

    model.add_data(
        {
            P: {("x1", "y1"): Fact.TRUE, ("x2", "y2"): Fact.TRUE},
            Q: {"z1": Fact.TRUE},
        }
    )
    model.infer()

    assert all(x == Fact.TRUE for x in P.state().values())
    assert all(x == Fact.TRUE for x in Q.state().values())
    assert all(x == Fact.TRUE for x in f.state().values())
    assert all(x == Fact.TRUE for x in q1.state().values())
    assert all(x == Fact.TRUE for x in q2.state().values())


def test_nested_exists_false_groundings():
    model = Model()
    P = Predicate("P", arity=2, model=model)
    Q = Predicate("Q", model=model)

    x, y, z = Variables("x", "y", "z")
    f = And(P(x, y), Q(z))
    q1 = Exists(y, f)
    q2 = Exists(z, q1)

    model.add_data(
        {
            P: {("x1", "y1"): Fact.FALSE, ("x2", "y2"): Fact.FALSE},
            Q: {"z1": Fact.FALSE},
        }
    )
    model.infer()

    assert all(x == Fact.FALSE for x in P.state().values())
    assert all(x == Fact.FALSE for x in Q.state().values())
    assert all(x == Fact.FALSE for x in f.state().values())
    assert all(x == Fact.UNKNOWN for x in q1.state().values())
    assert all(x == Fact.UNKNOWN for x in q2.state().values())


def test_nested_exists_unknown_groundings():
    x, y, z = Variables("x", "y", "z")

    model = Model()
    P = Predicate("P", arity=2, model=model)
    Q = Predicate("Q", model=model)

    f = And(P(x, y), Q(z))
    q1 = Exists(y, f)
    q2 = Exists(z, q1)

    model.add_data(
        {
            P: {("x1", "y1"): Fact.UNKNOWN, ("x2", "y2"): Fact.UNKNOWN},
            Q: {"z1": Fact.UNKNOWN},
        }
    )
    model.infer()

    assert all(x == Fact.UNKNOWN for x in P.state().values())
    assert all(x == Fact.UNKNOWN for x in Q.state().values())
    assert all(x == Fact.UNKNOWN for x in f.state().values())
    assert all(x == Fact.UNKNOWN for x in q1.state().values())
    assert all(x == Fact.UNKNOWN for x in q2.state().values())


def test_fully_quantified_formula():
    model = Model()
    Q = Predicate("Q", model=model)

    x = Variables("x")
    f = Forall(x, Q(x))

    model.add_data(
        {
            Q: {"z1": Fact.UNKNOWN, "z2": Fact.TRUE},
        }
    )
    model.infer()

    assert f.grounding_table is None


def test_axiom_fully_quantified_formula_as_proposition():
    x, y = Variables("x", "y")

    model = Model()
    P = Predicate("P", arity=2, model=model)
    Q = Predicate("Q", model=model)

    f = And(P(x, y), Forall(x, Q(x), world=World.AXIOM))

    model.add_data(
        {
            P: {("x1", "y1"): Fact.TRUE},
            Q: {"x1": Fact.TRUE},
        }
    )
    model.infer()

    assert all(x == Fact.TRUE for x in f.state().values())


def test_fully_quantified_formula_as_proposition():
    x, y = Variables("x", "y")

    model = Model()
    P = Predicate("P", arity=2, model=model)
    Q = Predicate("Q", model=model)

    f = And(P(x, y), Forall(x, Q(x)))

    model.add_data(
        {
            P: {("x1", "y1"): Fact.TRUE},
            Q: {"x1": Fact.TRUE},
        }
    )
    model.infer()

    assert all(x == Fact.UNKNOWN for x in f.state().values())


def test_variable_sharing():
    x, y, z = Variables("x", "y", "z")

    model = Model()
    P = Predicate("P", arity=2, model=model)
    Q = Predicate("Q", model=model)
    R = Predicate("R", model=model)

    f = And(Q(x), R(y))
    And(P(x, y), Forall(x, f))
    model.add_data(
        {
            P: {("x1", "y1"): Fact.TRUE, ("x2", "y2"): Fact.FALSE},
            Q: {"x1": Fact.UNKNOWN},
            R: {"y1": Fact.TRUE},
        }
    )
    model.infer()

    assert f.grounding_table.get(("x1", "y1")) is not None
    assert f.grounding_table.get(("x1", "y2")) is not None


def test_axiom_forall_exists_positive_example():
    x, y, z = Variables("x", "y", "z")

    model = Model()
    P = Predicate("P", arity=2, model=model)
    Q = Predicate("Q", model=model)

    f = And(P(x, y), Q(z))
    q1 = Exists(y, f)
    q2 = Forall(z, q1, world=World.AXIOM)

    model.add_data(
        {
            P: {("x1", "y1"): Fact.TRUE, ("x2", "y2"): Fact.FALSE},
            Q: {"z1": Fact.TRUE, "z2": Fact.TRUE},
        }
    )
    model.infer()

    q1_gt = {
        ("x1", "z1"): Fact.TRUE,
        ("x1", "z2"): Fact.TRUE,
        ("x2", "z1"): Fact.TRUE,
        ("x2", "z2"): Fact.TRUE,
    }
    q2_gt = {("x1",): Fact.TRUE, ("x2",): Fact.TRUE}

    assert all(q1.state(g) == q1_gt[g] for g in q1_gt)
    assert all(q2.state(g) == q2_gt[g] for g in q2_gt)


def test_forall_exists_positive_example():
    x, y, z = Variables("x", "y", "z")

    model = Model()
    P = Predicate("P", arity=2, model=model)
    Q = Predicate("Q", model=model)
    f = And(P(x, y), Q(z))
    q1 = Exists(y, f)
    q2 = Forall(z, q1)

    model.add_data(
        {
            P: {("x1", "y1"): Fact.TRUE, ("x2", "y2"): Fact.FALSE},
            Q: {"z1": Fact.TRUE, "z2": Fact.TRUE},
        }
    )
    model.infer()

    q1_gt = {
        ("x1", "z1"): Fact.TRUE,
        ("x1", "z2"): Fact.TRUE,
        ("x2", "z1"): Fact.UNKNOWN,
        ("x2", "z2"): Fact.UNKNOWN,
    }
    q2_gt = {("x1",): Fact.UNKNOWN, ("x2",): Fact.UNKNOWN}

    assert all(q1.state(g) == q1_gt[g] for g in q1_gt)
    assert all(q2.state(g) == q2_gt[g] for g in q2_gt)


def test_exists_axiom_forall_positive_example():
    x, y, z = Variables("x", "y", "z")

    model = Model()
    P = Predicate("P", arity=2, model=model)
    Q = Predicate("Q", model=model)
    f = And(P(x, y), Q(z))
    q1 = Forall(y, f, world=World.AXIOM)
    q2 = Exists(z, q1)

    model.add_data(
        {
            P: {("x1", "y1"): Fact.TRUE, ("x2", "y2"): Fact.TRUE},
            Q: {"z1": Fact.TRUE, "z2": Fact.FALSE},
        }
    )
    model.infer()

    q1_gt = {
        ("x1", "z1"): Fact.TRUE,
        ("x1", "z2"): Fact.CONTRADICTION,
        ("x2", "z1"): Fact.TRUE,
        ("x2", "z2"): Fact.CONTRADICTION,
    }
    q2_gt = {("x1",): Fact.TRUE, ("x2",): Fact.TRUE}

    assert all(q1.state(g) == q1_gt[g] for g in q1_gt)
    assert all(q2.state(g) == q2_gt[g] for g in q2_gt)


def test_exists_forall_positive_example():
    x, y, z = Variables("x", "y", "z")

    model = Model()
    P = Predicate("P", arity=2, model=model)
    Q = Predicate("Q", model=model)

    f = And(P(x, y), Q(z))
    q1 = Forall(y, f)
    q2 = Exists(z, q1)

    model.add_data(
        {
            P: {("x1", "y1"): Fact.TRUE, ("x2", "y2"): Fact.TRUE},
            Q: {"z1": Fact.TRUE, "z2": Fact.FALSE},
        }
    )
    model.infer()

    q1_gt = {
        ("x1", "z1"): Fact.UNKNOWN,
        ("x1", "z2"): Fact.FALSE,
        ("x2", "z1"): Fact.UNKNOWN,
        ("x2", "z2"): Fact.FALSE,
    }
    q2_gt = {("x1",): Fact.UNKNOWN, ("x2",): Fact.UNKNOWN}

    assert all(q1.state(g) == q1_gt[g] for g in q1_gt)
    assert all(q2.state(g) == q2_gt[g] for g in q2_gt)


if __name__ == "__main__":
    test_nested_quantifiers_extend_neuron_arity()

    test_nested_forall_true_groundings()
    test_nested_axiom_forall_true_groundings()
    test_nested_forall_unknown_groundings()
    test_nested_axiom_forall_unknown_groundings()
    test_nested_forall_false_groundings()
    test_nested_axiom_forall_false_groundings()

    test_nested_exists_true_groundings()
    test_nested_exists_false_groundings()
    test_nested_exists_unknown_groundings()

    test_fully_quantified_formula()
    test_fully_quantified_formula_as_proposition()
    test_axiom_fully_quantified_formula_as_proposition()
    test_variable_sharing()

    test_forall_exists_positive_example()
    test_axiom_forall_exists_positive_example()
    test_exists_forall_positive_example()
    test_exists_axiom_forall_positive_example()
