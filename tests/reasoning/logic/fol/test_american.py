from lnn import Predicate, Variable, And, Exists, Implies, Forall, Model, Fact, World


def test_1():
    """The 'American' theorem proving example with outer joins"""

    x, y, z, w = map(Variable, ["x", "y", "z", "w"])
    model = Model()  # Instantiate a model.

    # Define and add predicates to the model.
    owns = Predicate("owns", arity=2, model=model)
    missile = Predicate("missile", model=model)
    american = Predicate("american", model=model)
    enemy = Predicate("enemy", arity=2, model=model)
    hostile = Predicate("hostile", model=model)
    criminal = Predicate("criminal", model=model)
    weapon = Predicate("weapon", model=model)
    sells = Predicate("sells", arity=3, model=model)

    # Define and add the background knowledge to  the model.

    query = Exists(
        x,
        criminal(x),
    )

    Forall(x, Implies(enemy(x, "America"), hostile(x)), world=World.AXIOM)
    Forall(
        x,
        Forall(
            y,
            Forall(
                z,
                Implies(
                    And(american(x), weapon(y), sells(x, y, z), hostile(z)),
                    criminal(x),
                ),
            ),
        ),
        world=World.AXIOM,
    )
    Forall(
        x,
        Implies(And(missile(x), owns("Nono", x)), sells("West", x, "Nono")),
        world=World.AXIOM,
    )
    Forall(
        x,
        Implies(
            missile(x),
            weapon(x),
        ),
        world=World.AXIOM,
    )

    model.set_query(query)

    # Add facts to model.
    model.add_data(
        {
            owns: {("Nono", "M1"): Fact.TRUE},
            missile: {"M1": Fact.TRUE},
            american: {"West": Fact.TRUE},
            enemy: {("Nono", "America"): Fact.TRUE},
        }
    )

    model.infer()
    GT_o = dict([("West", Fact.TRUE)])
    assert all([query.state(groundings=g) is GT_o[g] for g in GT_o]), "FAILED 😔"


def test_2():
    """The 'American' theorem proving example with  inner joins"""

    x, y, z, w = map(Variable, ["x", "y", "z", "w"])
    model = Model()  # Instantiate a model.

    # Define and add predicates to the model.
    owns = Predicate("owns", arity=2, model=model)
    missile = Predicate("missile", model=model)
    american = Predicate("american", model=model)
    enemy = Predicate("enemy", arity=2, model=model)
    hostile = Predicate("hostile", model=model)
    criminal = Predicate("criminal", model=model)
    weapon = Predicate("weapon", model=model)
    sells = Predicate("sells", arity=3, model=model)

    # Define and add the background knowledge to  the model.

    query = Exists(x, criminal(x))

    Forall(x, Implies(enemy(x, "America"), hostile(x)), world=World.AXIOM)
    Forall(
        x,
        Forall(
            y,
            Forall(
                z,
                Implies(
                    And(american(x), weapon(y), sells(x, y, z), hostile(z)),
                    criminal(x),
                ),
            ),
        ),
        world=World.AXIOM,
    )
    Forall(
        x,
        Implies(
            And(missile(x), owns("Nono", x)),
            sells("West", x, "Nono"),
        ),
        world=World.AXIOM,
    ),
    Forall(x, Implies(missile(x), weapon(x)), world=World.AXIOM)

    model.set_query(query)

    # Add facts to model.
    model.add_data(
        {
            owns: {("Nono", "M1"): Fact.TRUE},
            missile: {"M1": Fact.TRUE},
            american: {"West": Fact.TRUE},
            enemy: {("Nono", "America"): Fact.TRUE},
        }
    )

    model.infer()
    GT_o = dict([("West", Fact.TRUE)])
    assert all([model.query.state(groundings=g) is GT_o[g] for g in GT_o]), "FAILED 😔"


if __name__ == "__main__":
    test_1()
    test_2()
