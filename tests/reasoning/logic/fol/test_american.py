from lnn import (Predicate, Variable, Join, And, Exists,
                 Implies, ForAll, Model, Fact)


def test_1():
    """The 'American' theorem proving example with outer joins"""

    x, y, z, w = map(Variable, ["x", "y", "z", "w"])
    model = Model()  # Instantiate a model.

    # Define and add predicates to the model.
    owns = Predicate("owns", arity=2)
    missile = Predicate("missile")
    american = Predicate("american")
    enemy = Predicate("enemy", arity=2)
    hostile = Predicate("hostile")
    criminal = Predicate("criminal")
    weapon = Predicate("weapon")
    sells = Predicate("sells", arity=3)

    # Define and add the background knowledge to  the model.

    query = Exists(x, criminal(x), join=Join.OUTER)

    model.add_knowledge(
        ForAll(
            x,
            y,
            Implies(enemy(x, y, bind={y: "America"}), hostile(x),
                    join=Join.OUTER),
            join=Join.OUTER,
        ),
        ForAll(
            x,
            y,
            z,
            Implies(
                And(
                    american(x), weapon(y), sells(x, y, z), hostile(z),
                    join=Join.OUTER
                ),
                criminal(x),
                join=Join.OUTER,
            ),
            join=Join.OUTER,
        ),
        ForAll(
            x,
            y,
            z,
            Implies(
                And(missile(x), owns(y, x, bind={y: "Nono"}),
                    join=Join.OUTER),
                sells(z, x, y, bind={z: "West", y: "Nono"}),
                join=Join.OUTER,
            ),
            join=Join.OUTER,
        ),
        ForAll(x, Implies(missile(x), weapon(x), join=Join.OUTER),
               join=Join.OUTER),
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
    model.print()
    GT_o = dict([("West", Fact.TRUE)])
    model.print()
    assert all([model.query.state(groundings=g) is GT_o[g] for g in GT_o]), \
        "FAILED 😔"


def test_2():
    """The 'American' theorem proving example with  inner joins"""

    x, y, z, w = map(Variable, ["x", "y", "z", "w"])
    model = Model()  # Instantiate a model.

    # Define and add predicates to the model.
    owns = Predicate("owns", arity=2)
    missile = Predicate("missile")
    american = Predicate("american")
    enemy = Predicate("enemy", arity=2)
    hostile = Predicate("hostile")
    criminal = Predicate("criminal")
    weapon = Predicate("weapon")
    sells = Predicate("sells", arity=3)

    # Define and add the background knowledge to  the model.

    query = Exists(x, criminal(x))

    model.add_knowledge(
        ForAll(
            x,
            y,
            Implies(enemy(x, y, bind={y: "America"}), hostile(x)),
        ),
        ForAll(
            x,
            y,
            z,
            Implies(
                And(
                    american(x), weapon(y), sells(x, y, z), hostile(z)
                ),
                criminal(x),
            ),
        ),
        ForAll(
            x,
            y,
            z,
            Implies(
                And(missile(x), owns(y, x, bind={y: "Nono"})),
                sells(z, x, y, bind={z: "West", y: "Nono"}),
            ),
        ),
        ForAll(x, Implies(missile(x), weapon(x))),
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
    model.print()
    GT_o = dict([("West", Fact.TRUE)])
    model.print()
    assert all([model.query.state(groundings=g) is GT_o[g] for g in GT_o]), \
        "FAILED 😔"


if __name__ == "__main__":
    test_1()
    test_2()
    print("success")
