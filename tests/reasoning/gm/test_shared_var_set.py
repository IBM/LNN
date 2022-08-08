from lnn import Predicate, Variable, And, Join, Implies, ForAll, Model, Fact


def test_1():
    """Simple theorem proving example
    Unary predicates with overlapping variable set.
    """

    x = Variable("x")
    square = Predicate(name="square")
    rectangle = Predicate(name="rectangle")

    square_rect = ForAll(
        x, Implies(square(x), rectangle(x), join=Join.OUTER), join=Join.OUTER
    )

    model = Model()
    model.add_knowledge(square, rectangle, square_rect)
    model.add_data({square: {"c": Fact.TRUE, "k": Fact.TRUE}})

    model.upward()

    assert len(rectangle.groundings) == 2, "FAILED 😔"


def test_2():
    """
    Binary and unary predicates with an overlapping variable subset.
    :return:
    """

    x, y = map(Variable, ["x", "y"])
    model = Model()  # Instantiate a model.

    enemy = Predicate("enemy", arity=2)
    hostile = Predicate("hostile")

    model.add_knowledge(
        ForAll(
            x,
            Implies(enemy(x, y, bind={y: "America"}), hostile(x), join=Join.OUTER),
            join=Join.OUTER,
        )
    )

    # Add facts to model.
    model.add_data({enemy: {("Nono", "America"): Fact.TRUE}})

    model.upward()
    assert len(hostile.groundings) == 1, "FAILED 😔"


def test_3():
    """
    Tenary and binary predicates with an overlapping variable subset.
    :return:
    """

    x, y, z = map(Variable, ["x", "y", "z"])
    model = Model()  # Instantiate a model.

    f1 = Predicate("F1", 3)
    f2 = Predicate("F2", 2)

    rule = And(f1(x, y, z), f2(x, y), join=Join.OUTER)

    model.add_knowledge(f1, f2, rule)
    model.add_data({f1: {("x1", "y1", "z1"): Fact.TRUE}})

    model.upward()

    assert len(f2.groundings) == 1, "FAILED 😔"


def test_4():
    """
    Tenary predicate (x,y,z) and 3 unary predicates (x),(y),(z) requiring
    product join and overlapping variable subset.
    :return:
    """

    x, y, z = map(Variable, ["x", "y", "z"])
    american = Predicate("american")
    hostile = Predicate("hostile")
    weapon = Predicate("weapon")
    sells = Predicate("sells", 3)

    model = Model()  # Instantiate a model.
    rule = And(american(x), weapon(y), hostile(z), sells(x, y, z), join=Join.OUTER)
    model.add_knowledge(american, hostile, weapon, rule)
    model.add_data(
        {
            american: {"West": Fact.TRUE},
            hostile: {"Nono": Fact.TRUE},
            weapon: {"m1": Fact.TRUE},
        }
    )

    model.upward()

    assert len(sells.groundings) == 1, "FAILED 😔"


if __name__ == "__main__":
    test_1()
    test_2()
    test_3()
    test_4()
    print("success")
