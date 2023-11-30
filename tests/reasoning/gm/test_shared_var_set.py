from lnn import Predicate, Variables, And, Implies, Forall, Model, Fact


def test_1():
    """Simple theorem proving example
    Unary predicates with overlapping variable set.
    """

    model = Model()
    x = Variables("x")
    square = Predicate(name="square", model=model)
    rectangle = Predicate(name="rectangle", model=model)
    Forall(
        x,
        Implies(
            square(x),
            rectangle(x),
        ),
    )

    model.add_data({square: {"c": Fact.TRUE, "k": Fact.TRUE}})

    model.upward()

    assert len(rectangle.groundings) == 2, "FAILED 😔"


def test_2():
    """
    Binary and unary predicates with an overlapping variable subset.
    :return:
    """

    model = Model()  # Instantiate a model.
    x, y = Variables("x", "y")
    enemy = Predicate("enemy", arity=2, model=model)
    hostile = Predicate("hostile", model=model)
    Forall(x, Implies(enemy(x, "America"), hostile(x)))

    # Add facts to model.
    model.add_data({enemy: {("Nono", "America"): Fact.TRUE}})

    model.upward()
    assert len(hostile.groundings) == 1, "FAILED 😔"


def test_3():
    """
    Tenary and binary predicates with an overlapping variable subset.
    :return:
    """

    model = Model()  # Instantiate a model.
    x, y, z = Variables("x", "y", "z")
    f1 = Predicate("F1", arity=3, model=model)
    f2 = Predicate("F2", arity=2, model=model)
    And(f1(x, y, z), f2(x, y))

    model.add_data({f1: {("x1", "y1", "z1"): Fact.TRUE}})

    model.upward()

    assert len(f2.groundings) == 1, "FAILED 😔"


def test_4():
    """
    Tenary predicate (x,y,z) and 3 unary predicates (x),(y),(z) requiring
    product join and overlapping variable subset.
    :return:
    """

    # Instantiate a model.
    model = Model()
    x, y, z = Variables("x", "y", "z")
    american = Predicate("american", model=model)
    hostile = Predicate("hostile", model=model)
    weapon = Predicate("weapon", model=model)
    sells = Predicate("sells", arity=3, model=model)
    And(american(x), weapon(y), hostile(z), sells(x, y, z))

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
