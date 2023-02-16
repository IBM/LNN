from lnn import Predicate, Variable, Exists, Implies, Forall, Model, Fact, World


def test_1():
    """Simple theorem proving example
    Square(c)
    Square(k)

    """

    x = Variable("x")
    square = Predicate("square")
    rectangle = Predicate("rectangle")
    foursides = Predicate("foursides")
    square_rect = Forall(x, Implies(square(x), rectangle(x)))
    rect_foursides = Forall(x, Implies(rectangle(x), foursides(x)))

    model = Model()
    model.add_knowledge(square_rect, rect_foursides, world=World.AXIOM)
    model.set_query(Exists(x, foursides(x)))
    model.add_data({square: {"c": Fact.TRUE, "k": Fact.TRUE}})

    steps, facts_inferred = model.infer()

    # Currently finishes in 2 inference steps when grounding on demand
    assert steps == 3, "FAILED 😔"

    GT_o = dict([("c", Fact.TRUE), ("k", Fact.TRUE)])
    model.print()
    assert all([model.query.state(groundings=g) is GT_o[g] for g in GT_o]), "FAILED 😔"


if __name__ == "__main__":
    test_1()
