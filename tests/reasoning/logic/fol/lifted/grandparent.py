from tqdm import tqdm

from lnn._utils import add_checkpoint, unpack_checkpoints, reset_checkpoints
from lnn import Model, Predicate, ForAll, Implies, Variable, Fact, World

TRUE = Fact.TRUE


def get_model():
    add_checkpoint("start")

    # Populating the model
    model = Model()
    x, y, z, a, b = map(Variable, ["x", "y", "z", "a", "b"])

    Parent = Predicate("Parent", 2)
    Mother = Predicate("Mother", 2)
    Grandparent = Predicate("GrandParent", 2)

    model.add_knowledge(
        ForAll(Implies(Mother(a, b), Parent(a, b))),
        ForAll(Implies(Parent(x, y), Implies(Parent(y, z), Grandparent(x, z)))),
    )

    model.set_query(
        ForAll(Implies(Mother(x, y), Implies(Mother(y, z), Grandparent(x, z))))
    )
    return model, (Mother, Parent, Grandparent)


def get_data(model, preds, filename: str):
    Mother, Parent, Grandparent = preds  # noqa: F401
    with open(filename, "r") as file:
        facts = dict()
        for line in file.readlines():
            src = line.lstrip("\ufeff").strip("\n")
            if src:
                pred, _ = src.split("(")
                if not eval(pred) in facts:
                    facts[eval(pred)] = dict()
                inputs = _.split(")")[0].split(",")
                facts[eval(pred)][tuple(map(str, inputs))] = TRUE
    model.add_data(facts)


def test_grounded(groundings: str = [None, "5", "10", "100"]):
    if not isinstance(groundings, list):
        groundings = [groundings]
    for db in groundings:
        model, (Mother, Parent, Grandparent) = get_model()
        if db:
            get_data(
                model,
                (Mother, Parent, Grandparent),
                f"../../../../benchmarks/Grandparent/{db}.db",
            )
        add_checkpoint("build model")

        model.infer(lifted=True)
        add_checkpoint("lifted reasoning")

        assert model.query.world_state() is World.AXIOM, (
            f"expected {model.query} as AXIOM, received " f"{model.query.world}"
        )
        # print(unpack_checkpoints())
        # reset_checkpoints()


if __name__ == "__main__":
    n = 100
    total_result = []
    for grounding in [None, "5", "10", "100"]:
        for _ in tqdm(range(n)):
            test_grounded(grounding)
        result = [t[1] for t in unpack_checkpoints() if t[0] == "lifted reasoning"]
        if result:
            print(
                f"{n} runs, {len(result)} successes, runtime: "
                f"{sum(result) / len(result)}"
            )
        total_result += result
        reset_checkpoints()
    print(
        f"TOTAL: {n} runs, {len(total_result)} successes, "
        f"runtime: {sum(total_result) / len(total_result)}"
    )
    print("success")
