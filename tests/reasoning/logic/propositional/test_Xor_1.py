from lnn import Propositions, XOr, Model, Fact

P, Q, R, S = Propositions("P", "Q", "R", "S")
xor = XOr(P, Q, R, S)
model = Model(knowledge=xor)
model.add_data(
    {
        xor: Fact.TRUE,
        P: Fact.FALSE,
        Q: Fact.TRUE,
        R: Fact.TRUE,
    }
)
model.infer()
print(
    xor.state(),
    P.state(),
    Q.state(),
    R.state(),
    S.state(),
)
