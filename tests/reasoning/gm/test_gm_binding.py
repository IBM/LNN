##
# Copyright 2023 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn import And, Fact, Model, Predicate, Variables

x, y = Variables("x", "y")


def get_data(p, q):
    return {
        p: {("x", "y"): Fact.TRUE},
        q: {("x",): Fact.TRUE},
    }


def test_p_x_binding():
    model = Model()
    p = Predicate("p", arity=2, model=model)
    q = Predicate("q", model=model)
    and_ = And(p("x", y), q(x))

    model.add_data(get_data(p, q))
    model.infer()

    assert p.state() == {("x", "y"): Fact.TRUE}
    assert q.state() == {("x",): Fact.TRUE}
    assert and_.state() == {("y", "x"): Fact.TRUE}


def test_q_x_binding():
    model = Model()
    p = Predicate("p", arity=2, model=model)
    q = Predicate("q", model=model)

    and_ = And(p(x, y), q("x"))
    model.add_data(get_data(p, q))
    model.infer()

    assert p.state() == {("x", "y"): Fact.TRUE}
    assert q.state() == {("x",): Fact.TRUE}
    assert and_.state() == {("x", "y"): Fact.TRUE}


def test_p_x_y_binding():
    model = Model()
    p = Predicate("p", arity=2, model=model)
    q = Predicate("q", model=model)

    and_ = And(p("x", "y"), q(x))
    model.add_knowledge(and_)
    model.add_data(get_data(p, q))
    model.infer()

    assert p.state() == {("x", "y"): Fact.TRUE}
    assert q.state() == {("x",): Fact.TRUE}
    assert and_.state() == {("x",): Fact.TRUE}


def test_p_x_and_q_x():
    model = Model()
    p = Predicate("p", arity=2, model=model)
    q = Predicate("q", model=model)
    and_ = And(p("x", y), q("x"))

    model.add_data(get_data(p, q))
    model.infer()

    assert p.state() == {("x", "y"): Fact.TRUE}
    assert q.state() == {("x",): Fact.TRUE}
    assert and_.state() == {("y",): Fact.TRUE}


def test_p_x_y_and_q_x():
    model = Model()
    p = Predicate("p", arity=2, model=model)
    q = Predicate("q", model=model)
    and_ = And(p("x", "y"), q("x"))

    model.add_data(get_data(p, q))
    model.infer()

    assert p.state() == {("x", "y"): Fact.TRUE}
    assert q.state() == {("x",): Fact.TRUE}
    assert and_.state() == {}


if __name__ == "__main__":
    test_p_x_binding()
    test_p_x_y_binding()
    test_q_x_binding()
    test_p_x_and_q_x()
    test_p_x_y_and_q_x()
