##
# Copyright 2023 IBM Corp. All Rights Reserved.
#
# SPDX-License-Identifier: Apache-2.0
##

from lnn import Fact

TRUE = Fact.TRUE
FALSE = Fact.FALSE


# def test_0():
#     """Quantifier with free variables, upward on predicate"""
#     x, y, z = map(Variable, ("x", "y", "z"))
#     # a, b, c = map(Variable, ("a", "b", "c"))
#     model = Model()
#     P = Predicate("P", 3)
#     R = Exists(z, y, Exists(x, P(x, y, z)))
#     model.add_knowledge(R)
#
#     model.add_data(
#         {
#             P: {
#                 ("1", "a", "u"): TRUE,
#                 ("2", "b", "v"): FALSE,
#                 ("3", "c", "u"): TRUE,
#                 ("2", "a", "u"): FALSE,
#                 ("3", "b", "v"): FALSE,
#             }
#         }
#     )
#
#     model.infer()
#     # assert R.state() is TRUE
#     # assert Some.true_groundings == set(GT), (
#     #     f'expected True groundings to be {GT}, '
#     #     f'received {Some.true_groundings}'
#     # )


# def test_1():
#     """Quantifier with free variables, upward on predicate"""
#     p, c = map(Variable, ('dbo:Person', 'dbo:City'))
#     model = Model()
#     mayor = model.add_predicates(2, 'dbo:Mayor')
#
#     # List all the mayors of Boston
#     Some = Exists(p, mayor(p, c))
#     model.add_knowledge(Some)
#
#     GT = ['dbr:Marty_Walsh_(politician)', 'dbr:Lori_Lightfoot']
#
#     model.add_data({
#         mayor: {
#             ('dbr:Kim_Janey', 'dbr:Boston'): UNKNOWN,
#             (GT[0], 'dbr:Boston'): TRUE,
#             ('dbr:Tishaura_Jones', 'dbr:St._Louis'): UNKNOWN,
#             (GT[1], 'dbr:Chicago'): TRUE}
#         })
#
#     model.upward()
#     assert Some.true_groundings == set(GT), (
#         f'expected True groundings to be {GT}, '
#         f'received {Some.true_groundings}'
#     )
#
#
# def test_2():
#     """Quantifier with free variables, upward on predicate
#     UNKNOWN result when not fully grounded
#     """
#     p, c = map(Variable, ('dbo:Person', 'dbo:City'))
#     model = Model()
#     mayor = model.add_predicates(2, 'dbo:Mayor')
#
#     # List all the mayors of Boston
#     Some = Exists(p, mayor(p, (c, ['dbr:Chicago', 'dbr:Boston'])))
#     model.add_knowledge(Some)
#     Some = Some]
#
#     GT_truth = ['dbr:Marty_Walsh_(politician)']
#     GT_bindings = [('dbr:Lori_Lightfoot', 'dbr:Chicago'),
#                    ('dbr:Kim_Janey', 'dbr:Boston'),
#                    (GT_truth[0], 'dbr:Boston')]
#
#     model.add_data({
#         mayor: {
#             GT_bindings[0]: UNKNOWN,
#             GT_bindings[1]: UNKNOWN,
#             ('dbr:Tishaura_Jones', 'dbr:St._Louis'): UNKNOWN,
#             GT_bindings[2]: TRUE,
#         }})
#
#     model.upward()
#     assert Some.groundings == set(GT_bindings), (
#         f'expected groundings to be bound to GT bindings {GT_bindings}, '
#         f'received {Some.groundings}')
#     assert Some.true_groundings == set(GT_truth), (
#         f'expected True groundings to be {GT_truth}, '
#         f'received {Some.true_groundings}')
#
#
# def test_3():
#     """Quantifier with free variables, upward on predicate
#     Single predicate truth updates quantifier truth
#     """
#     x = Variable('x')
#     model = Model()
#     A, S = model.add_predicates(1, 'A', 'S')
#     All = Forall(x, A(x), world=World.OPEN)
#     Some = Exists(x, S(x))
#     model.add_knowledge(All, Some)
#     All = All]
#     Some = Some]
#
#     model.add_data({
#         'A': {
#             '0': TRUE,
#             '1': TRUE,
#             '2': FALSE},
#         'S': {
#             '0': FALSE,
#             '1': FALSE,
#             '2': TRUE}
#         })
#
#     model.upward()
#     assert Some.state() is TRUE, (
#         f'Forall expected as TRUE, received {Some.state()}')
#     assert All.state() is FALSE, (
#         f'Exists expected as FALSE, received {All.state()}')


if __name__ == "__main__":
    # test_0()
    # test_1()
    # test_2()
    # test_3()
    print("Empty")
