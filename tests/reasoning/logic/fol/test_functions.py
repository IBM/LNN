# from lnn import Predicate, Variable Implies, Forall, \
#     Model, Fact, Function, Not
#
#
# def test_1():
#     """
#     :return:
#     """
#
#     x, y = map(Variable, ['x', 'y'])
#     f1 = Function(name="f")
#     assert str(f1) == "f(dim_0)", "FAILED 😔"
#
#     f2 = Function(name="f", input_dim=2)
#     assert str(f2) == "f(dim_0, dim_1)", "FAILED 😔"
#
#     f3 = Function(name='f', input_dim=3)
#     assert str(f3) == "f(dim_0, dim_1, dim_2)", "FAILED 😔"
#
#
# def test_2():
#     """
#     Recursive functions
#     :return:
#     """
#     x, y = map(Variable, ['x', 'y'])
#     model = Model()  # Instantiate a model.
#
#     integer = Predicate(name="Int")
#     model.add_knowledge(integer)
#     model.add_data(
#         {
#             integer: {"one": Fact.TRUE,
#                       "two": Fact.TRUE},
#         }
#     )
#
#     g = Function(name='g')
#     f = Function(name='f', input_dim=2)
#     for n in integer.groundings:
#         model.add_data({integer: {str(f(n, g(n))): Fact.TRUE}})
#
#     assert all([str(g) in ["one", "f(one, g(one))", "two", "f(two, g(two))"]
#                 for g in integer.groundings]), "FAILED 😔"
#
#
# def test_3():
#     """
#     :return:
#     """
#     model = Model()
#
#     person = Predicate(name="person")
#     wallet = Predicate(name="wallet")
#     wallet_of = Function(name="wallet-of")
#
#     model.add_knowledge(person, wallet)
#
#     model.add_data(
#         {
#             wallet: {wallet_of("John"): Fact.TRUE,
#                      wallet_of("Jack"): Fact.TRUE},
#             person: {"John": Fact.TRUE, "Jack": Fact.TRUE},
#         }
#     )
#     model.print()
#     GT_o = dict([("wallet-of(John)", Fact.TRUE),
#                  ("wallet-of(Jack)", Fact.TRUE)])
#     assert all([wallet.state(groundings=g) is GT_o[g] for g in GT_o]), \
#         "FAILED 😔"
#
#
# def test_4():
#     """
#     :return:
#     """
#     g, h = map(Variable, ['G', 'H'])
#     model = Model()
#     human = Predicate(name='human')
#     grade = Function(name='grade-of')
#     grade("John")
#     grade("David")
#
#     human_not_grade = Forall(h, Not(human(g, bind={g: grade(h)})), )
#     model.add_knowledge(human_not_grade)
#     model.infer()
#
#     assert all([str(gr) in ["grade-of(John)", "grade-of(David)"]
#                 for gr in human.groundings]), (
#         f"Not all groundings for {human}"
#         "are created from functions. 😔"
#     )
#
#
# def test_5():
#     """
#     Binding on homogeneous variables.
#     :return:
#     """
#
#     model = Model()
#     x, h = map(Variable, ['x', 'h'])
#     square = Predicate(name='square', arity=1)
#     rectangle = Predicate(name='rectangle', arity=1)
#     rectangle_of = Function(name='rectangle-of')
#     rectangle_of('john')
#     rectangle_of('david')
#
#     square_rect = Forall(x, Implies(square(x),
#                                     rectangle(x, bind={x: rectangle_of(h)}),
#                                     name='square-rect'),
#                          name='all-square-rect')
#
#     model.add_knowledge(square_rect)
#     model.infer()
#
#     assert all([str(g) in ["rectangle-of(john)", "rectangle-of(david)"]
#                 for g in rectangle.groundings]), (
#         f"Not all groundings for {rectangle}"
#         "are created from functions. 😔"
#     )
#
#
# def test_6():
#     """
#     Predicate bound to bound function
#     :return:
#     """
#
#     g, h = map(Variable, ['G', 'H'])
#     model = Model()
#     human = Predicate(name='human')
#     grade = Function(name='grade-of', input_dim=2)
#     grade("John", "Mary")
#     grade("David", "Grace")
#
#     human_not_grade = Forall(h, Not(human(g, bind={g: grade(h, "Mary")})),
#                              )
#     model.add_knowledge(human_not_grade)
#     model.infer()
#
#     assert all([str(g) in ['grade-of(John, Mary)']
#                 for g in human.groundings]) \
#            and all([str(g) in ['grade-of(John, Mary)',
#                                'grade-of(David, Grace)']
#                     for g in grade.groundings.values()]), (
#         f"Not all groundings for {human}"
#         "are created from functions. 😔"
#     )
#
#
# def test_7():
#     """
#     Bound compound functions
#     :return:
#     """
#
#     model = Model()
#     x, y, z, a_27a = map(Variable, ['X', 'Y', 'Z', 'A_27a'])
#     ne = Predicate(name='ne')
#     mem = Predicate(name="mem", arity=2)
#     c_2Emin_2E_3D = Function(name='c_2Emin_2E_3D')
#     arr = Function(name='arr', input_dim=2)
#     bool_const = "bool"
#     arr('a1', bool_const)
#     arr('a2', bool_const)
#     arr('a1', 'True')
#
#     formula = Forall(a_27a,
#                      Implies(ne(a_27a),
#                              mem(x, y,
#                                  bind={x: c_2Emin_2E_3D(a_27a),
#                                        y: arr(a_27a, arr(a_27a, bool_const))}),
#                              ),
#                      )
#     model.add_knowledge(formula)
#     model.infer()
#     model.print()
#
#
# def test_8():
#     """
#     Bound compound functions
#     :return:
#     """
#
#     model = Model()
#     x, y, z, a_27a = map(Variable, ['X', 'Y', 'Z', 'A_27a'])
#     ne = Predicate(name='ne')
#     mem = Predicate(name="mem", arity=2)
#     c_2Emin_2E_3D = Function(name='c_2Emin_2E_3D')
#     arr = Function(name='arr', input_dim=2)
#     arr1 = Function(name='arr1', input_dim=2)
#     bool_const = "bool"
#
#     formula = Forall(a_27a,
#                      Implies(ne(a_27a),
#                              mem(x, y,
#                                  bind={x: c_2Emin_2E_3D(a_27a),
#                                        y: arr(a_27a, arr1(a_27a, bool_const))}),
#                              ),
#                      )
#     model.add_knowledge(formula)
#     model.print()
#     model.infer()
#     model.print()
#     print(f'arr groundings: {arr.groundings}')
#     # Binding list should increase when Equality is implemented.
#
#     arr('a1', bool_const)
#     model.infer()
#     model.print()
#     print(f'arr groundings: {arr.groundings}')
#     # Binding list should increase when Equality is implemented.
#
#     arr('a2', bool_const)
#     model.infer()
#     model.print()
#     print(f'arr groundings: {arr.groundings}')
#     # Binding list should increase when Equality is implemented.
#
#     arr('a1', 'True')
#     model.infer()
#     model.print()
#     print(f'arr groundings: {arr.groundings}')
#
#
# if __name__ == "__main__":
#     test_1()
#     test_2()
#     test_3()
#     test_4()
#     test_5()
#     test_6()
#     test_7()
#     test_8()
#
