import random
import numpy
import time
from lnn import Model, And, Variable, Fact, Join


def get_groundings_and_truths(max_len, var_indices):

    joined_len = int(max_len / 4)
    n_preds = len(var_indices)
    groundings = {}
    for i in range(n_preds):
        groundings[i] = dict()

    pred_lens = [None]*n_preds
    ground_tuples = [None]*n_preds

    for i in range(n_preds):
        pred_lens[i] = random.randint(joined_len, max_len)
        n_v = len(var_indices[i])
        ground_tuples[i] = numpy.full([pred_lens[i], n_v], "", dtype=object)

    for i in range(n_preds):
        n_v = len(var_indices[i])
        for j in range(pred_lens[i]):
            for k in range(len(var_indices[i])):
                st1 = "p%d_%d_%d" % (i, k, random.randint(10, 99))
                ground_tuples[i][j][k] = st1

    a_s = [set(aa) for aa in var_indices]

    join_var_all = []
    join_pos_all = []

    for i, aa in enumerate(a_s):
        set_u = set()
        j_v = []
        j_p = []
        for j, bb in enumerate(a_s):
            if i != j:
                set_u.update(bb)
        for j, a2 in enumerate(aa):
            if a2 in set_u:
                j_v.append(a2)
                j_p.append(var_indices[i].index(a2))
        join_var_all.append(j_v)
        join_pos_all.append(j_p)

    update_pos = [None] * n_preds
    for i in range(n_preds):
        update_pos[i] = sorted(random.sample(range(0,
                                             pred_lens[i]),
                                             joined_len))
        for j in range(joined_len):
            for k in range(len(join_var_all[i])):
                st1 = "j_%d_%d" % (join_var_all[i][k], j)
                ground_tuples[i][update_pos[i][j]][join_pos_all[i][k]] = st1

    GT = dict()
    gt_truth = [None] * joined_len
    for j in range(joined_len):
        j_t = ground_tuples[0][update_pos[0][j]].tolist()
        for i in range(1, n_preds):
            for k in range(len(var_indices[i])):
                t_ = ground_tuples[i][update_pos[i][j]][k]
                if t_ not in j_t:
                    j_t.append(t_)

        if random.randint(0, 100) < 50:
            gt_truth[j] = Fact.TRUE
        else:
            gt_truth[j] = Fact.FALSE
        GT[tuple(j_t)] = gt_truth[j]

    for i in range(n_preds):
        groundings[i] = dict()
        update_pos_c = 0
        for j in range(pred_lens[i]):
            tup = tuple(ground_tuples[i][j, :].tolist())

            if update_pos_c < joined_len:
                update_pos_s = update_pos[i][update_pos_c]
                if j < update_pos_s:
                    groundings[i][tup] = Fact.TRUE
                else:
                    groundings[i][tup] = gt_truth[update_pos_c]
                    update_pos_c = update_pos_c + 1
            else:
                groundings[i][tup] = Fact.TRUE
    return groundings


def use_case1(max_len=250, join=Join.OUTER):

    random.seed(90)
    var_indices = [[0, 1], [0, 2], [0, 1, 2]]
    grounds = get_groundings_and_truths(max_len, var_indices)

    model = Model()
    x, y, z = map(Variable, ("x", "y", "z"))

    p1 = model.add_predicates(2, "p1")
    p2 = model.add_predicates(2, "p1")
    p3 = model.add_predicates(3, "p3")

    model.add_data({p1: grounds[0]})
    model.add_data({p2: grounds[1]})
    model.add_data({p3: grounds[2]})

    op = And(p1(x, y), p2(x, z), p3(x, y, z), join=join)
    model.add_knowledge(op)
    op.upward()


def use_case2(max_len=100, join=Join.OUTER):

    random.seed(91)
    var_indices = [[0, 1], [1, 2], [2, 3]]
    grounds = get_groundings_and_truths(max_len, var_indices)

    model = Model()
    x, y, z, a = map(Variable, ("x", "y", "z", "a"))

    p1 = model.add_predicates(2, "p1")
    p2 = model.add_predicates(2, "p1")
    p3 = model.add_predicates(2, "p3")

    model.add_data({p1: grounds[0]})
    model.add_data({p2: grounds[1]})
    model.add_data({p3: grounds[2]})

    op = And(p1(x, y), p2(y, z), p3(z, a), join=join)
    model.add_knowledge(op)
    op.upward()


def use_case3(max_len=400, join=Join.OUTER):

    random.seed(92)
    var_indices = [[0, 1, 2, 3, 5], [1, 2, 3, 4]]
    grounds = get_groundings_and_truths(max_len, var_indices)

    model = Model()
    x, y, z, a, b, c = map(Variable, ("x", "y", "z", "a", "b", "c"))

    p1 = model.add_predicates(5, "p1")
    p2 = model.add_predicates(4, "p2")

    model.add_data({p1: grounds[0]})
    model.add_data({p2: grounds[1]})

    op = And(p1(x, y, z, a, c), p2(y, z, a, b), join=join)
    model.add_knowledge(op)
    op.upward()


if __name__ == "__main__":
    t0 = time.time()
    use_case1()
    t1 = time.time() - t0
    print("Use Case 1 %.2f s " % (t1))

    t0 = time.time()
    use_case2()
    t2 = time.time() - t0
    print("Use Case 2 %.2f s" % (t2))

    t0 = time.time()
    use_case3()
    t3 = time.time() - t0
    print("Use Case 3 %.2f s" % (t3))
