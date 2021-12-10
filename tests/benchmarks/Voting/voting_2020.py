# %%
from lnn_fun import *
import scipy.io as sio 
import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import itertools


# %%

T,F,U = Formula.true(), Formula.false(), Formula.unknown()
def truth(val, flip=False): 
    if flip:
        return F if val == 1 else T
    return T if val == 1 else F
true = lambda : T
false = lambda : F
unknown = lambda : U


# %%
file_path = 'voting_lnn/train.csv'
data = pd.read_csv(file_path, header=None)
y = torch.from_numpy(data[0].astype('float32').to_numpy().reshape((-1,1)))
X = torch.from_numpy(data.drop(0, axis='columns').astype('float32').to_numpy())
num_predicates = X.shape[1]
loader = DataLoader(TensorDataset(X, y), batch_size=32, shuffle=True)
preds = open('voting_lnn/train').readline().split('\n')[0].split(',')
democrat = preds[0]
preds = preds[1:]


# %%
for i,P in enumerate(preds):
    print(i,P)


# %%
Xn = (2*X-1)
yn = (2*y-1)
features = []
for p in range(len(preds)):
    features.append(round(abs((len(np.where(X[:,p] == y[:,0])[0])/len(X))-0.5),3))
sorted(zip(abs((Xn.T@yn)/len(X)), preds))[: : -1]

# %% [markdown]
# sorted(zip(abs((Xn.T@Xn[:,15])/len(X)), preds))[::-1]

# %%
import itertools 
model = LNN(learning=True, alpha=0.9545, learn_alpha=True, alpha_per_node=True)
P_list = [0, 10, 15]#[4]#[4,6,14] #range(len(preds))# [0,1,2,3] [4,6,14]
P_list_named = [preds[P] for P in P_list]
# choose_range = 3
# choose_list = range(1, choose_range+1)
choose_list = [1,2]#[1,2] #[2]

P_combination_list = []
for r in choose_list:
    for combination in list(itertools.combinations(P_list_named, r=r)):
        P_combination_list.append(combination)

# print(len(P_combination_list), P_combination_list)

sub_rules = []
for P_combined in P_combination_list:

    combined = []
    for P_name in P_combined:
        model[f'Not_{str(P_name)}'] = Not((P_name, 0), lnn=model)
        model[f'Combined_{str(P_name)}'] = And((P_name, 0), model[f'Not_{str(P_name)}'])
        combined.append(model[f'Combined_{str(P_name)}'])

    if len(P_combined) > 1:
        model[f'And{str(P_combined)}'] = And(*combined, weight_learning=True)
        sub_rules.append(model[f'And{str(P_combined)}'])
    else:
        sub_rules.append(combined[0])

model['rule'] = Or(*sub_rules, lnn=model, weight_learning=True)

# print(model['rule'].operands, '\n')

model.print_graph()


# %%
for i in range(X.shape[0]):
    for P in P_list:
        model[preds[P]][str(i)] = truth(X[i, P])
    model['rule'].target_bounds[model.grounding(str(i))] = truth(y[i, 0])

loss = model.train(direction='forward', epochs=20, progress_training=True, lr=1, w_min=0, force=True, iPPP=True)


# %%
print(loss)


# %%
model['rule'].print_graph(weights=True, groundings=[])
print(model['rule'].operands, '\n')

for s in model['rule'].operands:
    print(s.print_graph(weights=True, groundings=[]))
    for ss in s.operands:
        print(ss.print_graph(weights=True, groundings=[]))
        print(ss.operands, '\n')


# %%
model.print_graph(alpha=True, groundings=[])


# %%
# training results
model.reset_bounds()
for i in range(X.shape[0]):
    for P in P_list:
        model[preds[P]][str(i)] = truth(X[i, P])

model.propagate(direction='forward')

predictions_train = []
for person in range(X.shape[0]):
    predictions_train.append(model['rule'][model.grounding(str(person))].mean().round())

print(sum(np.array(predictions_train) == np.array(y[:,0])) , '/', len(y))


# %%
# test results
file_path = 'voting_lnn/test.csv'
data = pd.read_csv(file_path, header=None)
y_hat = torch.from_numpy(data[0].astype('float32').to_numpy().reshape((-1,1)))
X_hat = torch.from_numpy(data.drop(0, axis='columns').astype('float32').to_numpy())
num_predicates_hat = X.shape[1]

model.reset_bounds()
for i in range(X_hat.shape[0]):
    for P in P_list:
        model[preds[P]][str(i)] = truth(X_hat[i, P])

model.propagate(direction='forward')

predictions = []
for person in range(X_hat.shape[0]):
    predictions.append(model['rule'][model.grounding(str(person))].mean().round())

# predictions
print(sum(np.array(predictions) == np.array(y_hat[:,0])) , '/', len(y_hat))


# %%


