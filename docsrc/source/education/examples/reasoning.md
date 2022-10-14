# Reasoning 

This section provides several illustrative reasoning examples using the LNN. Each problem consists
of the natural language description of the background knowledge and the query. The logical 
representations of the background knowledge and query follows in First-order Logic. Finally,
code snippets showing the direct mapping to the LNN representation are provided. There are examples
typically found in the theorem proving literature. The goal is to prove that the background 
knowledge entails the query. 

The problems can be parsed into First-order Logic using any semantic parsing modules.
The output of semantic parsing is facts and rules. 
We extract <span style="color:orange">predicates</span> , `constants` (or `objects`)
and logical connectives (&rarr;, &forall;, etc.). 
Facts are made of <span style="color:orange">predicates</span> and `constants`. Here, we manually
parse each text into FOL to illustrative theorem proving.

## Simple Geometry reasoning example

The first example shows how to apply theorem proving, and LNN in particular, on a simple geometric
example. The background knowledge contains some general rules about shapes of objects, and
facts about some given objects. The goal is to prove geometrical properties of existing objects.

### Problem in Natural Language
**Background knowledge:** All <span style="color:orange">square</span> objects are <span style="color:orange">rectangular</span>. 
All <span style="color:orange">rectangular</span> objects have <span style="color:orange">four sides</span>. 
The object `c` is a <span style="color:orange">square</span>. The object `k` is a <span style="color:orange">square</span>.

**Query:** Is there any object with <span style="color:orange">four sides</span>?

We extract the following predicates: `Square`, `Rectangle` and `Fourside`; and two objects
(constants): `c` and `k`.

### Problem in First-order Logic
**Background knowledge**

``Square(c)`` 

``Square(k)``

&forall;x ``Square(x) `` &rarr; ``Rectangle(x)``

&forall;x ``Rectangle(x) `` &rarr; ``Fourside(x)``

**Query**

&exist;x ``Fourside(x)``?

### LNN Code

Since the LNN has a one-to-one mapping with First-order Logic, the problem can be 
directly translated to the LNN representation. The following code snippet shows how to
build an LNN model for the above problem and performance inference to answer queries.

```python
from lnn import (Predicate, Variable,
                 Exists, Implies, ForAll, Model, Fact, World)

model = Model()

# Variablle
x = Variable('x')

# Predicate declarations
square = Predicate('square')
rectangle = Predicate('rectangle')
foursides = Predicate('foursides')

# Axioms declarations
square_rect = ForAll(x, Implies(square(x), rectangle(x)))
rect_foursides = ForAll(x, Implies(rectangle(x), foursides(x)))

# Query
query = Exists(x, foursides(x))

# Add predicates and rules to the model
model.add_knowledge(square_rect, rect_foursides, query)

# Add facts to the model
model.add_data({square: {'c': Fact.TRUE, 'k': Fact.TRUE}})

# Perform inference
steps, facts_inferred = model.infer()

# Inspect the query node
print(model[query].true_groundings)
```

Expected output: `{'c', 'k'}`

## More complex reasoning example

### Problem in Natural Language
**Background knowledge:** The law says that it is a <span style="color:orange">crime</span> for an <span style="color:orange">American</span> to <span style="color:orange">sell</span> <span style="color:orange">weapons</span> to <span style="color:orange">hostile</span> nations. 
The <span style="color:orange">country</span> `Nono`, an <span style="color:orange">enemy</span> of `America`, and other countries `Gotham` and `Wakanda` all have some <span style="color:orange">missiles</span>. 
`Nono`’s <span style="color:orange">missiles</span> were <span style="color:orange">sold</span> to it by `Colonel West`, who is <span style="color:orange">American</span>.

**Query:** Prove that `Colonel West` is a <span style="color:orange">criminal</span>.

We manually extract the following predicates: `Owns`, `Missile`, `American`, `Enemy`, 
`Weapon`, `Sells`, and `Hostile`; and the following objects: `nono`, `m1`, `m2`, `m3`, `west`
and`america`. Below is the problem in FOL.

### Problem in First-order Logic

`Owns(nono,m1)`

`Missile(m1)`

`Missile(m2)`

`Missile(m3)`

`American(west)`

`Enemy(nono,america)`

`Enemy(wakanda,america)`

`Enemy(gotham,america)`

&forall; x,y,z `American(x)` &and; `Weapon(y)` &and; `Sells(x,y,z)` &and; `Hostile(z)` &rarr; `Criminal(x)`

&forall; x `Missile(x)` &and; `Owns(nono,x)` &rarr; `Sells(west,x,nono)`

&forall; x `Missile(x)` &rarr; `Weapon(x)`

&forall; x `Enemy(x,america)` &rarr; `Hostile(x)`

### LNN Code

The code snippet below shows the declaration of the model and all the predicates.

```python
from lnn import (Predicate, Variable, Join, And,
                 Exists, Implies, ForAll, Model, Fact, World)

model = Model()  # Instantiate a model.
x, y, z, w = map(Variable, ['x', 'y', 'z', 'w'])

# Define and add predicates to the model.
owns = model['owns'] = Predicate('owns', 2)  # binary predicate
missile = model['missile'] = Predicate('missile')
american = model['american'] = Predicate('american')
enemy = model['enemy'] = Predicate('enemy', 2)
hostile = model['hostile'] = Predicate('hostile')
criminal = model['criminal'] = Predicate('criminal')
weapon = model['weapon'] = Predicate('weapon')
sells = model['sells'] = Predicate('sells', 3)  # ternary predicate

```

The code snippet below shows an example of adding the background knowledge into the model. All
the other rules can be added to the model similarly.
```python
# Define and add the background knowledge to  the model.
america_enemies = (
    ForAll(x, Implies(enemy(x, (y, 'America')), 
                      hostile(x),
                      join=Join.OUTER),
           join=Join.OUTER, 
           world=World.AXIOM)
    )
model.add_knowledge(america_enemies)
```

The code snippet below shows how to add the query into the model.

```python
# Define queries
query = Exists(x, criminal(x))
model.add_knowledge(query)
```

Finally, we can add facts into the model.

```python
# Add facts to model.
model.set_facts({
    owns: {('Nono', 'M1'): Fact.TRUE},
    missile: {'M1': Fact.TRUE},
    american: {'West': Fact.TRUE},
    enemy: {('Nono', 'America'): Fact.TRUE},
})
```

Then we are ready to perform inference and answer queries (aka theorem proving):

```python
model.infer()
print(model[query].true_groundings)
```

Expected output: `{west}`.