# Reasoning 

## Simple Geometry reasoning example

### Problem in Natural Language
**Background knowledge:** All <span style="color:orange">square</span> objects are <span style="color:orange">rectangular</span>. 
All <span style="color:orange">rectangular</span> objects have <span style="color:orange">four sides</span>. 
The object `c` is a <span style="color:orange">square</span>. The object `k` is a <span style="color:orange">square</span>.

**Query:** Is there any object with <span style="color:orange">four sides</span>?

The problem can be parsed into First-order logic as shown below, using any semantic parsing modules.
The output of semantic parsing is facts and rules. We extract <span style="color:orange">predicates</span> , `constants` (or `objects`)
and logical connectives (&rarr;, &forall;, etc.). Facts are made of <span style="color:orange">predicates</span> and `constants`.
For example, `Square(c)` says object `c` is a square. Similarly, for object `k`. Finally,
'All square objects are rectangular' is parsed to &forall;x ``Square(x) `` &rarr; ``Rectangle(x)``.

### Problem in First-order Logic
**Background knowledge**

``Square(c)`` 

``Square(k)``

&forall;x ``Square(x) `` &rarr; ``Rectangle(x)``

&forall;x ``Rectangle(x) `` &rarr; ``Fourside(x)``

**Query**

&exist;x ``Fourside(x)``?

### LNN Code

Since the LNN has a one-to-one mapping with First-order Logic, the problem can be directly translated to the LNN representation.

```python
from lnn import (Predicate, Variable,
                 Exists, Implies, ForAll, Model, Fact, World)

model = Model()
     
# Variablle
x = Variable('x')
     
# Predicate declarations
square = Predicate(name='square')
rectangle = Predicate(name='rectangle')
foursides = Predicate(name='foursides')
     
# Axioms declarations
square_rect = ForAll(x, Implies(square(x), rectangle(x),
                                name='square-rect'),
                    name='all-square-rect', world=World.AXIOM)
rect_foursides = ForAll(x, Implies(rectangle(x), foursides(x),
                                   name='rect-foursides'),
                        name='all-rect-foursides', world=World.AXIOM)
                             
# Query
query = Exists(x, foursides(x), name='foursided_objects')
     
# Add predicates and rules to the model
model.add_formulae(square, rectangle, square_rect, rect_foursides, query)
     
# Add facts to the model
model.set_facts({'square': {'c': Fact.TRUE, 'k': Fact.TRUE}})
     
# Perform inference
steps, facts_inferred = model.infer()
     
# Inspect the query node
print(model['foursided_objects'].true_groundings)
```

Expected output: `{'c', 'k'}`

## More complex reasoning example

### Problem in Natural Language
**Background knowledge:** The law says that it is a <span style="color:orange">crime</span> for an <span style="color:orange">American</span> to <span style="color:orange">sell</span> <span style="color:orange">weapons</span> to <span style="color:orange">hostile</span> nations. 
The <span style="color:orange">country</span> `Nono`, an <span style="color:orange">enemy</span> of `America`, and other countries `Gotham` and `Wakanda` all have some <span style="color:orange">missiles</span>. 
`Nono`’s <span style="color:orange">missiles</span> were <span style="color:orange">sold</span> to it by `Colonel West`, who is <span style="color:orange">American</span>.

**Query:** Prove that `Colonel West` is a <span style="color:orange">criminal</span>.

### Problem in First-order Logic

`Owns(nono,m1)`

`Missile(m1)`

`Missile(m2)`

`Missile(m3)`

`American(\texttt{west})`

`Enemy(nono,america)`

`Enemy(wakanda,america)`

`Enemy(gotham,america)`

&forall; x,y,z `American(x)` &and; `Weapon(y)` &and; `Sells(x,y,z)` &and; `Hostile(z)` &rarr; `Criminal(x)`

&forall; x `Missile(x)` &and; `Owns(nono,x)` &rarr; `Sells(west,x,nono)`

&forall; x `Missile(x)` &rarr; `Weapon(x)`

&forall; x `Enemy(x,america)` &rarr; `Hostile(x)`

### LNN Code
```python
from lnn import (Predicate, Variable, Join, And,
                 Exists, Implies, ForAll, Model, Fact, World)

model = Model()  # Instantiate a model.
x, y, z, w = map(Variable, ['x', 'y', 'z', 'w'])

# Define and add predicates to the model.
owns = model['owns'] = Predicate(arity=2, name='owns')
missile = model['missile'] = Predicate(arity=1, name='missile')
american = model['american'] = Predicate(arity=1, name='american')
enemy = model['enemy'] = Predicate(arity=2, name='enemy')
hostile = model['hostile'] = Predicate(arity=1, name='hostile')
criminal = model['criminal'] = Predicate(arity=1, name='criminal')
weapon = model['weapon'] = Predicate(arity=1, name='weapon')
sells = model['sells'] = Predicate(arity=3, name='sells')


```