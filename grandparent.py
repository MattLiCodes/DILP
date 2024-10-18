import tensorflow as tf
tf.compat.v1.enable_eager_execution()

from src.core import Term, Atom
from src.ilp import Language_Frame, Program_Template, Rule_Template
from src.dilp import DILP

atoms = [
    Atom([Term(False, 'alice'), Term(False, 'bob')], 'parent'),
    Atom([Term(False, 'bob'), Term(False, 'carl')], 'parent'),
    Atom([Term(False, 'david'), Term(False, 'emily')], 'parent'),
    Atom([Term(False, 'emily'), Term(False, 'frank')], 'parent'),
    Atom([Term(False, 'gary'), Term(False, 'helen')], 'parent'),
    Atom([Term(False, 'helen'), Term(False, 'ian')], 'parent'),
]
predicates = {
    Atom([Term(True, 'X_0'), Term(True, 'X_1')], 'parent'),
    Atom([Term(True, 'X_0'), Term(True, 'X_1')], 'grandparent')
}
constants = {'alice', 'bob', 'carl', 'david', 'emily', 'frank', 'gary', 'helen', 'ian'}

positive_examples = [
    Atom([Term(False, 'alice'), Term(False, 'carl')], 'grandparent'),
    Atom([Term(False, 'david'), Term(False, 'frank')], 'grandparent'),
    Atom([Term(False, 'gary'), Term(False, 'ian')], 'grandparent')
]

negative_examples = [
    Atom([Term(False, 'alice'), Term(False, 'emily')], 'grandparent'),
    Atom([Term(False, 'bob'), Term(False, 'frank')], 'grandparent'),
    Atom([Term(False, 'gary'), Term(False, 'frank')], 'grandparent')
]

# target predicate template
term_x_0 = Term(True, 'X_0')
term_x_1 = Term(True, 'X_1')
p_e = list(predicates)
p_a = [Atom([term_x_0, term_x_1], 'grandparent')]
target = Atom([term_x_0, term_x_1], 'grandparent')

# instructions
p_a_rule = (Rule_Template(1, False), None)
target_rule = (Rule_Template(0, False), Rule_Template(1, True))
rules = {p_a[0]: p_a_rule, target: target_rule}

# lang frame and program template
language_frame = Language_Frame(target, p_e, constants)
program_template = Program_Template(p_a, rules, 10)

# run dilp
dilp = DILP(language_frame, atoms, positive_examples, negative_examples, program_template)
dilp.train(steps=250)
