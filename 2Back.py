import tensorflow as tf
tf.compat.v1.enable_eager_execution()

from src.core import Term, Atom
from src.ilp import Language_Frame, Program_Template, Rule_Template
from src.dilp import DILP

atoms = [
    Atom([Term(False, '(A, 1)'), Term(False, '(B, 2)')], 'succ'),
    Atom([Term(False, '(B, 2)'), Term(False, '(C, 3)')], 'succ'),
    Atom([Term(False, '(C, 3)'), Term(False, '(D, 4)')], 'succ'),
    Atom([Term(False, '(D, 4)'), Term(False, '(E, 5)')], 'succ'),
    Atom([Term(False, '(E, 5)'), Term(False, '(D, 6)')], 'succ'),
    Atom([Term(False, '(D, 6)'), Term(False, '(C, 7)')], 'succ'),
    Atom([Term(False, '(C, 7)'), Term(False, '(C, 8)')], 'succ'),
    Atom([Term(False, '(C, 8)'), Term(False, '(E, 9)')], 'succ'),
    Atom([Term(False, '(E, 9)'), Term(False, '(C, 10)')], 'succ'),
    Atom([Term(False, '(C, 10)'), Term(False, '(E, 11)')], 'succ'),

    Atom([Term(False, '(D, 4)'), Term(False, '(D, 6)')], 'sameLetter'),
    Atom([Term(False, '(C, 3)'), Term(False, '(C, 7)')], 'sameLetter'),
    Atom([Term(False, '(C, 3)'), Term(False, '(C, 8)')], 'sameLetter'),
    Atom([Term(False, '(C, 3)'), Term(False, '(C, 10)')], 'sameLetter'),
    Atom([Term(False, '(C, 7)'), Term(False, '(C, 8)')], 'sameLetter'),
    Atom([Term(False, '(E, 5)'), Term(False, '(E, 9)')], 'sameLetter'),
    Atom([Term(False, '(C, 7)'), Term(False, '(C, 10)')], 'sameLetter'),
    Atom([Term(False, '(C, 8)'), Term(False, '(C, 10)')], 'sameLetter'),
    Atom([Term(False, '(E, 5)'), Term(False, '(E, 11)')], 'sameLetter'),
    Atom([Term(False, '(E, 9)'), Term(False, '(E, 11)')], 'sameLetter')
]
predicates = {
    Atom([Term(True, 'X_0'), Term(True, 'X_1')], 'succ'),
    Atom([Term(True, 'X_0'), Term(True, 'X_1')], 'sameLetter'),
    Atom([Term(True, 'X_0'), Term(True, 'X_1')], '2Back')
}

constants = {'(A, 1)', '(B, 2)', '(C, 3)', '(D, 4)', '(E, 5)', '(D, 6)', '(C, 7)', 
 '(C, 8)', '(E, 9)', '(C, 10)', '(E, 11)'}

positive_examples = [
    Atom([Term(False, '(E, 5)'), Term(False, '(D, 6)')], '2Back'),
    Atom([Term(False, '(E, 9)'), Term(False, '(C, 10)')], '2Back'),
    Atom([Term(False, '(C, 10)'), Term(False, '(E, 11)')], '2Back')
]

negative_examples = [
    Atom([Term(False, '(A, 1)'), Term(False, '(B, 2)')], '2Back'),
    Atom([Term(False, '(B, 2)'), Term(False, '(C, 3)')], '2Back'),
    Atom([Term(False, '(C, 3)'), Term(False, '(D, 4)')], '2Back'),
    Atom([Term(False, '(D, 4)'), Term(False, '(E, 5)')], '2Back'),
    Atom([Term(False, '(D, 6)'), Term(False, '(C, 7)')], '2Back'),
    Atom([Term(False, '(C, 7)'), Term(False, '(C, 8)')], '2Back'),
    Atom([Term(False, '(C, 8)'), Term(False, '(E, 9)')], '2Back')
]

# target predicate template
term_x_0 = Term(True, 'X_0')
term_x_1 = Term(True, 'X_1')
p_e = list(predicates)
target = Atom([term_x_0, term_x_1], '2Back')

# instructions
p_a = [Atom([term_x_0, term_x_1], '2Back')]

p_a_rule = (Rule_Template(1, False), None)
target_rule = (Rule_Template(2, False), Rule_Template(2, False), Rule_Template(2, False), Rule_Template(2, False), Rule_Template(2, False))
rules = {p_a[0]: p_a_rule, target: target_rule}

# lang frame and program template
language_frame = Language_Frame(target, p_e, constants)
program_template = Program_Template(p_a, rules, 10)

# run dilp
dilp = DILP(language_frame, atoms, positive_examples, negative_examples, program_template)
dilp.train(steps=250)