import tensorflow as tf
tf.compat.v1.enable_eager_execution()

from src.core import Term, Atom
from src.ilp import Language_Frame, Program_Template, Rule_Template
from src.dilp import DILP

# Define the atoms based on the sequence of presented items
atoms = [
    Atom([Term(False, 'A'), Term(False, '1')], 'Presented'),
    Atom([Term(False, 'B'), Term(False, '2')], 'Presented'),
    Atom([Term(False, 'C'), Term(False, '3')], 'Presented'),
    Atom([Term(False, 'D'), Term(False, '4')], 'Presented'),
    Atom([Term(False, 'E'), Term(False, '5')], 'Presented'),
    Atom([Term(False, 'D'), Term(False, '6')], 'Presented'),
    Atom([Term(False, 'C'), Term(False, '7')], 'Presented'),
    Atom([Term(False, 'C'), Term(False, '8')], 'Presented'),
    Atom([Term(False, 'E'), Term(False, '9')], 'Presented'),
    Atom([Term(False, 'C'), Term(False, '10')], 'Presented'),
    Atom([Term(False, 'E'), Term(False, '11')], 'Presented'),
    Atom([Term(False, 'C'), Term(False, 'C')], 'Same'),
    Atom([Term(False, 'D'), Term(False, 'D')], 'Same'),
    Atom([Term(False, 'E'), Term(False, 'E')], 'Same'),
    Atom([Term(False, '1'), Term(False, '2')], 'successor'),
    Atom([Term(False, '2'), Term(False, '3')], 'successor'),
    Atom([Term(False, '3'), Term(False, '4')], 'successor'),
    Atom([Term(False, '4'), Term(False, '5')], 'successor'),
    Atom([Term(False, '5'), Term(False, '6')], 'successor'),
    Atom([Term(False, '6'), Term(False, '7')], 'successor'),
    Atom([Term(False, '7'), Term(False, '8')], 'successor'),
    Atom([Term(False, '8'), Term(False, '9')], 'successor'),
    Atom([Term(False, '9'), Term(False, '10')], 'successor'),
    Atom([Term(False, '10'), Term(False, '11')], 'successor')
]

# Define the predicates
predicates = {
    Atom([Term(True, 'X_0'), Term(True, 'X_1')], 'Presented'),
    Atom([Term(True, 'X_0'), Term(True, 'X_1')], 'Same'),
    Atom([Term(True, 'X_0'), Term(True, 'X_1')], 'Response'),
    Atom([Term(True, 'X_0'), Term(True, 'X_1')], 'successor')
}

# Define constants used in the learning task
constants = {'A', 'B', 'C', 'D', 'E', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11'}

# Define positive examples for the `Response` predicate
positive_examples = [
    Atom([Term(False, 'C'), Term(False, '10')], 'Response'),  # C at t=10 is the same as C at t=8
    Atom([Term(False, 'D'), Term(False, '6')], 'Response'),   # D at t=6 is the same as D at t=4
    Atom([Term(False, 'E'), Term(False, '11')], 'Response')    # E at t=11 is the same as E at t=9
]

# Define negative examples for the `Response` predicate
negative_examples = [
    Atom([Term(False, 'A'), Term(False, '3')], 'Response'),   # A at t=1 is not the same as C at t=3
    Atom([Term(False, 'B'), Term(False, '4')], 'Response'),   # B at t=2 is not the same as D at t=4
    Atom([Term(False, 'C'), Term(False, '5')], 'Response')    # C at t=3 is not the same as E at t=5
]

# Define the target predicate template
term_x_0 = Term(True, 'X_0')
term_x_1 = Term(True, 'X_1')
p_e = list(predicates)
target = Atom([term_x_0, term_x_1], 'Response')

# Instructions for the learning process
# p_a_rule = (Rule_Template(1, False), None)
target_rule = (
    Rule_Template(1, False),
    Rule_Template(3, False)
)
rules = {target: target_rule}

# Language frame and program template setup
language_frame = Language_Frame(target, p_e, constants)
program_template = Program_Template([], rules, 6)

# Run DILP to learn the 2Back predicate
dilp = DILP(language_frame, atoms, positive_examples, negative_examples, program_template)
dilp.train(steps=12)

# (Presented(X,T)∧Presented(Y,T1)∧Same(X,Y)∧successor(T1,T)∧successor(T2,T1))