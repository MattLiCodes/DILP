'''Optimized combinatorial class
'''
import logging

from src.ilp import Rule_Manger
from src.core import Atom, Term, Clause

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

class Optimized_Combinatorial_Generator(Rule_Manger):

    def generate_clauses(self):
        '''Generate all clauses with some level of optimization'''
        rule_matrix = []
        for rule in self.rules:
            if rule is None:
                rule_matrix.append([None])
                continue
            clauses = self.generate_clauses_with_depth(rule)
            rule_matrix.append(clauses)
        print("done generating clauses")
        return rule_matrix

    def generate_clauses_with_depth(self, rule, T=3):
        '''Generate clauses with a maximum predicate chain depth of T'''
        rule_matrix = []
        p = list(set(self.p_e + self.p_i + [self.target]))
        intensional_predicates = set(atom.predicate for atom in self.p_i) if rule.allow_intensional else set()

        target_variables = ['X_%d' % i for i in range(self.target.arity)]

        head = Atom([Term(True, var) for var in target_variables], self.target.predicate)

        # Recursive function to generate body atoms up to depth T
        def generate_body_atoms(depth):
            if depth == 0:
                return [[]]  # Base case: return an empty body

            body_atoms = []
            for pred in p:
                # Generate body atoms for the current predicate
                body_atom = Atom(self.generate_terms(pred.arity), pred.predicate)

                # # Check for circular references and valid predicates
                # if not self.is_valid_clause(head, [body_atom]):
                #     continue
                
                # Recur for the next depth and append this body atom
                for body_clause in generate_body_atoms(depth - 1):
                    body_atoms.append(body_clause + [body_atom])

            return body_atoms

        # Generate all body combinations up to depth T
        body_clauses = generate_body_atoms(T)

        # Add valid clauses to rule_matrix
        for body_clause in body_clauses:
            clause = Clause(head, body_clause)
            rule_matrix.append(clause)
        print(len(rule_matrix))
        return rule_matrix

    def generate_terms(self, arity):
        '''Generate terms for the given arity using variable names'''
        return [Term(True, 'X_%d' % i) for i in range(arity)]

    def is_valid_clause(self, head, body):
        '''Check if the clause is valid based on specific conditions'''
        target_variables = {v.name for v in head.terms}

        # All variables in head should be in the body
        if not target_variables.issubset({v.name for atom in body for v in atom.terms}):
            return False
        # No circular references
        if head in body:
            return False

        # Check for intensional predicates if required
        if any(atom.predicate in self.p_i for atom in body):
            return True  # If it's an intensional predicate and allowed

        return True