import logging
from pprint import pp

from src.ilp import Rule_Manger
from src.core import Atom, Term, Clause
import itertools

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
        for rule in rule_matrix:
            pp(len(rule))
            pp(rule[0:10])
        return rule_matrix

    def generate_clauses_with_depth(self, rule, T=3):
        '''Generate clauses with a maximum predicate chain depth of T'''
        rule_matrix = []
        if rule.allow_intensional:
            p = list(set(self.p_e + self.p_i + [self.target]))
        else:
            p = list(filter(lambda x: str(x) != 'Response(X_0,X_1)', list(set(self.p_e))))
        # intensional_predicates = set(atom.predicate for atom in self.p_i) if rule.allow_intensional else set()

        target_variables = ['X_%d' % i for i in range(self.target.arity)]

        head = Atom([Term(True, var) for var in target_variables], self.target.predicate)

        # Recursive function to generate body atoms up to depth T
        def generate_body_atoms(depth):
            if depth == 0:
                return [[]]  # Base case: return an empty body

            body_atoms = []
            for pred in p:
                pairs = itertools.product(range(rule.v+1), repeat=2)
                for i, j in pairs:
                    # Generate body atoms for the current predicate
                    if i != j:
                        body_atom = Atom(self.generate_terms([f"X_{i}", f"X_{j}"]), pred.predicate)
                        # # Check for circular references and valid predicates
                        # if not self.is_valid_clause(head, [body_atom]):
                        #     continue
                        
                        # Recur for the next depth and append this body atom
                        for body_clause in generate_body_atoms(depth - 1):
                            if body_atom not in body_clause:
                                body_atoms.append(body_clause + [body_atom])
            return body_atoms

        # Generate all body combinations up to depth T
        body_clauses = generate_body_atoms(T)

        # Add valid clauses to rule_matrix
        for body_clause in body_clauses:
            clause = Clause(head, body_clause)
            rule_matrix.append(clause)
        return rule_matrix

    def generate_terms(self, terms):
        '''Generate terms for the given arity using variable names based on v'''
        return [Term(True, term) for term in terms]