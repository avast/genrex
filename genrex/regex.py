import collections
import copy
import re
from typing import Generator, Iterable

from genrex.classes import (
    Alternation,
    Class,
    Concatenation,
    Expression,
    Literal,
    Meta,
    Range,
)
from genrex.clustering import Cluster
from genrex.enums import NamedObjectStrictType
from genrex.misc import filter_ngrams

TypeA = dict[int, dict[int, Expression]]
TypeB = dict[int, Expression]


def iterate_alternations(
    expr: Expression, iterate_cls: bool = True
) -> Generator[Expression, None, None]:
    """
    Iterate alternation class
    * if `iterate_cls` set to true also iterate class
    """
    if isinstance(expr, Alternation):
        for alt in expr.expr_alternations:
            yield from iterate_alternations(alt, iterate_cls=iterate_cls)
    elif isinstance(expr, Class):
        if iterate_cls:
            yield from expr.class_set
        else:
            yield expr
    else:
        yield expr


def iterate_concatenation(
    regex: Expression, ngram: Literal | None = None
) -> Generator[Expression, None, None]:
    """
    Iterate concatenation also split literals into len of 1
    * If ngram is specified. Detect that literal contains ngram and yield every char
      before ngram, ngram itself and every char after ngram
    * Range is expanded
    """

    # helper functions for code readability
    def iterate_range(rang: Range) -> Generator[Expression, None, None]:
        if rang.min > 0:
            yield from list(iterate_concatenation(rang.expr)) * rang.min
        if rang.max > rang.min:
            yield Range(rang.expr, 0, rang.max - rang.min)

    def iterate_literal(
        literal: Literal, ngram: Literal
    ) -> Generator[Literal, None, None]:
        index: int | None = literal.literal.find(ngram.literal)
        if index is not None and index >= 0:
            prefix = literal.literal[:index]
            suffix = literal.literal[index + len(ngram.literal) :]
            yield from (Literal(i) for i in prefix)
            yield ngram
            yield from (Literal(i) for i in suffix)
        else:
            yield from (Literal(i) for i in literal.literal)

    # Iterate over concatenation
    for expr in regex.split():
        if isinstance(expr, Range):
            yield from iterate_range(expr)
        elif isinstance(expr, Literal):
            if ngram is None:
                yield from (Literal(i) for i in expr.literal)
            else:
                yield from iterate_literal(expr, ngram)
        else:
            yield expr


def intersect(expr_x: Expression, expr_y: Expression) -> bool:
    """
    Return True if `expr_x` and `expr_y` have common alternative
    """
    if expr_x == Meta(".") and expr_y.min_max() == (1, 1):
        return True

    set1 = set(iterate_alternations(expr_x, iterate_cls=True))
    set2 = set(iterate_alternations(expr_y, iterate_cls=True))

    res = not set1.isdisjoint(set2)
    return res


def create_node(expr_a: TypeA) -> int:
    k = max(expr_a.keys()) + 1
    expr_a[k] = {}
    return k


def dive(nodes: Iterable[int], expr_a: TypeA) -> list[int]:
    result = set()
    for node in nodes:
        for sub in expr_a.get(node, {}):
            result.add(sub)
    return list(result)


def find_transition(
    atom: Expression,
    search: set[int],
    ignore: set[int],
    passed: set[int],
    expr_a: TypeA,
) -> list[tuple[int, Expression]]:
    result = []
    for from_node in search:
        if from_node in ignore:
            continue

        for to_node, expression in expr_a.get(from_node, {}).items():
            if to_node in ignore or to_node in passed:
                continue
            if intersect(atom, expression):
                result.append((to_node, expression))
    return result


def subnodes(node: int, expr_a: TypeA) -> Generator[int, None, None]:
    """
    Find all subnodes for specified node
    """
    expr_q = [node]
    expr_l = {node}

    while expr_q:
        node = expr_q.pop()
        yield node
        for i in expr_a[node].keys():
            if i in expr_l:
                continue
            expr_q.append(i)
            expr_l.add(i)


def to_range(expr: Expression) -> Expression:
    """
    If expression is alternation with empty literal change expression into Range
    * Also change Alternation into class if all alternations are literals which match
      one character
    """
    if isinstance(expr, Alternation) and Literal("") in expr.expr_alternations:
        alts = {*expr.expr_alternations}
        alts.remove(Literal(""))
        if all(x.min_max() == (1, 1) and isinstance(x, Literal) for x in alts):
            return Range(Class(alts), 0, 1)
        else:
            return Range(Alternation(alts), 0, 1)
    return expr


def convert_to_range(expr_a: Expression, expr_b: Expression) -> Expression | None:
    """
    Try to merge expressions into range instead of concatenation
    """

    expr_a = to_range(expr_a)
    expr_b = to_range(expr_b)
    if (
        isinstance(expr_a, Range)
        and isinstance(expr_b, Range)
        and expr_a.expr == expr_b.expr
    ):
        return Range(expr_a.expr, expr_a.min + expr_b.min, expr_a.max + expr_b.max)
    elif isinstance(expr_a, Range) and expr_a.expr == expr_b:
        return Range(expr_a.expr, expr_a.min + 1, expr_a.max + 1)
    elif isinstance(expr_b, Range) and expr_b.expr == expr_a:
        return Range(expr_b.expr, expr_b.min + 1, expr_b.max + 1)
    elif expr_a == expr_b and not isinstance(expr_a, Literal):
        return Range(expr_a, 2, 2)
    elif isinstance(expr_a, Concatenation) and isinstance(expr_b, Concatenation):
        ranged = convert_to_range(expr_a.expr_b, expr_b.expr_a)
        if ranged is not None:
            return Concatenation(expr_a.expr_a, Concatenation(ranged, expr_b.expr_b))
    elif isinstance(expr_a, Concatenation):
        ranged = convert_to_range(expr_a.expr_b, expr_b)
        if ranged is not None:
            return Concatenation(expr_a.expr_a, ranged)
    elif isinstance(expr_b, Concatenation):
        ranged = convert_to_range(expr_a, expr_b.expr_a)
        if ranged is not None:
            return Concatenation(ranged, expr_b.expr_b)

    return None


class State:
    def __init__(self):
        self.accept = False
        self.transitions = {}


class Trie:
    def __init__(self):
        self.root: State = State()
        self.nodes: list[State] = []
        self.ngram_start: int | None = None
        self.ngram: Literal | None = None

    def add(self, word: str, start: int = -1, ngram_trie_root=None):
        node = self.root
        i = 0
        connect: bool = ngram_trie_root is not None
        while i < len(word):
            char = word[i]
            found = False
            for child, child_transition in node.transitions.items():
                if child == char:
                    if (
                        connect
                        and char in ngram_trie_root.transitions
                        and child_transition is ngram_trie_root.transitions[char]
                    ):
                        connect = False
                    node = child_transition
                    found = True
                    break
            if not found:
                if i == start and connect:
                    node.transitions[char] = ngram_trie_root.transitions[char]
                    node = node.transitions[char]
                else:
                    new_node = State()
                    self.nodes.append(new_node)
                    node.transitions[char] = new_node
                    node = new_node
            i += 1
        node.accept = True

    def add_ngram(self, word: str):
        node = self.root
        for char in word:
            new_node = State()
            self.nodes.append(new_node)
            node.transitions[char] = new_node
            node = new_node

    def print_trie(self, node: State, space: str = ""):
        space += " "
        for child in node.transitions:
            print(
                space,
                child,
                "-",
                node.transitions[child].accept,
                "-",
                node.transitions[child].transitions,
            )
            self.print_trie(node.transitions[child], space)

    def to_regex(self) -> Expression:
        matrix_a: dict = {}
        matrix_b: dict = {}
        matrix_c: dict = {}

        for i, node in enumerate(self.nodes):
            if node.accept:
                matrix_b[i] = Literal("")

            matrix_a[i] = {}
            for child, state in node.transitions.items():
                index = self.nodes.index(state)
                matrix_a[i][index] = Literal(child)
                matrix_c[index] = i

        matrix_a, matrix_b = self.minimize(matrix_a, matrix_b, matrix_c)
        return self.solve(matrix_a, matrix_b)

    def minimize(self, matrix_a: dict, matrix_b: dict, matrix_c: dict):
        matrix_d = []
        b_list = sorted(list(matrix_b.keys()), reverse=True)
        for i, element in enumerate(b_list):
            state = element
            if state in matrix_b:
                char = matrix_a[matrix_c[state]][state]
                for j in range(i + 1, len(b_list)):
                    second_state = b_list[j]
                    if (
                        second_state in matrix_b
                        and second_state != matrix_c[state]
                        and second_state in matrix_a[matrix_c[second_state]]
                        and matrix_a[matrix_c[second_state]][second_state] == char
                    ):
                        if matrix_c[second_state] != state:
                            del matrix_a[matrix_c[second_state]][second_state]
                            matrix_a[matrix_c[second_state]][state] = char
                            del matrix_b[second_state]
                            if matrix_c[state] != 0:
                                matrix_d.append(matrix_c[state])

        if len(matrix_d) > 1:
            matrix_a = self.minimize_rest(matrix_a, matrix_c, matrix_d)
        return matrix_a, matrix_b

    def minimize_rest(self, matrix_a: dict, matrix_c: dict, matrix_d: list):
        states = []
        i = 0
        while i < len(matrix_d):
            state = matrix_d[i]
            if state in matrix_d and state in matrix_a[matrix_c[state]]:
                char = matrix_a[matrix_c[state]][state]
                j = i + 1
                while j < len(matrix_d):
                    second_state = matrix_d[j]
                    if (
                        second_state != matrix_c[state]
                        and second_state in matrix_a[matrix_c[second_state]]
                        and matrix_a[matrix_c[second_state]][second_state] == char
                    ):
                        if matrix_c[second_state] != state:
                            del matrix_a[matrix_c[second_state]][second_state]
                            matrix_a[matrix_c[second_state]][state] = char
                            del matrix_d[j]
                            if matrix_c[state] != 0:
                                states.append(matrix_c[state])
                    j = j + 1
            i = i + 1

        if len(states) > 1:
            matrix_a = self.minimize_rest(matrix_a, matrix_c, states)
        return matrix_a

    def solve(self, matrix_a: dict, matrix_b: dict) -> Expression:
        for k in range(len(matrix_a) - 1, -1, -1):
            for i in range(k):
                # matrix_b[i] = matrix_b[i] + (matrix_a[i][k] . matrix_b[k])
                if k in matrix_a[i]:
                    if k in matrix_b:
                        # concat = matrix_a[i][k] + matrix_b[k]
                        concat = self.concat(matrix_a[i][k], matrix_b[k])
                        if i in matrix_b:
                            matrix_b[i] = self.union(matrix_b[i], concat, True)
                        else:
                            matrix_b[i] = concat
                    for j in range(k):
                        # matrix_a[i][j] = matrix_a[i][j] + (matrix_a[i][k] . matrix_a[k][j])
                        if j in matrix_a[k]:
                            concat = self.concat(matrix_a[i][k], matrix_a[k][j])
                            if j in matrix_a[i]:
                                if j in matrix_b:
                                    matrix_a[i][j] = self.union(
                                        matrix_a[i][j], concat, True
                                    )
                                else:
                                    matrix_a[i][j] = self.union(
                                        matrix_a[i][j], concat, False
                                    )
                            else:
                                matrix_a[i][j] = concat

        # result is stored in matrix_b[0]
        return matrix_b[0]

    def concat(self, expr_a: Expression, expr_b: Expression) -> Expression:
        if len(expr_a) == 0:
            return expr_b
        elif len(expr_b) == 0:
            return expr_a

        ranged = convert_to_range(expr_a, expr_b)
        if ranged is not None:
            return ranged

        if isinstance(expr_a, Literal) and isinstance(expr_b, Literal):
            return Literal(expr_a.literal + expr_b.literal)
        elif (
            isinstance(expr_a, Literal)
            and isinstance(expr_b, Concatenation)
            and isinstance(expr_b.expr_a, Literal)
        ):
            return Concatenation(
                Literal(expr_a.literal + expr_b.expr_a.literal), expr_b.expr_b
            )
        elif (
            isinstance(expr_b, Literal)
            and isinstance(expr_a, Concatenation)
            and isinstance(expr_a.expr_b, Literal)
        ):
            return Concatenation(
                expr_a.expr_a, Literal(expr_a.expr_b.literal + expr_b.literal)
            )
        else:
            return Concatenation(expr_a, expr_b)

    def union(self, expr_a: Expression, expr_b: Expression, final: bool) -> Expression:
        expr_a = copy.deepcopy(expr_a)
        expr_b = copy.deepcopy(expr_b)
        res: Expression
        if expr_a != expr_b:
            end = self.common_end(expr_a, expr_b, final)
            if end != "":
                expr_a = expr_a.remove(end)
                expr_b = expr_b.remove(end)

            union_set: set[Expression] = set()
            union_set.update(expr_a)
            union_set.update(expr_b)

            expr_aa = len(expr_a) == 1 or isinstance(expr_a, Class)
            expr_bb = len(expr_b) == 1 or isinstance(expr_b, Class)
            # Cases below should not be created as class.
            # `.` meta should not be in Class
            # Range expr should not be in Class
            expr_cc = (
                expr_a != Meta(".")
                and expr_b != Meta(".")
                and not isinstance(expr_a, Range)
                and not isinstance(expr_b, Range)
            )
            if Meta(".") in union_set and all(x.min_max() == (1, 1) for x in union_set):
                res = Meta(".")
            elif expr_aa and expr_bb and expr_cc:
                res = Class(union_set)
            else:
                res = Alternation(union_set)

            if end != "":
                res = Concatenation(res, Literal(end))
            return res
        else:
            return expr_a

    def common_end(self, expr_a: Expression, expr_b: Expression, final: bool) -> str:
        values = []
        if isinstance(expr_a, Alternation):
            values.extend(expr_a.get_list())
        else:
            values.append(expr_a)
        if isinstance(expr_b, Alternation):
            values.extend(expr_b.get_list())
        else:
            values.append(expr_b)
        if len(values) != 0:
            end = repr(values[0])
            while end:
                if all(value.endswith(end) for value in values):
                    if len(end) == 1 and end.isalnum() and final:
                        return ""
                    else:
                        return end
                end = end[1:]
        return ""

    def union_with_transition(
        self, expr: Expression, from_node: int, to_node: int, expr_a: TypeA
    ) -> Expression:
        if to_node in expr_a[from_node]:
            return self.union(expr, expr_a[from_node][to_node], final=False)
        else:
            return expr

    def add_atom(
        self,
        nodes: set[int],
        atom: Expression,
        search: set[int],
        ignore: set[int],
        passed: set[int],
        expr_a: TypeA,
    ) -> tuple[set[int], set[int], set[int]]:
        """
        Add one atomic regular expression into `expr_a`
        """

        # atom length is zero: Literal('')
        if atom.min_max() == (0, 0):
            return nodes, search, passed

        # Nodes in which transition ends from specified nodes
        to_nodes = set()
        # Nodes in which transitions should be searched
        to_search = set()

        # Find all transitions which are from search nodes and use expression which
        # have common alternative with atom
        all_transitions = set(find_transition(atom, search, ignore, passed, expr_a))
        new: int | None = None
        for node in nodes:
            # Do not create transition when some transition already exists
            transitions = set(all_transitions)
            if not transitions.isdisjoint(expr_a[node].items()):
                transitions.intersection_update(expr_a[node].items())

            # Iterate over transitions
            for to_node, _ in transitions:
                to_search.add(to_node)
                expr_a[node][to_node] = self.union_with_transition(
                    atom, node, to_node, expr_a
                )
                to_nodes.add(to_node)

            # If no transitions were detected create new node
            if not transitions:
                # Create max one node for atom
                if new is None:
                    new = create_node(expr_a)
                expr_a[node][new] = atom
                to_nodes.add(new)

        passed.update(to_nodes)
        return to_nodes, to_search, passed

    def add_range(
        self,
        expr: Range,
        nodes: set[int],
        search: set[int],
        ignore: set[int],
        passed: set[int],
        expr_a: TypeA,
        expr_b: TypeB,
    ) -> tuple[set[int], set[int], set[int]]:
        # This function should be only called on range with min == 0
        assert expr.min == 0

        to_nodes = set(nodes)
        to_search = set(search)

        for _ in range(expr.min, expr.max):
            nodes, search, unit_passed = self.add_unit(
                expr.expr, nodes, search, ignore, passed, expr_a, expr_b
            )
            to_nodes.update(nodes)
            to_search.update(search)
            passed.update(unit_passed)

        return to_nodes, to_search, passed

    def add_unit(
        self,
        unit: Expression,
        nodes: set[int],
        search: set[int],
        ignore: set[int],
        passed: set[int],
        expr_a: TypeA,
        expr_b: TypeB,
    ) -> tuple[set[int], set[int], set[int]]:
        """
        Add one unit into expr_a, expr_b
        """
        passed = set(passed)

        # Unit nodes
        unit_nodes: set[int] = set()
        unit_searches: set[int] = set()
        unit_passed: set[int] = set()

        # Iterate over alternations of unit
        for alt in iterate_alternations(
            unit,
            iterate_cls=False,
        ):
            # Every alternation start from same nodes, search, passed
            alt_nodes: set[int] = set(nodes)
            alt_search: set[int] = set(search)
            alt_passed: set[int] = set(passed)

            # Iterate over concatenations in unit
            for atom in iterate_concatenation(alt):
                # if atom is alternation recursively call `add_unit`
                if isinstance(atom, Alternation):
                    alt_nodes, alt_search, alt_passed = self.add_unit(
                        atom, alt_nodes, alt_search, ignore, alt_passed, expr_a, expr_b
                    )
                # if atom is range call `add_range`
                elif isinstance(atom, Range) and atom.min == 0:
                    alt_nodes, alt_search, alt_passed = self.add_range(
                        atom, alt_nodes, alt_search, ignore, alt_passed, expr_a, expr_b
                    )
                # Otherwise call `add_atom`
                else:
                    alt_nodes, alt_search, alt_passed = self.add_atom(
                        alt_nodes, atom, alt_search, ignore, alt_passed, expr_a
                    )

            # Save final nodes of alternation
            unit_nodes.update(alt_nodes)
            unit_searches.update(alt_search)
            unit_passed.update(alt_passed)

        return unit_nodes, unit_searches, unit_passed

    def add_regex(
        self,
        regex: Expression,
        expr_a: TypeA,
        expr_b: TypeB,
        n_gram: Literal | None,
        n_gram_start: int | None,
    ):
        # Do not pass through these nodes
        ignore: set[int] = set()
        if n_gram_start:
            ignore.add(n_gram_start)
            ignore.update(subnodes(n_gram_start, expr_a))

        # Nodes from which transition in made
        nodes: set[int] = {0}
        # Nodes in which is transition searched
        search: set[int] = {0}
        # Nodes which were used
        passed: set[int] = {0}

        # Iterate over concatenation
        for unit in iterate_concatenation(regex, ngram=n_gram):
            # Align regex with ngram from cluster
            if (
                n_gram_start is not None
                and n_gram is not None
                and len(n_gram) > 0
                and unit == n_gram
            ):
                expr = Literal(unit.literal[0])  # type:ignore
                for node in nodes:
                    expr_a[node][n_gram_start] = self.union_with_transition(
                        expr, node, n_gram_start, expr_a
                    )

                nodes = {n_gram_start}
                search = {n_gram_start}
                ignore = set()

                unit = Literal(unit.literal[1:])  # type:ignore

            # Add unit into a,b
            nodes, search, passed = self.add_unit(
                unit, nodes, search, ignore, passed, expr_a, expr_b
            )

        # All nodes are terminated
        for node in nodes:
            expr_b[node] = Literal("")

        self.minimize_equations(expr_a, expr_b)

    def minimize_equations(self, expr_a: TypeA, expr_b: TypeB):
        """
        Merge transitions which use same expression from one node
        """
        queue = [0]
        while queue:
            node = queue.pop(0)
            # Find transitions which use same expression
            tran_expr = collections.defaultdict(set)
            for i, expr in expr_a[node].items():
                tran_expr[expr].add(i)

            # Merge transitions which use same expression
            for expr, nodes in tran_expr.items():
                if len(nodes) > 1:
                    new = create_node(expr_a)
                    expr_a[node][new] = expr
                    for old in nodes:
                        expr_a[node].pop(old)
                        # Merge expr_b
                        if old in expr_b:
                            if new in expr_b:
                                expr_b[new] = self.union(
                                    expr_b[new], expr_b[old], False
                                )
                            else:
                                expr_b[new] = expr_b[old]
                        # Merge expr_a
                        for sub, expr in expr_a[old].items():
                            if sub in expr_a[new]:
                                expr_a[new][sub] = self.union(
                                    expr_a[new][sub], expr_a[old][sub], False
                                )
                            else:
                                expr_a[new][sub] = expr_a[old][sub]

            # Minimize sub nodes
            for i in expr_a[node]:
                queue.append(i)

        # Remove unreachable transitions
        nodes = {0, *subnodes(0, expr_a)}
        for i in list(expr_a.keys()):
            if i not in nodes:
                expr_a[i] = {}


patterns: dict = {
    "GENREXGUIDBRACKET": (
        r"\{[A-Za-z0-9]{8}-[A-Za-z0-9]{4}-[A-Za-z0-9]{4}-[A-Za-z0-9]{4}-[A-Za-z0-9]{12}\}"
    ),
    "GENREXGUID": r"[A-Za-z0-9]{8}-[A-Za-z0-9]{4}-[A-Za-z0-9]{4}-[A-Za-z0-9]{4}-[A-Za-z0-9]{12}",
    "GENREXHEX8": r"[0-9A-Fa-f]{8}:[0-9A-Fa-f]:[0-9A-Fa-f]{8}:[0-9A-Fa-f]{8}:[0-9A-Fa-f]{8}",
    "GENREXURL": r"[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}",
    "GENREXTHEX": r"[0-9a-f]{3}\.[0-9a-f]{3}\.[0-9a-f]{3}",
    "GENREXDOT": r"\.[A-Za-z]{2,4}$",
    "GENREXPHONE": r"\+?[0-9]-[0-9]{3}-[0-9]{3}-[0-9]{4}",
}


class Regex:
    def __init__(self):
        self.regexps = ""
        self.found_patterns = set()
        self.dot_array = set()
        self.prefix = False
        self.suffix = False

    def get_regexps(self):
        return self.regexps

    def make_regex(self, cluster: Cluster):
        if NamedObjectStrictType.contains(str(cluster.input_type)):
            self.prefix = True
            self.suffix = True

        if len(cluster.similars) == 0:
            return
        ngram = cluster.ngram
        trie = Trie()
        trie.nodes.append(trie.root)
        if ngram != "":
            trie.ngram = Literal(ngram)
            trie.ngram_start = len(trie.nodes)

            ngram_trie = Trie()
            ngram_trie.add_ngram(ngram)
            trie.nodes.extend(ngram_trie.nodes)

        for samples in cluster.similars.values():
            for sample in samples:
                sample = self.detect_patterns(sample)
                if ngram != "":
                    start = sample.find(ngram)
                    trie.add(sample, start, ngram_trie.root)
                else:
                    trie.add(sample)

        self.regexps = trie.to_regex()

    def detect_patterns(self, input_str: str) -> str:
        for key, pattern in patterns.items():
            found = re.search(pattern, input_str)
            if found:
                self.found_patterns.add(key)
                if "GENREXDOT" in self.found_patterns:
                    self.dot_array.add(found.group(0)[1:])
                input_str = re.sub(pattern, key, input_str)
                # break
        return input_str

    def remove_patterns(self, input_str: str) -> str:
        for pattern in self.found_patterns:
            if pattern == "GENREXDOT":
                input_str = re.sub(
                    "GENREXDOT",
                    r"\." + str(Alternation(self.dot_array)),
                    input_str,
                )
            else:
                input_str = re.sub(
                    pattern,
                    patterns[pattern],
                    input_str,
                )
        return input_str

    def get_results(self, reduce: bool = False) -> str:
        if reduce:
            res = self.regexps.heuristic().reduce()
        else:
            res = self.regexps.heuristic()

        result = str(res)

        result = self.remove_patterns(result)

        # add (^|\\) prefix
        if self.prefix:
            result = "(^|\\\\)" + result

        if self.suffix:
            result = result + "$"

        if reduce:
            result = max(result.split(r"\.\*"), key=len)

            for remove_string in filter_ngrams["urldot"]:
                result = result.replace(remove_string, ".*")

        return result
