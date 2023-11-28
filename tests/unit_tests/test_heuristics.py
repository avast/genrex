"""
Tests regex algorithm with heuristics
"""
import unittest

from genrex.classes import Concatenation, Meta, Range
from genrex.clustering import Cluster
from genrex.enums import InputType
from genrex.regex import Alternation, Class, Literal, Regex


def create_cluster(data, ngrams, input_type=InputType.MUTEX):
    clusters = []
    for i, sample in enumerate(data):
        cluster_heuristics = Cluster(ngrams[i])
        cluster_heuristics.input_type = input_type
        for k in sample.keys():
            for string in sample[k]:
                cluster_heuristics.similars[k].append(string)
        clusters.append(cluster_heuristics)
    return clusters


class HeuristicTests(unittest.TestCase):
    # Classes heuristics

    def test_set1(self):
        data = [{"sample1": ["hello1", "hello2", "hello3", "hello4"]}]
        ngrams = ["hello"]
        clusters = create_cluster(data, ngrams)
        res_list = ["(^|\\\\)hello[0-9]$"]
        for i, cluster in enumerate(clusters):
            regex = Regex()
            regex.make_regex(cluster)
            res = regex.get_results()
            self.assertEqual(res_list[i], res)

    def test_set2(self):
        data = [{"sample1": ["helloa", "hellob", "helloc", "hellod"]}]
        ngrams = ["hello"]
        clusters = create_cluster(data, ngrams)
        res_list = ["(^|\\\\)hello[0-9a-f]$"]
        for i, cluster in enumerate(clusters):
            regex = Regex()
            regex.make_regex(cluster)
            res = regex.get_results()
            self.assertEqual(res_list[i], res)

    def test_set3(self):
        data = [{"sample1": ["helloa", "hellob", "helloc", "hellod", "helloe"]}]
        ngrams = ["hello"]
        clusters = create_cluster(data, ngrams)
        res_list = ["(^|\\\\)hello[0-9a-f]$"]
        for i, cluster in enumerate(clusters):
            regex = Regex()
            regex.make_regex(cluster)
            res = regex.get_results()
            self.assertEqual(res_list[i], res)

    def test_set4(self):
        data = [{"sample1": ["a", "b", "c", "d", "p", "r", "x", "y", "z"]}]
        ngrams = [""]
        clusters = create_cluster(data, ngrams)
        res_list = ["(^|\\\\)[a-z]$"]
        for i, cluster in enumerate(clusters):
            regex = Regex()
            regex.make_regex(cluster)
            res = regex.get_results()
            self.assertEqual(res_list[i], res)

    def test_set5(self):
        data = [{"sample1": ["`", "a", "c", "e", "f", "g", "h", "i"]}]
        ngrams = [""]
        clusters = create_cluster(data, ngrams)
        res_list = ["(^|\\\\)[`ace-i]$"]
        for i, cluster in enumerate(clusters):
            regex = Regex()
            regex.make_regex(cluster)
            res = regex.get_results()
            self.assertEqual(res_list[i], res)

    def test_set6(self):
        data = [{"sample1": ["a", "i"]}]
        ngrams = [""]
        clusters = create_cluster(data, ngrams)
        res_list = ["(^|\\\\)[ai]$"]
        for i, cluster in enumerate(clusters):
            regex = Regex()
            regex.make_regex(cluster)
            res = regex.get_results()
            self.assertEqual(res_list[i], res)

    def test_set7(self):
        data = [{"sample1": ["a", "b"]}]
        ngrams = [""]
        clusters = create_cluster(data, ngrams)
        res_list = ["(^|\\\\)[ab]$"]
        for i, cluster in enumerate(clusters):
            regex = Regex()
            regex.make_regex(cluster)
            res = regex.get_results()
            self.assertEqual(res_list[i], res)

    def test_set8(self):
        data = [{"sample1": ["a", "b", "c", "x"]}]
        ngrams = [""]
        clusters = create_cluster(data, ngrams)
        res_list = ["(^|\\\\)[a-z]$"]
        for i, cluster in enumerate(clusters):
            regex = Regex()
            regex.make_regex(cluster)
            res = regex.get_results()
            self.assertEqual(res_list[i], res)

    def test_set9(self):
        data = [{"sample1": ["a", "\t"]}]
        ngrams = [""]
        clusters = create_cluster(data, ngrams)
        res_list = [r"(^|\\)[\ta]$"]
        for i, cluster in enumerate(clusters):
            regex = Regex()
            regex.make_regex(cluster)
            res = regex.get_results()
            self.assertEqual(res_list[i], res)

    def test_set10(self):
        data = [{"sample1": ["a", "	"]}]
        ngrams = [""]
        clusters = create_cluster(data, ngrams)
        res_list = [r"(^|\\)[\ta]$"]
        for i, cluster in enumerate(clusters):
            regex = Regex()
            regex.make_regex(cluster)
            res = regex.get_results()
            self.assertEqual(res_list[i], res)

    def test_set11(self):
        data = [
            {
                "sample1": ["hello1", "hello2", "hello3", "hello4"],
                "sample2": ["helloa", "hellob", "helloc", "hellod"],
            }
        ]
        ngrams = ["hello"]
        clusters = create_cluster(data, ngrams)
        res_list = ["(^|\\\\)hello[0-9a-f]$"]
        for i, cluster in enumerate(clusters):
            regex = Regex()
            regex.make_regex(cluster)
            res = regex.get_results()
            self.assertEqual(res_list[i], res)

    def test_set12(self):
        data = [
            {
                "sample1": ["hello1", "hello2", "hello3"],
                "sample2": ["hello4", "hello5", "hello6"],
            }
        ]
        ngrams = ["hello"]
        clusters = create_cluster(data, ngrams)
        res_list = ["(^|\\\\)hello[0-9]$"]
        for i, cluster in enumerate(clusters):
            regex = Regex()
            regex.make_regex(cluster)
            res = regex.get_results()
            self.assertEqual(res_list[i], res)

    # Alternation heuristics
    def test_set13(self):
        data = [{"sample1": ["helloabcd", "helloefg"]}]
        ngrams = ["hello"]
        clusters = create_cluster(data, ngrams)
        res_list = ["(^|\\\\)hello(abcd|efg)$"]
        for i, cluster in enumerate(clusters):
            regex = Regex()
            regex.make_regex(cluster)
            res = regex.get_results()
            self.assertEqual(res_list[i], res)

    def test_set14(self):
        data = [{"sample1": ["helloabc", "helloefg"]}]
        ngrams = ["hello"]
        clusters = create_cluster(data, ngrams)
        res_list = ["(^|\\\\)hello(abc|efg)$"]
        for i, cluster in enumerate(clusters):
            regex = Regex()
            regex.make_regex(cluster)
            res = regex.get_results()
            self.assertEqual(res_list[i], res)

    def test_set15(self):
        data = [{"sample1": ["ab", "bc"]}]
        ngrams = [""]
        clusters = create_cluster(data, ngrams)
        res_list = ["(^|\\\\)[0-9a-f]{2}$"]
        for i, cluster in enumerate(clusters):
            regex = Regex()
            regex.make_regex(cluster)
            res = regex.get_results()
            self.assertEqual(res_list[i], res)

    def test_set16(self):
        data = [{"sample1": ["hello123", "hello456", "helloabc"]}]
        ngrams = ["hello"]
        clusters = create_cluster(data, ngrams)
        res_list = ["(^|\\\\)hello[0-9a-f]{3}$"]
        for i, cluster in enumerate(clusters):
            regex = Regex()
            regex.make_regex(cluster)
            res = regex.get_results()
            self.assertEqual(res_list[i], res)

    def test_set17(self):
        data = [{"sample1": ["hello1x3", "hello4x6", "helloaxc"]}]
        ngrams = ["hello"]
        clusters = create_cluster(data, ngrams)
        res_list = ["(^|\\\\)hello(1x3|4x6|axc)$"]
        for i, cluster in enumerate(clusters):
            regex = Regex()
            regex.make_regex(cluster)
            res = regex.get_results()
            self.assertEqual(res_list[i], res)

    def test_set18(self):
        data = [{"sample1": ["hello123", "hello453", "helloabc"]}]
        ngrams = ["hell"]
        clusters = create_cluster(data, ngrams)
        res_list = ["(^|\\\\)hello[0-9a-f]{3}$"]
        for i, cluster in enumerate(clusters):
            regex = Regex()
            regex.make_regex(cluster)
            res = regex.get_results()
            self.assertEqual(res_list[i], res)

    def test_set19(self):
        data = [{"sample1": ["hello123", "hello457", "helloAB7"]}]
        ngrams = ["hell"]
        clusters = create_cluster(data, ngrams)
        res_list = ["(^|\\\\)hello[0-9A-F]{3}$"]
        for i, cluster in enumerate(clusters):
            regex = Regex()
            regex.make_regex(cluster)
            res = regex.get_results()
            self.assertEqual(res_list[i], res)

    def test_set20(self):
        data = [{"sample1": ["hello12x", "hello35x", "helloabx"]}]
        ngrams = ["hell"]
        clusters = create_cluster(data, ngrams)
        res_list = ["(^|\\\\)hello(12x|35x|abx)$"]
        for i, cluster in enumerate(clusters):
            regex = Regex()
            regex.make_regex(cluster)
            res = regex.get_results()
            self.assertEqual(res_list[i], res)

    def test_set21(self):
        data = [{"sample1": ["xxnecoc", "yyynecoc"]}]
        ngrams = ["neco"]
        clusters = create_cluster(data, ngrams)
        res_list = ["(^|\\\\)(xx|yyy)necoc$"]
        for i, cluster in enumerate(clusters):
            regex = Regex()
            regex.make_regex(cluster)
            res = regex.get_results()
            self.assertEqual(res_list[i], res)

    def test_set22(self):
        data = [{"sample1": ["1239999", "4569999"]}]
        ngrams = ["9999"]
        clusters = create_cluster(data, ngrams)
        res_list = ["(^|\\\\)[0-9]{3}9999$"]
        for i, cluster in enumerate(clusters):
            regex = Regex()
            regex.make_regex(cluster)
            res = regex.get_results()
            self.assertEqual(res_list[i], res)

    def test_set23(self):
        data = [{"sample1": ["abcdef", "abcdeabcde", "abcdexyz"]}]
        ngrams = ["abcde"]
        clusters = create_cluster(data, ngrams)
        res_list = ["(^|\\\\)abcde(abcde|f|xyz)$"]
        for i, cluster in enumerate(clusters):
            regex = Regex()
            regex.make_regex(cluster)
            res = regex.get_results()
            self.assertEqual(res_list[i], res)

    def test_set24(self):
        data = [{"sample1": ["foobarfoo", "foozapfoo"]}]
        ngrams = ["foo"]
        clusters = create_cluster(data, ngrams)
        res_list = ["(^|\\\\)foo(bar|zap)foo$"]
        for i, cluster in enumerate(clusters):
            regex = Regex()
            regex.make_regex(cluster)
            res = regex.get_results()
            self.assertEqual(res_list[i], res)

    def test_set25(self):
        data = [{"sample1": ["foobarfoo", "foozapfoo", "fooaarfoo"]}]
        ngrams = ["foo"]
        clusters = create_cluster(data, ngrams)
        res_list = ["(^|\\\\)foo(aar|bar|zap)foo$"]
        for i, cluster in enumerate(clusters):
            regex = Regex()
            regex.make_regex(cluster)
            res = regex.get_results()
            self.assertEqual(res_list[i], res)

    def test_set26(self):
        data = [{"sample1": ["foobar)foo", "foozapfoo", "fooaarfoo", "foobaf)foo"]}]
        ngrams = ["foo"]
        clusters = create_cluster(data, ngrams)
        res_list = [r"(^|\\)foo(aar|ba[fr]\)|zap)foo$"]
        for i, cluster in enumerate(clusters):
            regex = Regex()
            regex.make_regex(cluster)
            res = regex.get_results()
            self.assertEqual(res_list[i], res)

    def test_set27(self):
        data = [{"sample1": ["123helloabc", "456hellobcd"]}]
        ngrams = ["hello"]
        clusters = create_cluster(data, ngrams)
        res_list = ["(^|\\\\)[0-9]{3}hello[0-9a-f]{3}$"]
        for i, cluster in enumerate(clusters):
            regex = Regex()
            regex.make_regex(cluster)
            res = regex.get_results()
            self.assertEqual(res_list[i], res)

    def test_set28(self):
        data = [{"sample1": ["asome", "^some", "-some"]}]
        ngrams = ["some"]
        clusters = create_cluster(data, ngrams)
        res_list = [r"(^|\\)[\-\^a]some$"]
        for i, cluster in enumerate(clusters):
            regex = Regex()
            regex.make_regex(cluster)
            res = regex.get_results()
            self.assertEqual(res_list[i], res)

    def test_set29(self):
        data = [{"sample1": ["aso^me", "^so^me", "-so^me"]}]
        ngrams = ["so^me"]
        clusters = create_cluster(data, ngrams)
        res_list = [r"(^|\\)[\-\^a]so\^me$"]
        for i, cluster in enumerate(clusters):
            regex = Regex()
            regex.make_regex(cluster)
            res = regex.get_results()
            self.assertEqual(res_list[i], res)

    def test_set30(self):
        data = [
            {
                "sample1": ["asome", "bsome", "csome"],
                "sample2": ["csome", "dsome", "esome"],
            }
        ]
        ngrams = ["some"]
        clusters = create_cluster(data, ngrams)
        res_list = ["(^|\\\\)[0-9a-f]some$"]
        for i, cluster in enumerate(clusters):
            regex = Regex()
            regex.make_regex(cluster)
            res = regex.get_results()
            self.assertEqual(res_list[i], res)

    def test_set31(self):
        data = [
            {
                "sample1": ["asomea", "bsomea", "csomeb"],
                "sample2": ["csome1", "dsome2", "esome3"],
            }
        ]
        ngrams = ["some"]
        clusters = create_cluster(data, ngrams)
        res_list = ["(^|\\\\)[0-9a-f]some[0-9a-f]$"]
        for i, cluster in enumerate(clusters):
            regex = Regex()
            regex.make_regex(cluster)
            res = regex.get_results()
            self.assertEqual(res_list[i], res)

    def test_set32(self):
        data = [{"sample1": ["x\a", "x\t", "x\n", "x\f", "x\r"]}]
        ngrams = [""]
        clusters = create_cluster(data, ngrams)
        res_list = [r"(^|\\)x[\a\t\n\f\r]$"]
        for i, cluster in enumerate(clusters):
            regex = Regex()
            regex.make_regex(cluster)
            res = regex.get_results()
            self.assertEqual(res_list[i], res)

    def test_set33(self):
        data = [{"sample1": ["InstructionArea", "Instruction"]}]
        ngrams = [""]
        clusters = create_cluster(data, ngrams)
        res_list = [r"(^|\\)Instruction(Area)?$"]
        for i, cluster in enumerate(clusters):
            regex = Regex()
            regex.make_regex(cluster)
            res = regex.get_results()
            self.assertEqual(res_list[i], res)

    def test_set34(self):
        data = [{"sample1": ["Firefox", "Firefox1", "Firefox", "Firefox5", "Firefox8"]}]
        ngrams = [""]
        clusters = create_cluster(data, ngrams)
        res_list = [r"(^|\\)Firefox[0-9]?$"]
        for i, cluster in enumerate(clusters):
            regex = Regex()
            regex.make_regex(cluster)
            res = regex.get_results()
            self.assertEqual(res_list[i], res)

    def test_set35(self):
        data = [
            {"sample1": ["Firefoxbb", "Firefoxab", "Firefox", "Firefoxcc", "Firefoxcc"]}
        ]
        ngrams = [""]
        clusters = create_cluster(data, ngrams)
        res_list = [r"(^|\\)Firefox([0-9a-f]{2})?$"]
        for i, cluster in enumerate(clusters):
            regex = Regex()
            regex.make_regex(cluster)
            res = regex.get_results()
            self.assertEqual(res_list[i], res)

    def test_set36(self):
        data = [{"sample1": ["MainInstructionArea", "MainInstruction", "instructions"]}]
        ngrams = ["nstruct"]
        clusters = create_cluster(data, ngrams)
        res_list = [r"(^|\\)(Main)?[Ii]nstruction(Area|s)?$"]
        for i, cluster in enumerate(clusters):
            regex = Regex()
            regex.make_regex(cluster)
            res = regex.get_results()
            self.assertEqual(res_list[i], res)

    def test_set37(self):
        data = [{"sample1": ["EXPIRED", "EXPIREVER"]}]
        ngrams = ["EXPI"]
        clusters = create_cluster(data, ngrams)
        res_list = [r"(^|\\)EXPIRE(D|VER)$"]
        for i, cluster in enumerate(clusters):
            regex = Regex()
            regex.make_regex(cluster)
            res = regex.get_results()
            self.assertEqual(res_list[i], res)

    def test_class1(self):
        res = "[0-9a-f]"
        expr = Alternation(
            [Literal("a"), Literal("b"), Literal("c"), Literal("d"), Literal("a")]
        )
        expr = expr.heuristic()
        self.assertEqual(res, str(expr))

    def test_class2(self):
        res = r"[\a\t\n\f\r]"
        expr = Class(
            [Literal("\t"), Literal("\n"), Literal("\r"), Literal("\f"), Literal("\a")]
        )
        expr = expr.heuristic()
        self.assertEqual(res, str(expr))

    def test_class3(self):
        data = [
            {
                "sample1": [
                    "I'm Unicor\xef\xbe\x86",
                    "I'm Unicorn",
                ]
            }
        ]
        ngrams = [r"I'm Unic"]
        clusters = create_cluster(data, ngrams)
        res_list = [r"(^|\\)I'm Unicor(\xEF\xBE)?[n\x86]$"]
        for i, cluster in enumerate(clusters):
            regex = Regex()
            regex.make_regex(cluster)
            res = regex.get_results()
            self.assertEqual(res_list[i], res)

    def test_intervals1(self):
        data = [
            {
                "sample1": [
                    "firefox",
                    "firefox22",
                    "firefox1",
                ]
            }
        ]
        ngrams = [""]
        clusters = create_cluster(data, ngrams)
        res_list = [r"(^|\\)firefox([0-9]{1,2})?$"]
        for i, cluster in enumerate(clusters):
            regex = Regex()
            regex.make_regex(cluster)
            res = regex.get_results()
            self.assertEqual(res_list[i], res)

    def test_regex_heuristic1(self):
        regex = Concatenation(Literal("asd"), Meta("\\b")).heuristic()
        self.assertEqual("asd\\b", str(regex))

    def test_regex_heuristic2(self):
        regex = Alternation([Literal("asd"), Meta("\\b")]).heuristic()
        self.assertEqual("(\\b|asd)", str(regex))

    def test_regex_heuristic3(self):
        regex = Range(Alternation([Literal("asd"), Meta("\\b")]), 0, 5).heuristic()
        self.assertEqual("(\\b|asd){,5}", str(regex))

    def test_regex_heuristic4(self):
        regex = Concatenation(
            Alternation([Literal(x) for x in "012"]), Meta("\\b")
        ).heuristic()
        self.assertEqual("[0-9]\\b", str(regex))

    def test_regex_heuristic5(self):
        regex = Alternation([*[Literal(x) for x in "012"], Meta("\\b")]).heuristic()
        self.assertEqual("(0|1|2|\\b)", str(regex))

    def test_regex_heuristic6(self):
        regex = Range(Alternation([Literal(x) for x in "012"]), 3, 5).heuristic()
        self.assertEqual("[0-9]{3,5}", str(regex))

    def test_regex_heuristic7(self):
        regex = Concatenation(
            Meta("\\b"),
            Alternation([Literal(x) for x in "012"]),
        ).heuristic()
        self.assertEqual("\\b[0-9]", str(regex))

    def test_regex_heuristic8(self):
        regex = Alternation(
            [
                Concatenation(Literal("0123"), Class([Literal(x) for x in "789"])),
                Class([Literal(x) for x in "789abc"]),
            ]
        ).heuristic()
        self.assertEqual("[0-9a-f]{1,5}", str(regex))

    def test_regex_heuristic9(self):
        regex = Alternation(
            [
                Concatenation(Literal("abece"), Meta("\\b")),
                Class([Literal(x) for x in "789abc"]),
            ]
        ).heuristic()
        self.assertEqual("([0-9a-f]|abece\\b)", str(regex))

    def test_regex_heuristic10(self):
        regex = Alternation(
            [
                Meta("\\b"),
                Literal("123"),
            ]
        ).heuristic()
        self.assertEqual("(123|\\b)", str(regex))
