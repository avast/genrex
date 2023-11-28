"""
Tests regex algorithm without heuristics
"""
import re
import unittest

import pytest

from genrex.clustering import Cluster
from genrex.enums import InputType
from genrex.regex import Regex


def create_cluster(data, ngrams, input_type=InputType.MUTEX):
    clusters = []
    for i, sample in enumerate(data):
        cluster_regex = Cluster(ngrams[i])
        cluster_regex.input_type = input_type
        for k in sample.keys():
            for string in sample[k]:
                cluster_regex.similars[k].append(string)
        clusters.append(cluster_regex)
    return clusters


def escape(string):
    res = re.sub(
        r"\\x[a-fA-F0-9]{2}",
        lambda x: x.group(0)
        if x.group(0) == r"\x64"
        and x.end() + 1 < len(string)
        and string[x.end()] == "\\"
        and string[x.end() + 1] != "x"
        else x.group(0).encode("latin1").decode("unicode_escape"),
        string,
    )
    return res


class RegexTests(unittest.TestCase):
    def test_empty(self):
        data = [{"sample1": []}]
        ngrams = [""]
        clusters = create_cluster(data, ngrams)
        res_list = [""]
        for i, cluster in enumerate(clusters):
            regex = Regex()
            regex.make_regex(cluster)
            res = regex.get_regexps()
            self.assertEqual(res_list[i], str(res))

    def test_set1(self):
        data = [{"sample1": ["a", "b"]}]
        ngrams = [""]
        clusters = create_cluster(data, ngrams)
        res_list = ["[ab]"]
        for i, cluster in enumerate(clusters):
            regex = Regex()
            regex.make_regex(cluster)
            res = regex.get_regexps()
            self.assertEqual(res_list[i], str(res))

    def test_set2(self):
        data = [{"sample1": ["ab", "bc"]}]
        ngrams = [""]
        clusters = create_cluster(data, ngrams)
        res_list = ["(ab|bc)"]
        for i, cluster in enumerate(clusters):
            regex = Regex()
            regex.make_regex(cluster)
            res = regex.get_regexps()
            self.assertEqual(res_list[i], str(res))

    def test_set3(self):
        data = [{"sample1": ["abc", "abd"]}]
        ngrams = [""]
        clusters = create_cluster(data, ngrams)
        res_list = ["ab[cd]"]
        for i, cluster in enumerate(clusters):
            regex = Regex()
            regex.make_regex(cluster)
            res = regex.get_regexps()
            self.assertEqual(res_list[i], str(res))

    def test_set4(self):
        data = [{"sample1": ["abc", "abd"]}]
        ngrams = ["ab"]
        clusters = create_cluster(data, ngrams)
        res_list = ["ab[cd]"]
        for i, cluster in enumerate(clusters):
            regex = Regex()
            regex.make_regex(cluster)
            res = regex.get_regexps()
            self.assertEqual(res_list[i], str(res))

    def test_set5(self):
        data = [{"sample1": ["abc", "abd", "abe"]}]
        ngrams = ["ab"]
        clusters = create_cluster(data, ngrams)
        res_list = ["ab[cde]"]
        for i, cluster in enumerate(clusters):
            regex = Regex()
            regex.make_regex(cluster)
            res = regex.get_regexps()
            self.assertEqual(res_list[i], str(res))

    def test_set6(self):
        data = [{"sample1": ["a.c", "a.d", "a.a"]}]
        ngrams = ["a."]
        clusters = create_cluster(data, ngrams)
        res_list = [r"a\.[acd]"]
        for i, cluster in enumerate(clusters):
            regex = Regex()
            regex.make_regex(cluster)
            res = regex.get_regexps()
            self.assertEqual(res_list[i], str(res))

    def test_set7(self):
        data = [{"sample1": ["hello1", "hello2", "hello3", "hello4"]}]
        ngrams = ["hell"]
        clusters = create_cluster(data, ngrams)
        res_list = ["hello[1234]"]
        for i, cluster in enumerate(clusters):
            regex = Regex()
            regex.make_regex(cluster)
            res = regex.get_regexps()
            self.assertEqual(res_list[i], str(res))

    def test_set8(self):
        data = [{"sample1": ["hello123", "hello456", "helloabc"]}]
        ngrams = ["hello"]
        clusters = create_cluster(data, ngrams)
        res_list = ["hello(123|456|abc)"]
        for i, cluster in enumerate(clusters):
            regex = Regex()
            regex.make_regex(cluster)
            res = regex.get_regexps()
            self.assertEqual(res_list[i], str(res))

    def test_set9(self):
        data = [{"sample1": ["hello1x3", "hello4x6", "helloaxc"]}]
        ngrams = ["hello"]
        clusters = create_cluster(data, ngrams)
        res_list = ["hello(1x3|4x6|axc)"]
        for i, cluster in enumerate(clusters):
            regex = Regex()
            regex.make_regex(cluster)
            res = regex.get_regexps()
            self.assertEqual(res_list[i], str(res))

    def test_set10(self):
        data = [{"sample1": ["hello123", "hello453", "helloabc"]}]
        ngrams = ["hell"]
        clusters = create_cluster(data, ngrams)
        res_list = ["hello(123|453|abc)"]
        for i, cluster in enumerate(clusters):
            regex = Regex()
            regex.make_regex(cluster)
            res = regex.get_regexps()
            self.assertEqual(res_list[i], str(res))

    def test_set11(self):
        data = [{"sample1": ["hello123", "hello457", "helloab7"]}]
        ngrams = ["hell"]
        clusters = create_cluster(data, ngrams)
        res_list = ["hello(123|457|ab7)"]
        for i, cluster in enumerate(clusters):
            regex = Regex()
            regex.make_regex(cluster)
            res = regex.get_regexps()
            self.assertEqual(res_list[i], str(res))

    def test_set12(self):
        data = [{"sample1": ["hello12x", "hello45x", "helloabx"]}]
        ngrams = ["hell"]
        clusters = create_cluster(data, ngrams)
        res_list = ["hello(12x|45x|abx)"]
        for i, cluster in enumerate(clusters):
            regex = Regex()
            regex.make_regex(cluster)
            res = regex.get_regexps()
            self.assertEqual(res_list[i], str(res))

    def test_set13(self):
        data = [{"sample1": ["xxnecoc", "yyynecoc"]}]
        ngrams = ["neco"]
        clusters = create_cluster(data, ngrams)
        res_list = ["(xx|yyy)necoc"]
        for i, cluster in enumerate(clusters):
            regex = Regex()
            regex.make_regex(cluster)
            res = regex.get_regexps()
            self.assertEqual(res_list[i], str(res))

    def test_set14(self):
        data = [{"sample1": ["1239999", "4569999"]}]
        ngrams = ["9999"]
        clusters = create_cluster(data, ngrams)
        res_list = ["(123|456)9999"]
        for i, cluster in enumerate(clusters):
            regex = Regex()
            regex.make_regex(cluster)
            res = regex.get_regexps()
            self.assertEqual(res_list[i], str(res))

    def test_set15(self):
        data = [{"sample1": ["abcdef", "abcdeabcde", "abcdexyz"]}]
        ngrams = ["abcde"]
        clusters = create_cluster(data, ngrams)
        res_list = ["abcde(abcde|f|xyz)"]
        for i, cluster in enumerate(clusters):
            regex = Regex()
            regex.make_regex(cluster)
            res = regex.get_regexps()
            self.assertEqual(res_list[i], str(res))

    def test_set16(self):
        data = [{"sample1": ["foobar", "foobaz"]}]
        ngrams = ["fooba"]
        clusters = create_cluster(data, ngrams)
        res_list = ["fooba[rz]"]
        for i, cluster in enumerate(clusters):
            regex = Regex()
            regex.make_regex(cluster)
            res = regex.get_regexps()
            self.assertEqual(res_list[i], str(res))

    def test_set17(self):
        data = [{"sample1": ["a", "b", "c"]}]
        ngrams = [""]
        clusters = create_cluster(data, ngrams)
        res_list = ["[abc]"]
        for i, cluster in enumerate(clusters):
            regex = Regex()
            regex.make_regex(cluster)
            res = regex.get_regexps()
            self.assertEqual(res_list[i], str(res))

    def test_set18(self):
        data = [{"sample1": ["abcd", "abc$"]}]
        ngrams = ["abc"]
        clusters = create_cluster(data, ngrams)
        res_list = ["abc[$d]"]
        for i, cluster in enumerate(clusters):
            regex = Regex()
            regex.make_regex(cluster)
            res = regex.get_regexps()
            self.assertEqual(res_list[i], str(res))

    def test_set19(self):
        data = [{"sample1": ["abcd", "abc$", "abc^"]}]
        ngrams = ["abc"]
        clusters = create_cluster(data, ngrams)
        res_list = [r"abc[$\^d]"]
        for i, cluster in enumerate(clusters):
            regex = Regex()
            regex.make_regex(cluster)
            res = regex.get_regexps()
            self.assertEqual(res_list[i], str(res))

    def test_set20(self):
        data = [{"sample1": ["abcd", "abca", "abc^"]}]
        ngrams = ["abc"]
        clusters = create_cluster(data, ngrams)
        res_list = [r"abc[\^ad]"]
        for i, cluster in enumerate(clusters):
            regex = Regex()
            regex.make_regex(cluster)
            res = regex.get_regexps()
            self.assertEqual(res_list[i], str(res))

    def test_set21(self):
        data = [{"sample1": ["abca", "abc-", "abcd"]}]
        ngrams = ["abc"]
        clusters = create_cluster(data, ngrams)
        res_list = [r"abc[\-ad]"]
        for i, cluster in enumerate(clusters):
            regex = Regex()
            regex.make_regex(cluster)
            res = regex.get_regexps()
            self.assertEqual(res_list[i], str(res))

    def test_set22(self):
        data = [{"sample1": ["foobarfoo", "foozapfoo"]}]
        ngrams = ["foo"]
        clusters = create_cluster(data, ngrams)
        res_list = ["foo(bar|zap)foo"]
        for i, cluster in enumerate(clusters):
            regex = Regex()
            regex.make_regex(cluster)
            res = regex.get_regexps()
            self.assertEqual(res_list[i], str(res))

    def test_set23(self):
        data = [{"sample1": ["foobarfoo", "foozapfoo", "fooaarfoo"]}]
        ngrams = ["foo"]
        clusters = create_cluster(data, ngrams)
        res_list = ["foo(aar|bar|zap)foo"]
        for i, cluster in enumerate(clusters):
            regex = Regex()
            regex.make_regex(cluster)
            res = regex.get_regexps()
            self.assertEqual(res_list[i], str(res))

    def test_set24(self):
        data = [{"sample1": ["foobar)foo", "foozapfoo", "fooaarfoo", "foobaf)foo"]}]
        ngrams = ["foo"]
        clusters = create_cluster(data, ngrams)
        res_list = [r"foo(aar|ba[fr]\)|zap)foo"]
        for i, cluster in enumerate(clusters):
            regex = Regex()
            regex.make_regex(cluster)
            res = regex.get_regexps()
            self.assertEqual(res_list[i], str(res))

    def test_set25(self):
        data = [{"sample1": ["123helloabc", "456hellobcd"]}]
        ngrams = ["hello"]
        clusters = create_cluster(data, ngrams)
        res_list = ["(123|456)hello(abc|bcd)"]
        for i, cluster in enumerate(clusters):
            regex = Regex()
            regex.make_regex(cluster)
            res = regex.get_regexps()
            self.assertEqual(res_list[i], str(res))

    def test_set26(self):
        data = [{"sample1": ["asome", "^some", "-some"]}]
        ngrams = ["some"]
        clusters = create_cluster(data, ngrams)
        res_list = [r"[\-\^a]some"]
        for i, cluster in enumerate(clusters):
            regex = Regex()
            regex.make_regex(cluster)
            res = regex.get_regexps()
            self.assertEqual(res_list[i], str(res))

    def test_set37(self):
        data = [{"sample1": ["asome", "^some", "/some"]}]
        ngrams = ["some"]
        clusters = create_cluster(data, ngrams)
        res_list = [r"[\/\^a]some"]
        for i, cluster in enumerate(clusters):
            regex = Regex()
            regex.make_regex(cluster)
            res = regex.get_regexps()
            self.assertEqual(res_list[i], str(res))

    def test_hexdecimal1(self):
        data = [
            {
                "sample1": [
                    escape(r"asome\x00"),
                    escape(r"bsome\x00"),
                    escape(r"csome\x00"),
                ]
            }
        ]
        ngrams = ["some"]
        clusters = create_cluster(data, ngrams)
        res_list = [r"[abc]some\x00"]
        for i, cluster in enumerate(clusters):
            regex = Regex()
            regex.make_regex(cluster)
            res = regex.get_regexps()
            self.assertEqual(res_list[i], str(res))

    def test_hexdecimal2(self):
        data = [
            {"sample1": [escape("asome"), escape(r"\xffsome"), escape(r"\x00some")]}
        ]
        ngrams = ["some"]
        clusters = create_cluster(data, ngrams)
        res_list = [r"[\x00a\xFF]some"]
        for i, cluster in enumerate(clusters):
            regex = Regex()
            regex.make_regex(cluster)
            res = regex.get_regexps()
            self.assertEqual(res_list[i], str(res))

    def test_hexdecimal3(self):
        data = [
            {
                "sample1": [
                    escape(r"\ksome"),
                    escape(r"\\xffsome"),
                    escape(r"\\x00some"),
                ]
            }
        ]
        ngrams = ["some"]
        clusters = create_cluster(data, ngrams)
        res_list = [r"\\[\x00k\xFF]some"]
        for i, cluster in enumerate(clusters):
            regex = Regex()
            regex.make_regex(cluster)
            res = regex.get_regexps()
            self.assertEqual(res_list[i], str(res))

    def test_hexdecimal4(self):
        data = [
            {
                "sample1": [
                    escape("\\\tsome"),
                    escape("\\\xffsome"),
                    escape("\\\x00some"),
                ]
            }
        ]
        ngrams = ["some"]
        clusters = create_cluster(data, ngrams)
        res_list = [r"\\[\x00\t\xFF]some"]
        for i, cluster in enumerate(clusters):
            regex = Regex()
            regex.make_regex(cluster)
            res = regex.get_regexps()
            self.assertEqual(res_list[i], str(res))

    def test_hexdecimal5(self):
        data = [
            {
                "sample1": [
                    escape("\t\x11some"),
                    escape("\t\xffsome"),
                    escape("\t\x00some"),
                ]
            }
        ]
        ngrams = ["some"]
        clusters = create_cluster(data, ngrams)
        res_list = [r"\t[\x00\x11\xFF]some"]
        for i, cluster in enumerate(clusters):
            regex = Regex()
            regex.make_regex(cluster)
            res = regex.get_regexps()
            self.assertEqual(res_list[i], str(res))


if __name__ == "__main__":
    pytest.main()
