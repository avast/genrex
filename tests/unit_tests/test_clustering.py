import unittest
from collections import defaultdict

import pytest

from genrex.clustering import Corpus
from genrex.enums import InputType


def clustering(data, cuckoo_format, input_type=InputType.MUTEX):
    cluster = Corpus()
    cluster.add_resource(
        data,
        cuckoo_format=cuckoo_format,
        input_type=input_type,
    )
    clusters = cluster.cluster()
    return clusters


class ClusteringTests(unittest.TestCase):
    def test_empty(self):
        data = {}
        res = []
        cuckoo_format = False
        clusters = clustering(
            data,
            cuckoo_format,
        )
        self.assertEqual(res, clusters)

    def test_set1(self):
        data = {"sample1": ["MutexA", "MutexB"], "sample2": ["MutexY", "MutexZ"]}
        cluster_l = 1
        res_l = [2]
        res = [
            defaultdict(
                list, {"sample1": ["MutexA", "MutexB"], "sample2": ["MutexY", "MutexZ"]}
            )
        ]
        res_ngram = ["Mute"]
        res_stats = [{"unique": 4, "max": 2, "min": 2, "average": 2.0}]
        cuckoo_format = False
        clusters = clustering(data, cuckoo_format)
        self.assertEqual(cluster_l, len(clusters))
        for i, cluster in enumerate(clusters):
            self.assertEqual(res_ngram[i], cluster.ngram)
            self.assertEqual(res_l[i], len(cluster.similars))
            self.assertEqual(res[i], cluster.similars)
            self.assertEqual(res_stats[i]["unique"], cluster.unique)
            self.assertEqual(res_stats[i]["max"], cluster.max)
            self.assertEqual(res_stats[i]["min"], cluster.min)
            self.assertEqual(res_stats[i]["average"], cluster.average)

    def test_set2(self):
        data = {
            "sample1": ["hello123", "hello456"],
            "sample2": ["helloabc", "helloefg"],
        }
        cluster_l = 1
        res_l = [2]
        res = [
            defaultdict(
                list,
                {
                    "sample1": ["hello123", "hello456"],
                    "sample2": ["helloabc", "helloefg"],
                },
            )
        ]
        res_ngram = ["hell"]
        res_stats = [{"unique": 4, "max": 2, "min": 2, "average": 2.0}]
        cuckoo_format = False
        clusters = clustering(data, cuckoo_format)
        self.assertEqual(cluster_l, len(clusters))
        for i, cluster in enumerate(clusters):
            self.assertEqual(res_ngram[i], cluster.ngram)
            self.assertEqual(res_l[i], len(cluster.similars))
            self.assertEqual(res[i], cluster.similars)
            self.assertEqual(res_stats[i]["unique"], cluster.unique)
            self.assertEqual(res_stats[i]["max"], cluster.max)
            self.assertEqual(res_stats[i]["min"], cluster.min)
            self.assertEqual(res_stats[i]["average"], cluster.average)

    def test_set3(self):
        data = {
            "sample1": [
                "abcdef",
                "abcdeabcde",
                "abcdexyz",
                "1239999",
                "4569999",
                "hello123",
                "hello456",
                "helloabc",
                "xxnecoc",
                "yynecoc",
            ]
        }
        cluster_l = 4
        res_l = [1, 1, 1, 1]
        res = [
            defaultdict(list, {"sample1": ["1239999", "4569999"]}),
            defaultdict(list, {"sample1": ["abcdef", "abcdexyz"]}),
            defaultdict(list, {"sample1": ["hello123", "hello456", "helloabc"]}),
            defaultdict(list, {"sample1": ["xxnecoc", "yynecoc"]}),
        ]
        res_ngram = ["9999", "abcd", "hell", "neco"]
        res_stats = [
            {"unique": 2, "max": 2, "min": 2, "average": 2.0},
            {"unique": 2, "max": 2, "min": 2, "average": 2.0},
            {"unique": 3, "max": 3, "min": 3, "average": 3.0},
            {"unique": 2, "max": 2, "min": 2, "average": 2.0},
        ]
        cuckoo_format = False
        clusters = clustering(data, cuckoo_format)
        self.assertEqual(cluster_l, len(clusters))
        for i, cluster in enumerate(clusters):
            self.assertEqual(res_ngram[i], cluster.ngram)
            self.assertEqual(res_l[i], len(cluster.similars))
            self.assertEqual(res[i], cluster.similars)
            self.assertEqual(res_stats[i]["unique"], cluster.unique)
            self.assertEqual(res_stats[i]["max"], cluster.max)
            self.assertEqual(res_stats[i]["min"], cluster.min)
            self.assertEqual(res_stats[i]["average"], cluster.average)

    def test_set4(self):
        data = {
            "sample1": [
                "SPECIFICATION",
                "SPECC",
                "SPECCN",
                "SPECEFICATION",
                "SPECEN",
                "SPECFICATION",
                "SPECIFCATION",
                "ARRANGEMENT",
                "ARRAGEMENT",
                "ARRAGMENT",
                "ARRANCEMENT",
                "ARRANGEME",
                "ARRANGEMEMT",
                "ARRANGEMEN",
            ]
        }
        cluster_l = 4
        res_l = [1, 1, 1, 1]
        res = [
            defaultdict(
                list,
                {
                    "sample1": [
                        "ARRANGEMENT",
                        "ARRAGEMENT",
                        "ARRANGEMEN",
                    ]
                },
            ),
            defaultdict(
                list,
                {
                    "sample1": [
                        "ARRANGEME",
                        "ARRANGEMEMT",
                    ]
                },
            ),
            defaultdict(
                list,
                {
                    "sample1": [
                        "SPECC",
                        "SPECCN",
                    ]
                },
            ),
            defaultdict(
                list,
                {
                    "sample1": [
                        "SPECIFICATION",
                        "SPECEFICATION",
                        "SPECFICATION",
                        "SPECIFCATION",
                    ]
                },
            ),
        ]
        res_ngram = ["GEMEN", "ARRAN", "SPECC", "CATIO"]
        res_stats = [
            {"unique": 3, "max": 3, "min": 3, "average": 3.0},
            {"unique": 2, "max": 2, "min": 2, "average": 2.0},
            {"unique": 2, "max": 2, "min": 2, "average": 2.0},
            {"unique": 4, "max": 4, "min": 4, "average": 4.0},
        ]
        cuckoo_format = False
        clusters = clustering(data, cuckoo_format)
        self.assertEqual(cluster_l, len(clusters))
        for i, cluster in enumerate(clusters):
            self.assertEqual(res_ngram[i], cluster.ngram)
            self.assertEqual(res_l[i], len(cluster.similars))
            self.assertEqual(res[i], cluster.similars)
            self.assertEqual(res_stats[i]["unique"], cluster.unique)
            self.assertEqual(res_stats[i]["max"], cluster.max)
            self.assertEqual(res_stats[i]["min"], cluster.min)
            self.assertEqual(res_stats[i]["average"], cluster.average)

    def test_set5(self):
        data = {
            "sample1": ["helloabc", "helloefg"],
            "sample2": ["hello123", "hello456", "helloabc"],
        }
        cluster_l = 1
        res_l = [2]
        res = [
            defaultdict(
                list,
                {
                    "sample1": ["helloabc", "helloefg"],
                    "sample2": ["helloabc", "hello123", "hello456"],
                },
            )
        ]
        res_ngram = ["hell"]
        res_stats = [{"unique": 4, "max": 3, "min": 2, "average": 2.5}]
        cuckoo_format = False
        clusters = clustering(data, cuckoo_format)
        self.assertEqual(cluster_l, len(clusters))
        for i, cluster in enumerate(clusters):
            self.assertEqual(res_ngram[i], cluster.ngram)
            self.assertEqual(res_l[i], len(cluster.similars))
            self.assertEqual(res[i], cluster.similars)
            self.assertEqual(res_stats[i]["unique"], cluster.unique)
            self.assertEqual(res_stats[i]["max"], cluster.max)
            self.assertEqual(res_stats[i]["min"], cluster.min)
            self.assertEqual(res_stats[i]["average"], cluster.average)

    def test_set6(self):
        data = {
            "sample1": ["helloabc", "helloefg", "helloabc"],
            "sample2": ["hello123", "hello456", "helloabc"],
        }
        cluster_l = 1
        res_l = [2]
        res = [
            defaultdict(
                list,
                {
                    "sample1": ["helloabc", "helloabc", "helloefg"],
                    "sample2": ["helloabc", "hello123", "hello456"],
                },
            )
        ]
        res_ngram = ["hell"]
        res_stats = [{"unique": 4, "max": 3, "min": 3, "average": 3.0}]
        cuckoo_format = False
        clusters = clustering(data, cuckoo_format)
        self.assertEqual(cluster_l, len(clusters))
        for i, cluster in enumerate(clusters):
            self.assertEqual(res_ngram[i], cluster.ngram)
            self.assertEqual(res_l[i], len(cluster.similars))
            self.assertEqual(res[i], cluster.similars)
            self.assertEqual(res_stats[i]["unique"], cluster.unique)
            self.assertEqual(res_stats[i]["max"], cluster.max)
            self.assertEqual(res_stats[i]["min"], cluster.min)
            self.assertEqual(res_stats[i]["average"], cluster.average)

    def test_prep1(self):
        data = ["helloabc", "helloefg", "helloabc", "hello123", "hello456", "helloabc"]
        res = ["helloabc", "helloefg", "helloabc", "hello123", "hello456", "helloabc"]
        corpus = Corpus()
        for i, cluster in enumerate(data):
            prep = corpus.substitute(cluster)
            self.assertEqual(res[i], prep)

    def test_prep2(self):
        data = ["helloabc", "helloefg", "helloabc", "hello123", "hello456", "helloabc"]
        res = ["helloabc", "helloefg", "helloabc", "hello123", "hello456", "helloabc"]
        corpus = Corpus()
        for i, cluster in enumerate(data):
            prep = corpus.cuckoo_preprocessing(cluster)
            prep = corpus.substitute(prep)
            self.assertEqual(res[i], prep)

    def test_prep3(self):
        data = [r"\x64", r"false_path\x64\x63", r"path\x64\rest"]
        res = ["d", "false_pathdc", r"path\x64\rest"]
        corpus = Corpus()
        for i, cluster in enumerate(data):
            prep = corpus.cuckoo_preprocessing(cluster)
            prep = corpus.substitute(prep)
            self.assertEqual(res[i], prep)

    def test_prep4(self):
        data = [r"\x64", r"false_path\x64\x63", r"path\x64\rest"]
        res = [r"\x64", r"false_path\x64\x63", r"path\x64\rest"]
        corpus = Corpus()
        for i, cluster in enumerate(data):
            prep = corpus.substitute(cluster)
            self.assertEqual(res[i], prep)

    def test_prep5(self):
        data = [r"\xFF", r"\x00\xab\x11\xFF"]
        res = ["ÿ", "\x00«\x11ÿ"]
        corpus = Corpus()
        for i, cluster in enumerate(data):
            prep = corpus.cuckoo_preprocessing(cluster)
            prep = corpus.substitute(prep)
            self.assertEqual(res[i], prep)

    def test_prep6(self):
        data = [r"\\.\C:\Program Files\path_to_something"]
        res = [r"C:\Program Files\path_to_something"]
        corpus = Corpus()
        for i, cluster in enumerate(data):
            prep = corpus.cuckoo_preprocessing(cluster)
            prep = corpus.substitute(prep)
            self.assertEqual(res[i], prep)

    def test_prep7(self):
        data = [r"\\?\C:\Program Files\path_to_something"]
        res = [r"C:\Program Files\path_to_something"]
        corpus = Corpus()
        for i, cluster in enumerate(data):
            prep = corpus.cuckoo_preprocessing(cluster)
            prep = corpus.substitute(prep)
            self.assertEqual(res[i], prep)

    def test_prep8(self):
        data = [r"\??\C:\Program Files\path_to_something"]
        res = [r"C:\Program Files\path_to_something"]
        corpus = Corpus()
        for i, cluster in enumerate(data):
            prep = corpus.cuckoo_preprocessing(cluster)
            prep = corpus.substitute(prep)
            self.assertEqual(res[i], prep)

    def test_prep9(self):
        data = [r"\??\mailslot\server\path_to_something"]
        res = [r"path_to_something"]
        corpus = Corpus()
        for i, cluster in enumerate(data):
            prep = corpus.cuckoo_preprocessing(cluster)
            prep = corpus.substitute(prep)
            self.assertEqual(res[i], prep)

    def test_prep10(self):
        data = [r"\??\mailslot\path_to_something"]
        res = [r"path_to_something"]
        corpus = Corpus()
        for i, cluster in enumerate(data):
            prep = corpus.cuckoo_preprocessing(cluster)
            prep = corpus.substitute(prep)
            self.assertEqual(res[i], prep)

    def test_prep11(self):
        data = [r"\??\path_to_something"]
        res = [r"path_to_something"]
        corpus = Corpus()
        for i, cluster in enumerate(data):
            prep = corpus.cuckoo_preprocessing(cluster)
            prep = corpus.substitute(prep)
            self.assertEqual(res[i], prep)

    def test_prep12(self):
        data = [r"\KernelObjects\path_to_something"]
        res = [r"path_to_something"]
        corpus = Corpus()
        for i, cluster in enumerate(data):
            prep = corpus.cuckoo_preprocessing(cluster)
            prep = corpus.substitute(prep)
            self.assertEqual(res[i], prep)

    def test_prep13(self):
        data = {"sample1": ["MutexA", "MutexB"], "sample2": ["MutexY", "MutexZ"]}

        cluster_l = 1
        res_l = [2]
        res = [
            defaultdict(
                list, {"sample1": ["MutexA", "MutexB"], "sample2": ["MutexY", "MutexZ"]}
            )
        ]
        res_ngram = ["Mute"]
        res_stats = [{"unique": 4, "max": 2, "min": 2, "average": 2.0}]
        cuckoo_format = False
        clusters = clustering(data, cuckoo_format)
        assert cluster_l == len(clusters)
        for i, cluster in enumerate(clusters):
            assert res_ngram[i] == cluster.ngram
            assert res_l[i] == len(cluster.similars)
            assert res[i] == cluster.similars
            assert res_stats[i]["unique"] == cluster.unique
            assert res_stats[i]["max"] == cluster.max
            assert res_stats[i]["min"] == cluster.min
            assert res_stats[i]["average"] == cluster.average

    def test_prep14(self):
        data = {
            "sample1": ["hello123", "hello456"],
            "sample2": ["helloabc", "helloefg"],
        }
        cluster_l = 1
        res_l = [2]
        res = [
            defaultdict(
                list,
                {
                    "sample1": ["hello123", "hello456"],
                    "sample2": ["helloabc", "helloefg"],
                },
            )
        ]
        res_ngram = ["hell"]
        res_stats = [{"unique": 4, "max": 2, "min": 2, "average": 2.0}]
        cuckoo_format = False
        clusters = clustering(data, cuckoo_format)
        self.assertEqual(cluster_l, len(clusters))
        for i, cluster in enumerate(clusters):
            self.assertEqual(res_ngram[i], cluster.ngram)
            self.assertEqual(res_l[i], len(cluster.similars))
            self.assertEqual(res[i], cluster.similars)
            self.assertEqual(res_stats[i]["unique"], cluster.unique)
            self.assertEqual(res_stats[i]["max"], cluster.max)
            self.assertEqual(res_stats[i]["min"], cluster.min)
            self.assertEqual(res_stats[i]["average"], cluster.average)

    def test_prep15(self):
        data = {
            "sample1": [
                "abcdef",
                "abcdeabcde",
                "abcdexyz",
                "1239999",
                "4569999",
                "hello123",
                "hello456",
                "helloabc",
                "xxnecoc",
                "yynecoc",
            ]
        }
        cluster_l = 4
        res_l = [1, 1, 1, 1]
        res = [
            defaultdict(list, {"sample1": ["1239999", "4569999"]}),
            defaultdict(list, {"sample1": ["abcdef", "abcdexyz"]}),
            defaultdict(list, {"sample1": ["hello123", "hello456", "helloabc"]}),
            defaultdict(list, {"sample1": ["xxnecoc", "yynecoc"]}),
        ]
        res_ngram = ["9999", "abcd", "hell", "neco"]
        res_stats = [
            {"unique": 2, "max": 2, "min": 2, "average": 2.0},
            {"unique": 2, "max": 2, "min": 2, "average": 2.0},
            {"unique": 3, "max": 3, "min": 3, "average": 3.0},
            {"unique": 2, "max": 2, "min": 2, "average": 2.0},
        ]
        cuckoo_format = False
        clusters = clustering(data, cuckoo_format)
        self.assertEqual(cluster_l, len(clusters))
        for i, cluster in enumerate(clusters):
            self.assertEqual(res_ngram[i], cluster.ngram)
            self.assertEqual(res_l[i], len(cluster.similars))
            self.assertEqual(res[i], cluster.similars)
            self.assertEqual(res_stats[i]["unique"], cluster.unique)
            self.assertEqual(res_stats[i]["max"], cluster.max)
            self.assertEqual(res_stats[i]["min"], cluster.min)
            self.assertEqual(res_stats[i]["average"], cluster.average)

    def test_prep16(self):
        data = {
            "sample1": [
                "abcdef",
                "abcdeabcde",
                "abcdexyz",
                "1239999",
                "4569999",
                "hello123",
                "hello456",
                "helloabc",
                "xxnecoc",
                "yynecoc",
            ]
        }
        cluster_l = 4
        res_l = [1, 1, 1, 1]
        res = [
            defaultdict(list, {"sample1": ["1239999", "4569999"]}),
            defaultdict(list, {"sample1": ["abcdef", "abcdexyz"]}),
            defaultdict(list, {"sample1": ["hello123", "hello456", "helloabc"]}),
            defaultdict(list, {"sample1": ["xxnecoc", "yynecoc"]}),
        ]
        res_ngram = ["9999", "abcd", "hell", "neco"]
        res_stats = [
            {"unique": 2, "max": 2, "min": 2, "average": 2.0},
            {"unique": 2, "max": 2, "min": 2, "average": 2.0},
            {"unique": 3, "max": 3, "min": 3, "average": 3.0},
            {"unique": 2, "max": 2, "min": 2, "average": 2.0},
        ]
        cuckoo_format = False
        clusters = clustering(data, cuckoo_format)
        self.assertEqual(cluster_l, len(clusters))
        for i, cluster in enumerate(clusters):
            self.assertEqual(res_ngram[i], cluster.ngram)
            self.assertEqual(res_l[i], len(cluster.similars))
            self.assertEqual(res[i], cluster.similars)
            self.assertEqual(res_stats[i]["unique"], cluster.unique)
            self.assertEqual(res_stats[i]["max"], cluster.max)
            self.assertEqual(res_stats[i]["min"], cluster.min)
            self.assertEqual(res_stats[i]["average"], cluster.average)

    def test_prep_with_types1(self):
        data = {
            "sample1": ["helloabc", "helloefg", "helloabc"],
            "sample2": ["hello123", "hello456", "helloabc"],
        }
        input_type = "mutex"
        cluster_l = 1
        res_l = [2]
        res = [
            defaultdict(
                list,
                {
                    "sample1": ["helloabc", "helloabc", "helloefg"],
                    "sample2": ["helloabc", "hello123", "hello456"],
                },
            )
        ]
        res_ngram = ["hell"]
        res_type = ["mutex"]
        res_stats = [{"unique": 4, "max": 3, "min": 3, "average": 3.0}]
        cuckoo_format = False
        clusters = clustering(data, cuckoo_format, input_type=input_type)
        self.assertEqual(cluster_l, len(clusters))
        for i, cluster in enumerate(clusters):
            self.assertEqual(res_ngram[i], cluster.ngram)
            self.assertEqual(res_type[i], cluster.input_type)
            self.assertEqual(res_l[i], len(cluster.similars))
            self.assertEqual(res[i], cluster.similars)
            self.assertEqual(res_stats[i]["unique"], cluster.unique)
            self.assertEqual(res_stats[i]["max"], cluster.max)
            self.assertEqual(res_stats[i]["min"], cluster.min)
            self.assertEqual(res_stats[i]["average"], cluster.average)

    def test_prep_with_types2(self):
        data = {
            "sample1": ["helloabc", "helloefg", "helloabc"],
            "sample2": ["hello123", "hello456", "helloabc"],
        }
        input_type = "http_request"
        cluster_l = 1
        res_l = [2]
        res = [
            defaultdict(
                list,
                {
                    "sample1": ["helloabc", "helloabc", "helloefg"],
                    "sample2": ["helloabc", "hello123", "hello456"],
                },
            )
        ]
        res_ngram = ["hell"]
        res_type = ["http_request", "http_request"]
        res_stats = [{"unique": 4, "max": 3, "min": 3, "average": 3.0}]
        cuckoo_format = False
        clusters = clustering(data, cuckoo_format, input_type=input_type)
        self.assertEqual(cluster_l, len(clusters))
        for i, cluster in enumerate(clusters):
            self.assertEqual(res_ngram[i], cluster.ngram)
            self.assertEqual(res_type[i], cluster.input_type)
            self.assertEqual(res_l[i], len(cluster.similars))
            self.assertEqual(res[i], cluster.similars)
            self.assertEqual(res_stats[i]["unique"], cluster.unique)
            self.assertEqual(res_stats[i]["max"], cluster.max)
            self.assertEqual(res_stats[i]["min"], cluster.min)
            self.assertEqual(res_stats[i]["average"], cluster.average)

    def test_prep_with_types3(self):
        data = {
            "sample1": [
                "abcdef",
                "abcdeabcde",
                "abcdexyz",
                "hello123",
                "hello456",
                "helloabc",
                "xxnecoc",
                "yynecoc",
            ],
            "sample2": [
                "1239999",
                "4569999",
            ],
        }
        cluster_l = 4
        res_l = [1, 1, 1, 1]
        res = [
            defaultdict(list, {"sample2": ["1239999", "4569999"]}),
            defaultdict(list, {"sample1": ["abcdef", "abcdexyz"]}),
            defaultdict(list, {"sample1": ["hello123", "hello456", "helloabc"]}),
            defaultdict(list, {"sample1": ["xxnecoc", "yynecoc"]}),
        ]
        res_ngram = ["9999", "abcd", "hell", "neco"]
        input_type = "mutex"
        res_stats = [
            {"unique": 2, "max": 2, "min": 2, "average": 2.0},
            {"unique": 2, "max": 2, "min": 2, "average": 2.0},
            {"unique": 3, "max": 3, "min": 3, "average": 3.0},
            {"unique": 2, "max": 2, "min": 2, "average": 2.0},
        ]
        res_type = ["mutex", "mutex", "mutex", "mutex"]
        cuckoo_format = False
        clusters = clustering(data, cuckoo_format, input_type=input_type)
        self.assertEqual(cluster_l, len(clusters))
        for i, cluster in enumerate(clusters):
            self.assertEqual(res_ngram[i], cluster.ngram)
            self.assertEqual(res_type[i], cluster.input_type)
            self.assertEqual(res_l[i], len(cluster.similars))
            self.assertEqual(res[i], cluster.similars)
            self.assertEqual(res_stats[i]["unique"], cluster.unique)
            self.assertEqual(res_stats[i]["max"], cluster.max)
            self.assertEqual(res_stats[i]["min"], cluster.min)
            self.assertEqual(res_stats[i]["average"], cluster.average)

    def test_prep_with_types4(self):
        data = {
            "sample1": [
                "abcdef",
                "abcdeabcde",
                "abcdexyz",
                "hello123",
                "hello456",
                "helloabc",
                "xxnecoc",
                "yynecoc",
            ],
            "sample2": [
                "1239999",
                "4569999",
            ],
        }
        cluster_l = 4
        res_l = [1, 1, 1, 1]
        res = [
            defaultdict(list, {"sample2": ["1239999", "4569999"]}),
            defaultdict(list, {"sample1": ["abcdef", "abcdexyz"]}),
            defaultdict(list, {"sample1": ["hello123", "hello456", "helloabc"]}),
            defaultdict(list, {"sample1": ["xxnecoc", "yynecoc"]}),
        ]
        res_ngram = ["9999", "abcd", "hell", "neco"]
        input_type = "mutex"
        res_stats = [
            {"unique": 2, "max": 2, "min": 2, "average": 2.0},
            {"unique": 2, "max": 2, "min": 2, "average": 2.0},
            {"unique": 3, "max": 3, "min": 3, "average": 3.0},
            {"unique": 2, "max": 2, "min": 2, "average": 2.0},
        ]
        res_type = ["mutex", "mutex", "mutex", "mutex"]
        cuckoo_format = False
        clusters = clustering(data, cuckoo_format, input_type=input_type)
        self.assertEqual(cluster_l, len(clusters))
        for i, cluster in enumerate(clusters):
            self.assertEqual(res_ngram[i], cluster.ngram)
            self.assertEqual(res_type[i], cluster.input_type)
            self.assertEqual(res_l[i], len(cluster.similars))
            self.assertEqual(res[i], cluster.similars)
            self.assertEqual(res_stats[i]["unique"], cluster.unique)
            self.assertEqual(res_stats[i]["max"], cluster.max)
            self.assertEqual(res_stats[i]["min"], cluster.min)
            self.assertEqual(res_stats[i]["average"], cluster.average)

    def test_prep_with_types5(self):
        data = {
            "sample1": [
                "abcdef",
                "abcdeabcde",
                "abcdexyz",
                "hello123",
                "hello456",
                "helloabc",
                "xxnecoc",
                "yynecoc",
            ],
            "sample2": [
                "1239999",
                "4569999",
            ],
        }
        cluster_l = 4
        res_l = [1, 1, 1, 1]
        res = [
            defaultdict(list, {"sample2": ["1239999", "4569999"]}),
            defaultdict(list, {"sample1": ["abcdef", "abcdexyz"]}),
            defaultdict(list, {"sample1": ["hello123", "hello456", "helloabc"]}),
            defaultdict(list, {"sample1": ["xxnecoc", "yynecoc"]}),
        ]
        res_ngram = ["9999", "abcd", "hell", "neco"]
        input_type = "http_post"
        res_stats = [
            {"unique": 2, "max": 2, "min": 2, "average": 2.0},
            {"unique": 2, "max": 2, "min": 2, "average": 2.0},
            {"unique": 3, "max": 3, "min": 3, "average": 3.0},
            {"unique": 2, "max": 2, "min": 2, "average": 2.0},
        ]
        res_type = ["http_post", "http_post", "http_post", "http_post"]
        cuckoo_format = False
        clusters = clustering(data, cuckoo_format, input_type=input_type)
        self.assertEqual(cluster_l, len(clusters))
        for i, cluster in enumerate(clusters):
            self.assertEqual(res_ngram[i], cluster.ngram)
            self.assertEqual(res_type[i], cluster.input_type)
            self.assertEqual(res_l[i], len(cluster.similars))
            self.assertEqual(res[i], cluster.similars)
            self.assertEqual(res_stats[i]["unique"], cluster.unique)
            self.assertEqual(res_stats[i]["max"], cluster.max)
            self.assertEqual(res_stats[i]["min"], cluster.min)
            self.assertEqual(res_stats[i]["average"], cluster.average)

    def test_prep_with_types6(self):
        data = {
            "sample1": [
                "abcdef",
                "abcdeabcde",
                "abcdexyz",
                "hello123",
                "hello456",
                "helloabc",
                "xxnecoc",
                "yynecoc",
            ],
            "sample2": [
                "1239999",
                "4569999",
            ],
            "sample3": [
                "hello123",
                "hello456",
            ],
        }
        cluster_l = 4
        res_l = [1, 1, 2, 1]
        res = [
            defaultdict(list, {"sample2": ["1239999", "4569999"]}),
            defaultdict(list, {"sample1": ["abcdef", "abcdexyz"]}),
            defaultdict(
                list,
                {
                    "sample1": ["hello123", "hello456", "helloabc"],
                    "sample3": ["hello123", "hello456"],
                },
            ),
            defaultdict(list, {"sample1": ["xxnecoc", "yynecoc"]}),
        ]
        res_ngram = ["9999", "abcd", "hell", "neco"]
        input_type = "http_get"
        res_stats = [
            {"unique": 2, "max": 2, "min": 2, "average": 2.0},
            {"unique": 2, "max": 2, "min": 2, "average": 2.0},
            {"unique": 3, "max": 3, "min": 2, "average": 2.5},
            {"unique": 2, "max": 2, "min": 2, "average": 2.0},
        ]
        res_type = [
            "http_get",
            "http_get",
            "http_get",
            "http_get",
        ]
        cuckoo_format = False
        clusters = clustering(data, cuckoo_format, input_type=input_type)
        self.assertEqual(cluster_l, len(clusters))
        for i, cluster in enumerate(clusters):
            self.assertEqual(res_ngram[i], cluster.ngram)
            self.assertEqual(res_type[i], cluster.input_type)
            self.assertEqual(res_l[i], len(cluster.similars))
            print(cluster.similars)
            self.assertEqual(res[i], cluster.similars)
            self.assertEqual(res_stats[i]["unique"], cluster.unique)
            self.assertEqual(res_stats[i]["max"], cluster.max)
            self.assertEqual(res_stats[i]["min"], cluster.min)
            self.assertEqual(res_stats[i]["average"], cluster.average)


if __name__ == "__main__":
    pytest.main()
