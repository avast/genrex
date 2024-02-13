"""
Clustering:
Each sample is stored as Sample class with a resource (filename) and all ngrams.
Then, the index dictionary is created - keys are ngrams, and the values are
lists of samples that contain these ngrams.
In the corpus function, we iterate through the list of samples and create
the set of samples that we found in the same index list.
Every sample is added to the cluster only once.
Results are in class Cluster.
"""

import ntpath
import re
from collections import defaultdict
from dataclasses import dataclass, field
from statistics import mean

from genrex.enums import InputType
from genrex.logging import logger
from genrex.misc import filter_ngrams, ready_to_print, string2ngrams


@dataclass(slots=True)
class Cluster:
    ngram: str = ""
    similars: defaultdict[str, list] = field(default_factory=lambda: defaultdict(list))
    originals: set[str] = field(default_factory=set)
    input_type: str = ""
    regex: str = ""
    unique: int = 0

    @property
    def average(self) -> float:
        lengths = [len(x) for x in self.similars.values()]
        return sum(lengths) / len(lengths)

    @property
    def min(self) -> float:
        return min(len(x) for x in self.similars.values())

    @property
    def max(self) -> float:
        return max(len(x) for x in self.similars.values())

    def __str__(self) -> str:
        res_set = set()
        for similar in self.similars.values():
            res_set.update(similar)
        return (
            "Regex: {0}\n"
            + "Ngram: {1}\n"
            + "Unique: {2}\n"
            + "Min: {3}\n"
            + "Max: {4}\n"
            + "Average: {5}\n"
            + "Resources: {6}\n"
            + "Originals: {7}\n"
            + "Named object type: {8}\n"
            + "Hashes: {9}\n"
        ).format(
            self.regex,
            ready_to_print(self.ngram),
            self.unique,
            self.min,
            self.max,
            self.average,
            sorted(list(ready_to_print(x) for x in res_set)),
            sorted(list(self.originals)),
            self.input_type,
            sorted(list(self.similars.keys())),
        )

    def return_printable_dict(self):
        result = {}
        res_set = set()
        for similar in self.similars.values():
            res_set.update(similar)
        result["regex"] = self.regex
        result["ngram"] = ready_to_print(self.ngram)
        result["unique"] = self.unique
        result["min"] = self.min
        result["max"] = self.max
        result["average"] = self.average
        result["resources"] = sorted(list(ready_to_print(x) for x in res_set))
        result["originals"] = sorted(list(self.originals))
        result["input_type"] = self.input_type
        result["hashes"] = sorted(list(self.similars.keys()))
        return result


class Sample:
    def __init__(self, original: str, sample: str, filename: str):
        self.original = original
        self.sample = sample
        self.resource = filename

    def __repr__(self):
        return self.sample


class Corpus:
    def __init__(self, store_original_strings: bool = False):
        self.store_original_strings = store_original_strings
        self.samples: dict = defaultdict(dict)
        self.unique_ngrams: dict = defaultdict(dict)
        self.index: dict = defaultdict(dict)
        self.min_ngram = 4
        self.input_type: str = ""

    def add_resource(
        self,
        data: dict,
        cuckoo_format: bool = False,
        input_type: str = "",
    ):
        if len(data) == 0:
            logger.info("No input data. Exiting.")
            return

        self.input_type = input_type
        for source, strings in data.items():
            for string in strings:
                if len(string) < self.min_ngram:
                    logger.warning(f"String {string} is too short")
                    continue

                prep_string = string

                if cuckoo_format:
                    prep_string = self.cuckoo_preprocessing(string)

                prep_string = self.substitute(prep_string)

                if self.is_guid(prep_string):
                    logger.warning(f"String {string} is GUID")
                    continue

                self.add(string, prep_string, source)

        self.extract_ngrams(self.input_type)

    def estimate_len_of_ngram(self):
        if len(self.samples) == 0:
            logger.info("No input data after filtering. Exiting.")
            return
        self.ngrams = sum(map(len, self.samples)) // (2 * len(self.samples))
        if len(self.samples) < 10:
            self.ngrams = self.ngrams // 2
        self.ngrams = max(self.ngrams, self.min_ngram)

    def filter_short_strings(self):
        self.samples = dict(
            filter(lambda x: (len(x[0]) >= self.ngrams), self.samples.items())
        )

    def extract_ngrams(self, input_type):
        splited_list = {}
        for prep_string in self.samples:
            self.unique_ngrams[prep_string] = []
            ngram_string = prep_string
            for remove_string in filter_ngrams["url"] + filter_ngrams["others"]:
                pattern = re.compile(remove_string, re.IGNORECASE)
                ngram_string = pattern.sub("^", ngram_string)

            if len(ngram_string) < self.min_ngram:
                continue

            splited_list[prep_string] = re.split(
                r"/|\^|\\|\?|!|\+|&|=|\.", ngram_string
            )

        splited = [len(substr) for k in splited_list.values() for substr in k]
        if len(splited) == 0:
            logger.info("No input data after filtering. Exiting.")
            return

        self.ngrams = sum(splited) // (len(splited))
        self.ngrams = self.ngrams // 2
        self.ngrams = max(self.ngrams, self.min_ngram)
        if input_type in [
            InputType.FILE_ACCESS,
            InputType.KEY_ACCESS,
        ]:
            self.ngrams = self.ngrams * 4

        for key, value in splited_list.items():
            for part in value:
                if len(part) < self.ngrams:
                    continue
                self.unique_ngrams[key].extend(string2ngrams(part, self.ngrams))

    def is_guid(self, string: str) -> bool:
        found = re.match(
            r"^(\{)?[A-Za-z0-9]{8}-[A-Za-z0-9]{4}-[A-Za-z0-9]{4}-[A-Za-z0-9]{4}-[A-Za-z0-9]{12}(\})?$",
            ntpath.basename(string),
        )
        if found is not None:
            return True

        return False

    def cuckoo_preprocessing(self, string: str) -> str:
        string = string.replace(r"\\", "\\")
        string = re.sub(
            r"\\x[a-fA-F0-9]{2}",
            lambda x: (
                x.group(0)
                if (x.group(0) == r"\x64" or x.group(0) == r"\x86")
                and x.end() + 1 < len(string)
                and string[x.end() + 1] != "x"
                else x.group(0).encode("latin1").decode("unicode_escape")
            ),
            string,
        )
        return string

    def substitute(self, string: str) -> str:
        first_character = string[0]
        if first_character == "\\":
            string = self.substitute_prefix(string)
        elif first_character == "G":
            if string.startswith("Global\\"):
                string = string.replace("Global\\", "")
        elif first_character == "L":
            if string.startswith("Local\\"):
                string = string.replace("Local\\", "")
        elif first_character == "S":
            found = re.match("^Session\\\\\\d+\\\\", string)
            if found is not None:
                string = string[found.span()[1] :]
        elif first_character == "C":
            string = self.substitute_username(string)
        elif first_character == "H":
            string = self.remove_local_current(string)
        return string

    def substitute_prefix(self, string: str) -> str:
        found = re.match("^\\\\Sessions\\\\\\d+\\\\BaseNamedObjects\\\\", string)
        if found is not None:
            string = string[found.span()[1] :]
            return string

        if string.startswith(r"\.\C:\Program Files"):
            string = string.replace("\\.\\", "")
            return string

        if string.startswith(r"\?\C:\Program Files"):
            string = string.replace("\\?\\", "")
            return string

        if string.startswith(r"\??\C:\Program Files"):
            string = string.replace("\\??\\", "")
            return string

        replace_list = [
            "\\BaseNamedObjects\\",
            "\\BaseNamedObj\\",
            "\\??\\mailslot\\server\\",
            "\\??\\mailslot\\",
            "\\??\\",
            "\\KernelObjects\\",
        ]

        for replace_string in replace_list:
            if string.startswith(replace_string):
                string = string.replace(replace_string, "")
                return string

        return string

    def substitute_username(self, string: str) -> str:
        found = re.match("^C:\\\\Users\\\\[^\\\\]+", string)
        if found is not None:
            string = r"C:\Users\[^\]+" + string[found.span()[1] :]
            return string

        found = re.match("^C:\\\\Documents and Settings\\\\[^\\\\]+", string)
        if found is not None:
            string = r"C:\Documents and Settings\[^\]+" + string[found.span()[1] :]
            return string

        found = re.match(r"^C:\?DOCUMENTS AND SETTINGS\?[^?]+", string)
        if found is not None:
            string = r"C:?DOCUMENTS AND SETTINGS?[^?]+" + string[found.span()[1] :]
            return string

        return string

    def remove_local_current(self, string: str) -> str:
        if string.startswith("HKEY_LOCAL_MACHINE\\"):
            string = string.replace("HKEY_LOCAL_MACHINE\\", "")
        elif string.startswith("HKLM\\"):
            string = string.replace("HKLM\\", "")
        elif string.startswith("HKEY_CURRENT_USER\\"):
            string = string.replace("HKEY_CURRENT_USER\\", "")
        elif string.startswith("HKCU\\"):
            string = string.replace("HKCU\\", "")
        elif string.startswith("HKEY_CLASSES_ROOT\\"):
            string = string.replace("HKEY_CLASSES_ROOT\\", "")
        elif string.startswith("HKCR\\"):
            string = string.replace("HKCR\\", "")
        return string

    def add(self, string: str, prep_string: str, filename: str):
        if self.store_original_strings:
            sample = Sample(string, prep_string, filename)
        else:
            sample = Sample("", prep_string, filename)
        if prep_string in self.samples:
            self.samples[prep_string].append(sample)
        else:
            self.samples[prep_string] = [sample]

    def create_index(self):
        """
        Create a dictionary where the keys are unique ngrams and values are lists that
        contain these ngrams.
        """
        self.index = defaultdict(list)

        for sample in self.samples:
            for sequence in self.unique_ngrams[sample]:
                self.index[sequence].append(sample)

    def cluster(self) -> list[Cluster]:
        """
        The function iterates through the list of samples, and we create the set of samples that
        we found in the same index list.
        Every sample is added to the cluster only once.
        """
        seen = set()
        scores: dict = {}
        ngrams: dict = {}
        self.create_index()

        self.samples = dict(sorted(self.samples.items()))

        for sample in self.samples:
            if sample in seen:
                continue
            ngrams[sample] = ""

            seq = {}
            for sequence in self.unique_ngrams[sample]:
                if sequence in self.index:
                    seq[sequence] = self.index[sequence]

            seq = dict(sorted(seq.items(), key=lambda kv: len(kv[1]), reverse=True))

            for sequence, matches in seq.items():
                for match in matches:
                    if match in seen:
                        continue
                    if sample in scores:
                        scores[sample].append(match)
                    else:
                        scores[sample] = [match]
                        ngrams[sample] = sequence
                    seen.add(match)

        self.test_results(scores, ngrams)
        clusters = self.save_results(scores, ngrams)
        return clusters

    def test_results(self, scores: dict[str, list[str]], ngrams: dict[str, str]):
        welp_scores = {}
        welp_ngrams = {}
        for key, cluster in scores.items():
            cluster_checked = [
                x for x in cluster if ngrams[key] in x and isinstance(x, str)
            ]
            lentghs = [len(x) for x in cluster_checked]
            mean_value = mean(lentghs)
            first = [x for x in cluster_checked if len(x) <= mean_value + 1]
            second = [x for x in cluster_checked if len(x) > mean_value + 1]
            scores[key] = first
            if len(second) > 1:
                welp_ngrams[key + "+"] = ngrams[key]
                welp_scores[key + "+"] = second
        scores.update(welp_scores)  # type: ignore
        ngrams.update(welp_ngrams)  # type: ignore

    def save_results(self, scores: dict, ngrams: dict) -> list[Cluster]:
        clusters = []

        for key, cluster in scores.items():
            if len(cluster) == 1 and len(self.samples[cluster[0]]) == 1:
                logger.warning(f"Cluster {cluster} contains only one element")
                continue
            ngram = ngrams[key]
            strings = [x for x in cluster if isinstance(x, str)]
            cluster = Cluster(ngram=ngram, unique=len(strings))
            originals = set()
            cluster.input_type = self.input_type
            for string in strings:
                for similar in self.samples[string]:
                    # case: hello1 - ['example1', 'example2']
                    # c.similars[similar.sample].append(similar.resource)
                    # case: example1 - ['hello1', 'hello2']
                    cluster.similars[similar.resource].append(similar.sample)
                    if self.store_original_strings:
                        originals.add(similar.original)
            cluster.originals = originals

            clusters.append(cluster)

        return clusters
