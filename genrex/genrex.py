import os
import re

from genrex.clustering import Cluster, Corpus
from genrex.logging import logger
from genrex.regex import Regex


def generate(
    cuckoo_format: bool = False,
    store_original_strings: bool = False,
    *,
    input_type: str = "",
    **kwargs,
) -> list[Cluster]:
    """
    Generate function and entrypoint for GenRex.
    Input:
        cuckoo_format - optional, bool, default False
        store_original_strings - optional, bool, default False
        input_type - optional, default empty
        input data - two options:
            source - a dictionary of input
            directory - a path to directory
    Output:
        a list of Clusters
    """
    data_set: dict = {}
    if "source" in kwargs:
        data_set = kwargs["source"]
        logger.info(f"Adding source data: {data_set}")
    elif "directory" in kwargs:
        data_set = load_data_dir(kwargs["directory"])
    else:
        logger.error("No source of data was added!")
        raise IOError("No source of data was added!")

    clusters: list[Cluster] = clustering(
        data_set, cuckoo_format, store_original_strings, input_type
    )
    return generate_regex(clusters)


def load_data_dir(filepath: str) -> dict:
    data_set: dict = {}
    is_valid_directory(filepath)
    datafiles: list[str] = os.listdir(filepath)
    for datafile in datafiles:
        with open(os.path.join(filepath, datafile), "r", encoding="utf-8") as file:
            data_set[datafile] = [line.strip() for line in file if line.strip()]
    logger.info(f"Adding directory data: {data_set}")
    return data_set


def is_valid_directory(filepath: str):
    if not os.path.exists(filepath):
        logger.error(f"The directory {filepath} does not exist!")
        raise IOError(f"The directory {filepath} does not exist!")


def clustering(
    data: dict,
    cuckoo_format: bool,
    store_original_strings: bool,
    input_type: str = "",
) -> list[Cluster]:
    corpus: Corpus = Corpus(store_original_strings)
    corpus.add_resource(data, cuckoo_format, input_type=input_type)
    clusters: list[Cluster] = corpus.cluster()
    logger.info(f"Number of created clusters: {str(len(clusters))}")
    return clusters


def generate_regex(clusters: list[Cluster]) -> list[Cluster]:
    results: list[Cluster] = []
    for cluster in clusters:
        logger.info(
            f"Generating regular expression for cluster {list(cluster.similars.values())}"
        )
        regex: Regex = Regex()
        regex.make_regex(cluster)
        cluster.regex = regex.get_results(reduce=False)
        logger.info(f"Regular expression '{cluster.regex}' generated")
        results.append(cluster)
    return optimized(results)


def duplicate(compare: Cluster, clusters: list[Cluster]) -> Cluster | None:
    for cluster in clusters:
        if cluster.regex == compare.regex:
            return cluster
    return None


def optimized(clusters: list[Cluster]) -> list[Cluster]:
    results: list[Cluster] = []
    for cluster in clusters:
        if re.search(r"\(\^\|\\\\\)\.\{.*\}\$$", cluster.regex):
            logger.info(f"Removing too general regular expressions: {cluster.regex}")
            continue
        join_to = duplicate(cluster, results)
        if join_to is not None:
            logger.info(f"Unifying clusters using regex {cluster.regex}")
            join_to.ngram = ""
            for k in cluster.similars.keys():
                if k in join_to.similars:
                    join_to.similars[k].extend(cluster.similars[k])
                else:
                    join_to.similars[k] = cluster.similars[k]
            join_to.originals.update(cluster.originals)
            join_to.input_type = cluster.input_type
            join_to.unique = len(join_to.originals)
        else:
            results.append(cluster)
    return results
