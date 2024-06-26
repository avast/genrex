# GenRex 🦖
![PyPI](https://img.shields.io/pypi/v/genrex-py?label=genrex-py)

GenRex is a tool that generates regular expressions from strings such as artifacts dynamically generated by samples. 

For more information, check out:
- [Blog post](https://engineering.avast.io/know-your-yara-rules-series-6-we-present-genrex-a-generator-of-regular-expressions)
- [Wiki](https://www.github.com/avast/genrex/wiki)

## Citation 
For citing, please use the following entry for the [original paper](https://ieeexplore.ieee.org/document/10538538):
```
@misc{genrex-2023,
  doi = {10.1109/TrustCom60117.2023.00123},
  url = {https://ieeexplore.ieee.org/document/10538538},
  author = {Regeciova, Dominika and Kolar, Dusan},
  title = {GenRex: Leveraging Regular Expressions for Dynamic Malware Detection},
  publisher = {2023 IEEE 22nd International Conference on Trust, Security and Privacy in Computing and Communications (TrustCom)},
  year = {2023}
}
```

:snake: Minimal supported version of Python is `3.10`.

## Installation

```bash
pip install -U genrex-py
```

## How to Use

```python
import genrex

# pre-process the input from cuckoo_format reports (default: False)
cuckoo_format = False

# When True, GenRex will store the original strings as well,
# when False, GenRex will store the processed strings only
store_original_strings = True

results = genrex.generate(cuckoo_format=cuckoo_format, store_original_strings=store_original_strings, directory="samples")

print("Results:")

for result in results:
    print(result)


"""
Results:
Regex: hello[0-9a-f]                               # regular expression
Ngram: hell                                        # common part of strings in cluster
Unique: 6                                          # how many unique strings are in cluster
Min: 3                                             # minimal number of occurrences in samples
Max: 3                                             # maximal number of occurrences in samples
Average: 3.0                                       # average number of occurrences in samples
Resources: ['hello1', 'hello2', ..., 'helloc']     # list of preprocessed strings from cluster
Originals: []                                      # list of original strings from cluster
                                                   # (if store_original_strings is True)
Input type: ''                                     # input types (if defined)
Hashes: ['source1', 'source2']                     # list of sources

['input_type', 'ngram', 'original_regexes', 'originals', 'regex', 'similar_regex', 'similars', 'unique']
"""    
```

```python
import genrex

cuckoo_format = True

# When True, GenRex will store the original strings as well,
# when False, GenRex will store the processed strings only
store_original_strings = False

results = genrex.generate(
    cuckoo_format=cuckoo_format,
    store_original_strings=store_original_strings,
    input_type=genrex.InputType.MUTEX,
    source={
        "source1": ["helloa", "hellob", "helloc"],
        "source2": ["hello1", "hello2", "hello3"],
    }
)

print("Results:")

for result in results:
    res = result.return_printable_dict()
    print("Regex:", res["regex"])            # regular expression
    print("Ngram:", res["ngram"])            # common part of strings in cluster
    print("Unique:", res["unique"])          # how many unique strings are in cluster
    print("Min:", res["min"])                # minimal number of occurrences in samples
    print("Max:", res["max"])                # maximal number of occurrences in samples
    print("Avg:", res["average"])            # average number of occurrences in samples
    print("Resources:", res["resources"])    # list of preprocessed strings from cluster
    print("Type:", res["input_type"])        # input types (if defined)
    print("Hashes:", res["hashes"])          # list of sources

"""
Results:
Regex: (^|\\)hello[0-9a-f]$
Ngram: hell
Unique: 6
Min: 3
Max: 3
Avg: 3.0
Type: 'mutex'
Resources: ['helloc', 'hellob', 'hello3', 'hello1', 'helloa', 'hello2']
Hashes: ['source1', 'source2']
"""
```

## How to develop

Install GenRex in development mode with all necessary dependencies.

```bash
make setup
```

### Tests

You can run tests with the following command:

```bash
make tests
```

## License

Copyright (c) 2024 Avast Software, licensed under the MIT license. See the
[`LICENSE`](https://github.com/avast/genrex/blob/master/LICENSE) file for more
details.
