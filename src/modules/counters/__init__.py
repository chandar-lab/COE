from .hash_count import HashCount
from .hash_dict_count import HashDictCount

REGISTRY = {
    "simhash": HashCount,
    "simhash_dict": HashDictCount,
}
