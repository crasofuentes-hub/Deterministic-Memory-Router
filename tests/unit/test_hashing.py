from memory_router.utils.hashing import sha256_hex
def test_sha256_hex_stable():
    assert sha256_hex({"a":1,"b":2}) == sha256_hex({"b":2,"a":1})