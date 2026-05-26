import pytest

torch = pytest.importorskip("torch")
base_mod = pytest.importorskip("morphling.backend.base")
BaseBackend = base_mod.BaseBackend
MatMulRequestMessage = base_mod.MatMulRequestMessage


def test_base_backend_is_abstract_and_requires_dispatch_implementation():
    with pytest.raises(TypeError):
        _ = BaseBackend()  # pyright: ignore[reportAbstractUsage]


def test_matmul_request_message_round_trip_serialization():
    mat_a = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    mat_b = torch.arange(16, dtype=torch.float32).reshape(4, 4)

    request = MatMulRequestMessage()
    request.set(mat_a, mat_b, r=1, c=1, block_size=2, ld=[7])

    payload = request.serialize()

    decoded = MatMulRequestMessage()
    decoded.deserialize(payload)

    assert decoded.r == 1
    assert decoded.c == 1
    assert list(decoded.ld) == [7]
    assert torch.equal(decoded.a, mat_a[2:4, :])
    assert torch.equal(decoded.b, mat_b[:, 2:4])
