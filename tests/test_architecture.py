import pytest
import torch

from tt_ecog.losses import make_loss
from tt_ecog.model import UNet1D, architecture, build_model, count_parameters, reference_architecture, search_space

pytestmark = pytest.mark.verification("V13")


def test_reference_architecture_matches_the_protocol(config):
    arch = reference_architecture(config)
    assert arch.key == "k5_L5_f64"
    assert arch.channels == (64, 64, 128, 256, 512, 512)
    assert arch.kernels == (7, 7, 5, 5, 5) and arch.dilations == (1, 2, 3, 1, 2)


def test_search_space_is_the_27_construction_rule(config):
    space = search_space(config)
    assert len({a.key for a in space}) == 27 and reference_architecture(config) in space
    for arch in space:
        f, levels, k = arch.base_filters, arch.encoder_levels, arch.kernel_size
        assert arch.channels == tuple([f, f, 2 * f, 4 * f, 8 * f, 8 * f][:levels + 1])
        assert arch.kernels == tuple([7, 7] + [k] * (levels - 2))
        assert arch.dilations == tuple([1, 2, 3, 1, 2][:levels])
    with pytest.raises(ValueError):
        architecture(config, 5, 6, 64)


def test_reference_structure(config):
    model = build_model(config, 46)
    assert model.reduce.conv.kernel_size == (3,) and model.reduce.conv.in_channels == 46 * 40
    assert [b.conv.dilation[0] for b in model.encoder] == [1, 2, 3, 1, 2]
    assert [b.conv.kernel_size[0] for b in model.decoder] == [5, 5, 5, 7, 7]
    assert [b.conv.in_channels for b in model.decoder] == [512, 1024, 512, 256, 128]
    assert model.head.in_channels == 128 and model.head.out_channels == 5 and model.head.kernel_size == (1,)
    assert all(isinstance(b.norm, torch.nn.BatchNorm1d) and b.dropout.p == 0.3 for b in [model.reduce, *model.encoder])
    assert isinstance(model.pool, torch.nn.MaxPool1d)


@pytest.mark.parametrize("index", range(27))
def test_every_architecture_runs_forward_and_backward(config, index):
    arch = search_space(config)[index]
    torch.manual_seed(0)
    model = build_model(config, 46, arch).train()
    x = torch.randn(2, 46, 40, 256)
    y = torch.rand(2, 5, 256)
    out = model(x)
    assert out.shape == (2, 5, 256)
    loss = make_loss(config)(out, y)
    loss.backward()
    assert torch.isfinite(loss)
    grads = [p.grad for p in model.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert count_parameters(model) > 0


@pytest.mark.parametrize("channels", [61, 46, 64])
def test_reference_for_each_subject_channel_count(config, channels):
    model = build_model(config, channels).eval()
    with torch.no_grad():
        assert model(torch.randn(1, channels, 40, 256)).shape == (1, 5, 256)


@pytest.mark.parametrize("length", [200, 255, 300])
def test_other_lengths_keep_their_length(config, length):
    model = build_model(config, 46, architecture(config, 3, 3, 32)).eval()
    with torch.no_grad():
        assert model(torch.randn(1, 46, 40, length)).shape == (1, 5, length)


def test_wrong_input_shape_is_rejected(config):
    model = build_model(config, 46)
    for shape in ((2, 45, 40, 256), (2, 46, 39, 256), (46, 40, 256)):
        with pytest.raises(ValueError):
            model(torch.randn(*shape))


def test_same_seed_same_initialisation(config):
    torch.manual_seed(7)
    a = build_model(config, 46)
    torch.manual_seed(7)
    b = build_model(config, 46)
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))
    assert isinstance(a, UNet1D)
