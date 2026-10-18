import h5py
import numpy as np
import pytest
from scipy.io import savemat

from tt_ecog import io
from conftest import write_subject


def test_selective_classic_read_and_hashes(config, data_root, monkeypatch):
    path = write_subject(data_root, config)
    before = io.sha256_file(path)
    called = []
    real_reader = io.loadmat

    def spy(path, **kwargs):
        called.append(kwargs["variable_names"])
        return real_reader(path, **kwargs)

    monkeypatch.setattr(io, "loadmat", spy)
    record = io.read_ecog(config, data_root, "S1")
    assert record.values.shape == (62, 2000)
    assert not record.values.flags.writeable
    assert called == [["train_data"]]
    labels = io.validate_development_labels(config, data_root, record)
    assert called == [["train_data"], ["train_dg"]]
    assert labels["sampling_hz_glove_storage"] == 1000
    assert io.sha256_file(path) == before == record.metadata["sha256"]


def test_matlab_hdf5_explicit_adapter(config, data_root):
    path = config.path(data_root, "raw_dir") / "sub1_comp.mat"
    expected = np.arange(62 * 2000, dtype=float).reshape(2000, 62)
    with h5py.File(path, "w") as handle:
        data = handle.create_dataset("train_data", data=expected.T)
        data.attrs["MATLAB_class"] = np.bytes_("double")
        labels = handle.create_dataset("train_dg", data=np.ones((5, 50)))
        labels.attrs["MATLAB_class"] = np.bytes_("double")
        # An unresolved external test link must never be followed.
        handle["test_data"] = h5py.ExternalLink("does_not_exist_testlabels.mat", "test_dg")
    record = io.read_ecog(config, data_root, "S1")
    np.testing.assert_array_equal(record.values, expected.T)
    assert "reversed_axes" in record.metadata["adapter"]
    labels = io.validate_development_labels(config, data_root, record)
    assert labels["sampling_hz_glove_storage"] == 25


@pytest.mark.parametrize("key", ["test_data", "test_dg", "anything"])
def test_disallowed_variable_fails_before_file_access(tmp_path, key):
    with pytest.raises(PermissionError):
        io._read_variable(tmp_path / "nonexistent.mat", key)


def test_testlabel_filename_rejected_before_access(tmp_path):
    with pytest.raises(PermissionError):
        io._read_variable(tmp_path / "sub1_testlabels.mat", "train_data")


@pytest.mark.parametrize("matrix", [np.zeros((62, 2000)), np.zeros((62, 62)),
                                   np.zeros((2000, 61)), np.zeros((2, 3, 62)),
                                   np.full((2000, 62), np.nan), np.full((2000, 62), np.inf),
                                   np.ones((2000, 62), dtype=complex)])
def test_invalid_ecog_contract(config, data_root, matrix):
    write_subject(data_root, config, values=matrix)
    with pytest.raises(io.DataContractError):
        io.read_ecog(config, data_root, "S1")


def test_missing_development_key(config, data_root):
    path = config.path(data_root, "raw_dir") / "sub1_comp.mat"
    savemat(path, {"unrelated": np.ones((100, 62))})
    with pytest.raises(io.DataContractError, match="Missing development variable"):
        io.read_ecog(config, data_root, "S1")


@pytest.mark.parametrize("shape", [(2000, 4), (2000, 6), (73, 5)])
def test_invalid_labels_do_not_enter_qc(config, data_root, shape):
    write_subject(data_root, config, labels=np.ones(shape))
    record = io.read_ecog(config, data_root, "S1")
    with pytest.raises(io.DataContractError):
        io.validate_development_labels(config, data_root, record)


def test_source_change_between_ecog_and_labels(config, data_root):
    write_subject(data_root, config)
    record = io.read_ecog(config, data_root, "S1")
    write_subject(data_root, config, labels=np.ones((2000, 5)))
    with pytest.raises(io.DataContractError, match="Source changed"):
        io.validate_development_labels(config, data_root, record)


def test_generic_hdf5_requires_documented_adapter(config, data_root):
    path = config.path(data_root, "raw_dir") / "sub1_comp.mat"
    with h5py.File(path, "w") as handle:
        handle.create_dataset("train_data", data=np.ones((62, 2000)))
    with pytest.raises(io.DataContractError, match="MATLAB_class"):
        io.read_ecog(config, data_root, "S1")


def test_hdf5_development_cannot_redirect_to_external_labels(config, data_root):
    path = config.path(data_root, "raw_dir") / "sub1_comp.mat"
    with h5py.File(path, "w") as handle:
        handle["train_data"] = h5py.ExternalLink("sub1_testlabels.mat", "test_dg")
    with pytest.raises(io.DataContractError, match="must not follow"):
        io.read_ecog(config, data_root, "S1")
