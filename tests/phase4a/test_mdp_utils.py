"""
4A-4: Verify MDP file read, write, patch, and template copy utilities.

Rationale:
    mdp_utils.py is the only code that modifies simulation parameters.
    A bug here silently corrupts the physics of the simulation.
    Every function needs thorough round-trip and edge-case testing.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
from agent.recovery.models import MDPPatch
from agent.utils.mdp_utils import apply_patches, copy_template, read_mdp, write_mdp


# ---------------------------------------------------------------------------
# read_mdp
# ---------------------------------------------------------------------------

class TestReadMdp:

    def test_reads_simple_key_value(self, tmp_path):
        mdp = tmp_path / "test.mdp"
        mdp.write_text(
            "integrator = steep\n"
            "nsteps     = 5000\n"
            "emtol      = 1000.0\n"
        )
        params = read_mdp(mdp)
        assert params["integrator"] == "steep"
        assert params["nsteps"]     == "5000"
        assert params["emtol"]      == "1000.0"

    def test_strips_inline_comments(self, tmp_path):
        mdp = tmp_path / "test.mdp"
        mdp.write_text(
            "integrator = steep   ; steepest descent\n"
            "nsteps     = 5000    ; max steps\n"
        )
        params = read_mdp(mdp)
        assert params["integrator"] == "steep"
        assert params["nsteps"]     == "5000"
        # Comment text must not appear in values
        assert "steepest" not in params["integrator"]
        assert "max"       not in params["nsteps"]

    def test_ignores_comment_only_lines(self, tmp_path):
        mdp = tmp_path / "test.mdp"
        mdp.write_text(
            "; This is a comment\n"
            "integrator = steep\n"
            "; Another comment\n"
            "nsteps = 1000\n"
        )
        params = read_mdp(mdp)
        assert len(params) == 2
        assert "integrator" in params
        assert "nsteps"     in params

    def test_keys_lowercased(self, tmp_path):
        mdp = tmp_path / "test.mdp"
        mdp.write_text("Integrator = steep\nNSTEPS = 5000\n")
        params = read_mdp(mdp)
        assert "integrator" in params
        assert "nsteps"     in params
        assert "Integrator" not in params
        assert "NSTEPS"     not in params

    def test_ignores_lines_without_equals(self, tmp_path):
        mdp = tmp_path / "test.mdp"
        mdp.write_text(
            "integrator = steep\n"
            "this line has no equals sign\n"
            "nsteps = 1000\n"
        )
        params = read_mdp(mdp)
        assert "this line has no equals sign" not in params
        assert len(params) == 2

    def test_handles_empty_file(self, tmp_path):
        mdp = tmp_path / "empty.mdp"
        mdp.write_text("")
        params = read_mdp(mdp)
        assert params == {}

    def test_handles_values_with_spaces(self, tmp_path):
        mdp = tmp_path / "test.mdp"
        mdp.write_text("tc-grps = Protein Non-Protein\n")
        params = read_mdp(mdp)
        assert params["tc-grps"] == "Protein Non-Protein"

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            read_mdp(tmp_path / "nonexistent.mdp")


# ---------------------------------------------------------------------------
# write_mdp
# ---------------------------------------------------------------------------

class TestWriteMdp:

    def test_writes_all_params(self, tmp_path):
        mdp = tmp_path / "out.mdp"
        params = {
            "integrator": "steep",
            "nsteps":     "5000",
            "emtol":      "1000.0",
        }
        write_mdp(params, mdp)
        assert mdp.exists()
        content = mdp.read_text()
        assert "integrator" in content
        assert "steep"      in content
        assert "nsteps"     in content
        assert "5000"       in content

    def test_round_trip(self, tmp_path):
        """write_mdp then read_mdp must recover all parameters."""
        mdp = tmp_path / "round_trip.mdp"
        original = {
            "integrator":  "md",
            "dt":          "0.002",
            "nsteps":      "50000",
            "coulombtype": "PME",
            "pbc":         "xyz",
        }
        write_mdp(original, mdp)
        recovered = read_mdp(mdp)
        for key, value in original.items():
            assert key in recovered, f"Key '{key}' lost in round-trip"
            assert recovered[key] == value, (
                f"Value mismatch for '{key}': "
                f"expected '{value}', got '{recovered[key]}'"
            )

    def test_creates_parent_directories(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c" / "test.mdp"
        write_mdp({"integrator": "steep"}, nested)
        assert nested.exists()

    def test_overwrites_existing_file(self, tmp_path):
        mdp = tmp_path / "test.mdp"
        write_mdp({"integrator": "steep"}, mdp)
        write_mdp({"integrator": "md"},    mdp)
        params = read_mdp(mdp)
        assert params["integrator"] == "md"
        assert len(params) == 1     # old content fully replaced


# ---------------------------------------------------------------------------
# apply_patches
# ---------------------------------------------------------------------------

class TestApplyPatches:

    def test_applies_single_patch(self, tmp_mdp_file):
        patches = [
            MDPPatch("emstep", "0.01", "0.001", "reduce step size"),
        ]
        applied = apply_patches(tmp_mdp_file, patches, backup=False)
        params  = read_mdp(tmp_mdp_file)

        assert params["emstep"] == "0.001"
        assert "emstep" in applied
        assert applied["emstep"] == ("0.01", "0.001")

    def test_applies_multiple_patches(self, tmp_mdp_file):
        patches = [
            MDPPatch("emstep", "0.01",  "0.001",  "reduce step"),
            MDPPatch("nsteps", "5000",  "10000",  "more steps"),
            MDPPatch("emtol",  "1000.0","500.0",  "tighter tolerance"),
        ]
        applied = apply_patches(tmp_mdp_file, patches, backup=False)
        params  = read_mdp(tmp_mdp_file)

        assert params["emstep"] == "0.001"
        assert params["nsteps"] == "10000"
        assert params["emtol"]  == "500.0"
        assert len(applied)     == 3

    def test_adds_new_parameter_not_in_original(self, tmp_mdp_file):
        """Patching a key not in the original file should add it."""
        patches = [
            MDPPatch("lincs-order", None, "6", "increase LINCS order"),
        ]
        apply_patches(tmp_mdp_file, patches, backup=False)
        params = read_mdp(tmp_mdp_file)
        assert params.get("lincs-order") == "6"

    def test_backup_created_when_requested(self, tmp_mdp_file):
        patches = [MDPPatch("emstep", "0.01", "0.001", "test")]
        apply_patches(tmp_mdp_file, patches, backup=True)

        # At least one .bak file should exist in the same directory
        bak_files = list(tmp_mdp_file.parent.glob("*.bak*"))
        assert len(bak_files) >= 1, "Backup file was not created"

    def test_backup_not_created_when_disabled(self, tmp_mdp_file):
        patches = [MDPPatch("emstep", "0.01", "0.001", "test")]
        apply_patches(tmp_mdp_file, patches, backup=False)

        bak_files = list(tmp_mdp_file.parent.glob("*.bak*"))
        assert len(bak_files) == 0, "Backup file created when backup=False"

    def test_backup_preserves_original_content(self, tmp_mdp_file):
        original_content = tmp_mdp_file.read_text()
        patches = [MDPPatch("emstep", "0.01", "0.001", "test")]
        apply_patches(tmp_mdp_file, patches, backup=True)

        bak_files = list(tmp_mdp_file.parent.glob("*.bak*"))
        assert bak_files, "No backup file found"
        backup_content = bak_files[0].read_text()
        assert backup_content == original_content

    def test_empty_patches_list_leaves_file_unchanged(self, tmp_mdp_file):
        original = read_mdp(tmp_mdp_file)
        apply_patches(tmp_mdp_file, [], backup=False)
        after = read_mdp(tmp_mdp_file)
        assert original == after

    def test_patch_key_case_insensitive(self, tmp_mdp_file):
        """Patch parameter names should match regardless of case."""
        patches = [
            MDPPatch("EMSTEP", "0.01", "0.001", "uppercase key test"),
        ]
        apply_patches(tmp_mdp_file, patches, backup=False)
        params = read_mdp(tmp_mdp_file)
        assert params.get("emstep") == "0.001"


# ---------------------------------------------------------------------------
# copy_template
# ---------------------------------------------------------------------------

class TestCopyTemplate:

    @pytest.fixture
    def template_dir(self, tmp_path) -> Path:
        """Create a minimal fake template directory."""
        tdir = tmp_path / "mdp_templates"
        tdir.mkdir()
        (tdir / "em.mdp").write_text(
            "integrator = steep\nnsteps = 5000\n"
        )
        (tdir / "nvt.mdp").write_text(
            "integrator = md\nnsteps = 50000\n"
        )
        return tdir

    def test_copies_existing_template(self, tmp_path, template_dir):
        dest = tmp_path / "gmx_run" / "em.mdp"
        result = copy_template("em.mdp", dest, templates_dir=template_dir)
        assert result == dest
        assert dest.exists()
        content = read_mdp(dest)
        assert content["integrator"] == "steep"

    def test_copied_file_is_independent(self, tmp_path, template_dir):
        """Modifying the copy must not affect the template."""
        dest = tmp_path / "em_copy.mdp"
        copy_template("em.mdp", dest, templates_dir=template_dir)

        # Modify the copy
        apply_patches(dest, [MDPPatch("nsteps", "5000", "99999", "test")],
                      backup=False)

        # Original template must be unchanged
        original = read_mdp(template_dir / "em.mdp")
        assert original["nsteps"] == "5000"

    def test_missing_template_raises_file_not_found(self, tmp_path, template_dir):
        with pytest.raises(FileNotFoundError, match="not found"):
            copy_template(
                "nonexistent.mdp",
                tmp_path / "out.mdp",
                templates_dir=template_dir,
            )

    def test_creates_destination_parent_dirs(self, tmp_path, template_dir):
        dest = tmp_path / "deep" / "nested" / "dir" / "em.mdp"
        copy_template("em.mdp", dest, templates_dir=template_dir)
        assert dest.exists()