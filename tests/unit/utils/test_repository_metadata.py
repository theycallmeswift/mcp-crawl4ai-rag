"""
Unit tests for repository metadata functions.
"""

from src.utils.repository_metadata import (
    create_repository_source_id,
    create_documentation_url,
    create_documentation_metadata,
)


class TestCreateRepositorySourceId:
    """Tests for create_repository_source_id function."""

    def test_create_repository_source_id_github_https(self):
        """Test with standard GitHub HTTPS URL."""
        # Setup
        repo_url = "https://github.com/user/repo.git"

        # Exercise
        result = create_repository_source_id(repo_url)

        # Verify
        assert result == "github.com/user/repo"

    def test_create_repository_source_id_github_ssh(self):
        """Test with GitHub SSH URL."""
        # Setup
        repo_url = "git@github.com:user/repo.git"

        # Exercise
        result = create_repository_source_id(repo_url)

        # Verify
        assert result == "github.com/user/repo"

    def test_create_repository_source_id_removes_git_suffix(self):
        """Test that .git suffix is properly removed."""
        # Setup
        repo_url = "https://github.com/user/repo.git"

        # Exercise
        result = create_repository_source_id(repo_url)

        # Verify
        assert not result.endswith(".git")
        assert result == "github.com/user/repo"

    def test_create_repository_source_id_consistency_ssh_https(self):
        """Test that SSH and HTTPS URLs for the same repo generate the same source ID."""
        # Setup
        https_url = "https://github.com/user/repo.git"
        ssh_url = "git@github.com:user/repo.git"
        
        # Exercise
        https_result = create_repository_source_id(https_url)
        ssh_result = create_repository_source_id(ssh_url)
        
        # Verify
        assert https_result == ssh_result == "github.com/user/repo"

    def test_create_repository_source_id_handles_malformed_urls(self):
        """Test fallback behavior with invalid URLs."""
        # Setup
        malformed_url = "not-a-valid-url"

        # Exercise
        result = create_repository_source_id(malformed_url)

        # Verify
        assert result == "not-a-valid-url"

    def test_create_repository_source_id_other_git_hosts(self):
        """Test with GitLab, Bitbucket URLs."""
        # Setup & Exercise & Verify
        gitlab_url = "https://gitlab.com/user/repo.git"
        result = create_repository_source_id(gitlab_url)
        assert result == "gitlab.com/user/repo"

        bitbucket_url = "https://bitbucket.org/user/repo.git"
        result = create_repository_source_id(bitbucket_url)
        assert result == "bitbucket.org/user/repo"


class TestCreateDocumentationUrl:
    """Tests for create_documentation_url function."""

    def test_create_documentation_url_basic(self):
        """Test basic URL creation with repo URL and doc path."""
        # Setup
        repo_url = "https://github.com/user/repo.git"
        doc_path = "README.md"

        # Exercise
        result = create_documentation_url(repo_url, doc_path)

        # Verify
        assert result == "github.com/user/repo/README.md"

    def test_create_documentation_url_removes_git_suffix(self):
        """Test that .git suffix is handled correctly."""
        # Setup
        repo_url = "https://github.com/user/repo.git"
        doc_path = "docs/api.md"

        # Exercise
        result = create_documentation_url(repo_url, doc_path)

        # Verify
        assert result == "github.com/user/repo/docs/api.md"
        assert ".git" not in result

    def test_create_documentation_url_nested_paths(self):
        """Test with nested documentation paths."""
        # Setup
        repo_url = "https://github.com/user/repo.git"
        doc_path = "docs/api/reference.md"

        # Exercise
        result = create_documentation_url(repo_url, doc_path)

        # Verify
        assert result == "github.com/user/repo/docs/api/reference.md"

    def test_create_documentation_url_handles_malformed_urls(self):
        """Test fallback behavior with invalid URLs."""
        # Setup
        malformed_url = "not-a-valid-url"
        doc_path = "README.md"

        # Exercise
        result = create_documentation_url(malformed_url, doc_path)

        # Verify
        assert result == "not-a-valid-url/README.md"


class TestCreateDocumentationMetadata:
    """Tests for create_documentation_metadata function."""

    def test_create_documentation_metadata_readme_detection(self, repo_info):
        """Test that README files are categorized correctly."""
        # Setup
        doc_file_info = {"url": "README.md", "markdown": "# Test README"}

        # Exercise
        result = create_documentation_metadata(doc_file_info, repo_info)

        # Verify
        assert result["documentation_category"] == "readme"
        assert result["file_type"] == "md"
        assert result["repository_name"] == "test-repo"

    def test_create_documentation_metadata_api_detection(self, repo_info):
        """Test API documentation detection."""
        # Setup
        doc_file_info = {"url": "docs/api_reference.md", "markdown": "# API Reference"}

        # Exercise
        result = create_documentation_metadata(doc_file_info, repo_info)

        # Verify
        assert result["documentation_category"] == "api"

    def test_create_documentation_metadata_tutorial_detection(self, repo_info):
        """Test tutorial/guide detection."""
        # Setup
        doc_file_info = {
            "url": "docs/getting_started.md",
            "markdown": "# Getting Started",
        }

        # Exercise
        result = create_documentation_metadata(doc_file_info, repo_info)

        # Verify
        assert result["documentation_category"] == "tutorial"

    def test_create_documentation_metadata_changelog_detection(self, repo_info):
        """Test changelog detection."""
        # Setup
        doc_file_info = {"url": "CHANGELOG.md", "markdown": "# Changelog"}

        # Exercise
        result = create_documentation_metadata(doc_file_info, repo_info)

        # Verify
        assert result["documentation_category"] == "changelog"

    def test_create_documentation_metadata_license_detection(self, repo_info):
        """Test license file detection."""
        # Setup
        doc_file_info = {"url": "LICENSE.txt", "markdown": "MIT License"}

        # Exercise
        result = create_documentation_metadata(doc_file_info, repo_info)

        # Verify
        assert result["documentation_category"] == "license"

    def test_create_documentation_metadata_contributing_detection(self, repo_info):
        """Test contributing guide detection."""
        # Setup
        doc_file_info = {"url": "CONTRIBUTING.md", "markdown": "# Contributing"}

        # Exercise
        result = create_documentation_metadata(doc_file_info, repo_info)

        # Verify
        assert result["documentation_category"] == "contributing"

    def test_create_documentation_metadata_generic_documentation(self, repo_info):
        """Test fallback to generic 'documentation' category."""
        # Setup
        doc_file_info = {"url": "docs/other.md", "markdown": "# Other Documentation"}

        # Exercise
        result = create_documentation_metadata(doc_file_info, repo_info)

        # Verify
        assert result["documentation_category"] == "documentation"

    def test_create_documentation_metadata_code_example_counting(
        self, repo_info, mocker, monkeypatch
    ):
        """Test code example counting when USE_AGENTIC_RAG=true."""
        # Setup
        monkeypatch.setenv("USE_AGENTIC_RAG", "true")
        mock_extract_code_blocks = mocker.patch(
            "src.utils.code_extraction.extract_code_blocks", return_value=['print("hello")', "import os"]
        )
        doc_file_info = {
            "url": "docs/examples.md",
            "markdown": "# Examples\n\n```python\nprint('hello')\n```",
        }

        # Exercise
        result = create_documentation_metadata(doc_file_info, repo_info)

        # Verify
        assert result["code_example_count"] == 2  # Mocked to return 2 code blocks
        mock_extract_code_blocks.assert_called_once()