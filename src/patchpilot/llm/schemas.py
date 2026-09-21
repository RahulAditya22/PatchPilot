from __future__ import annotations

from typing import Any

SEARCH_CODE_TOOL: dict[str, Any] = {
    "name": "search_code",
    "description": "Search the codebase for specific patterns or symbols.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query (symbol name, regex, or natural language depending on type).",
            },
            "search_type": {
                "type": "string",
                "enum": ["semantic", "lexical", "hybrid"],
                "description": "Type of search to perform.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return.",
                "default": 10,
            },
        },
        "required": ["query", "search_type"],
    },
}

READ_FILE_TOOL: dict[str, Any] = {
    "name": "read_file",
    "description": "Read the contents of a specific file.",
    "input_schema": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Absolute or relative path to the file to read.",
            }
        },
        "required": ["file_path"],
    },
}

MODIFY_FILE_TOOL: dict[str, Any] = {
    "name": "modify_file",
    "description": "Modify an existing file with new content.",
    "input_schema": {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Path to the file to modify."},
            "new_content": {"type": "string", "description": "The full new content for the file."},
            "change_description": {
                "type": "string",
                "description": "Brief description of the changes made.",
            },
        },
        "required": ["file_path", "new_content", "change_description"],
    },
}

CREATE_FILE_TOOL: dict[str, Any] = {
    "name": "create_file",
    "description": "Create a new file with specified content.",
    "input_schema": {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "Path where the new file should be created.",
            },
            "content": {"type": "string", "description": "The initial content of the new file."},
            "description": {
                "type": "string",
                "description": "Description of what this file is for.",
            },
        },
        "required": ["file_path", "content", "description"],
    },
}

RUN_TESTS_TOOL: dict[str, Any] = {
    "name": "run_tests",
    "description": "Run the test suite or specific tests.",
    "input_schema": {
        "type": "object",
        "properties": {
            "test_command": {
                "type": "string",
                "description": "The command to run tests (e.g., 'pytest tests/').",
            },
            "working_dir": {
                "type": "string",
                "description": "The directory to run the tests from.",
            },
        },
        "required": ["test_command", "working_dir"],
    },
}

LIST_FILES_TOOL: dict[str, Any] = {
    "name": "list_files",
    "description": "List files in a directory matching an optional pattern.",
    "input_schema": {
        "type": "object",
        "properties": {
            "directory": {"type": "string", "description": "The directory to list files from."},
            "pattern": {
                "type": "string",
                "description": "Glob pattern to filter files (e.g., '*.py').",
            },
        },
        "required": ["directory", "pattern"],
    },
}

ANALYZE_DEPENDENCIES_TOOL: dict[str, Any] = {
    "name": "analyze_dependencies",
    "description": "Analyze dependencies of a specific file.",
    "input_schema": {
        "type": "object",
        "properties": {
            "file_path": {"type": "string", "description": "Path to the file to analyze."}
        },
        "required": ["file_path"],
    },
}

# Groupings of tools for different agent roles
PLANNER_TOOLS: list[dict[str, Any]] = [
    SEARCH_CODE_TOOL,
    READ_FILE_TOOL,
    LIST_FILES_TOOL,
    ANALYZE_DEPENDENCIES_TOOL,
]

CODER_TOOLS: list[dict[str, Any]] = [
    SEARCH_CODE_TOOL,
    READ_FILE_TOOL,
    LIST_FILES_TOOL,
    MODIFY_FILE_TOOL,
    CREATE_FILE_TOOL,
    RUN_TESTS_TOOL,
]

ANALYST_TOOLS: list[dict[str, Any]] = [
    SEARCH_CODE_TOOL,
    READ_FILE_TOOL,
    LIST_FILES_TOOL,
    ANALYZE_DEPENDENCIES_TOOL,
    RUN_TESTS_TOOL,
]
