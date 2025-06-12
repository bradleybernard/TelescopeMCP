# Laravel Telescope MCP Server

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Laravel](https://img.shields.io/badge/Laravel-Compatible-red.svg)](https://laravel.com)
[![MCP](https://img.shields.io/badge/MCP-Compatible-green.svg)](https://modelcontextprotocol.io/)

A Model Context Protocol (MCP) server that provides AI agents with direct access to Laravel Telescope data, enabling efficient debugging and analysis of Laravel applications.

## Key Features

- **Natural Language Debugging**: Ask "Why is the checkout page slow?" instead of clicking through Telescope.
- **Instant Pattern Recognition**: AI can spot N+1 queries, memory leaks, and performance issues across thousands of requests.
- **Direct Database Access**: Query Laravel Telescope data without the web UI.
- **Advanced Search**: Filter requests by controller, status, duration, method, and URI.
- **AI-Optimized Output**: Cleaned JSON data with numeric values for easy analysis.
- **Laravel Sail Ready**: Works out of the box with Laravel Sail's default database.

## Quick Start

Get up and running in 3 simple steps:

### Step 1: Install `uv` (Python package manager)
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Step 2: Clone and install dependencies
```bash
git clone git@github.com:bradleybernard/TelescopeMCP.git
cd TelescopeMCP
uv sync
```

### Step 3: Configure your AI assistant

#### For Claude Code (Recommended)
Use the `claude mcp` command to connect the MCP server. This example uses the default Laravel Sail database credentials.

```bash
# Get the full path to uv and the current directory
UV_PATH=$(which uv)
MCP_DIR=$(pwd)

# Add the MCP server configuration
claude mcp add-json telescope '{
  "type": "stdio",
  "command": "'"$UV_PATH"'",
  "args": [
    "--directory",
    "'"$MCP_DIR"'",
    "run",
    "python",
    "telescope_mcp_server.py"
  ],
  "env": {
    "DB_URL": "mysql+pymysql://sail:password@127.0.0.1:3306/laravel"
  }
}'

# Restart Claude Code to apply changes
```
To reconfigure, run `claude mcp remove telescope` first.

#### For Claude Desktop
1.  Open `Claude` → `Settings` → `Developer` → `Edit Config`.
2.  Add the `telescope` server to `mcpServers`. Replace placeholders with your actual paths and database URL.

```json
{
  "mcpServers": {
    "telescope": {
      "type": "stdio",
      "command": "/path/to/uv",
      "args": [
        "--directory",
        "/path/to/TelescopeMCP",
        "run",
        "python",
        "telescope_mcp_server.py"
      ],
      "env": {
        "DB_URL": "mysql+pymysql://user:pass@host:port/db"
      }
    }
  }
}
```
*Tip: Find your `uv` path with `which uv` (macOS/Linux) or `where uv` (Windows).*

3.  Restart Claude Desktop.

## How It Works

```
Your Laravel App → Telescope → MySQL Database
                                    ↓
                              Telescope MCP
                                    ↓
                            AI Assistant
                                    ↓
                          Natural Language Debugging
```

The MCP server connects directly to your Laravel database and queries the `telescope_entries` table, providing your AI assistant with structured data optimized for debugging.

## Usage Examples

Once configured, you can ask your AI assistant to interact with your Laravel Telescope data.

**You**: "Show me recent telescope requests."

**AI**: Analyzes recent requests, pointing out a 500 error on the `/api/checkout` endpoint that took over 3 seconds. *Would you like to investigate?*

**You**: "Why was that request so slow?"

**AI**: Identifies an N+1 query problem, suggesting an eager loading fix that could improve response time by over 50%. It also finds a query that could benefit from a new database index.

**You**: "Show me all failed API requests in the last hour."

**AI**: Lists failed requests, summarizing them by error type (validation, not found, authorization) and highlighting endpoints with recurring issues.

## Available Tools

1.  **requests(page=1, search=None)**: List recent HTTP requests.
2.  **search_requests(...)**: Advanced search with filters (`controller`, `status`, `min_duration_ms`, `method`, `uri_pattern`).
3.  **get_request(batch_id)**: Get an overview of a specific request.
4.  **get_request_queries(batch_id, page=1)**: Get SQL queries for a request.
5.  **get_request_models(batch_id, page=1)**: Get model operations for a request.
6.  **get_request_response(batch_id)**: Get the full response data.
7.  **slow_queries()**: Find all database queries exceeding a time threshold.

## Configuration

The recommended way to configure your database connection is with the `DB_URL` environment variable in the MCP setup.

`DB_URL="mysql+pymysql://username:password@host:port/database"`

Example for Laravel Sail:
`DB_URL="mysql+pymysql://sail:password@127.0.0.1:3306/laravel"`

Alternatively, you can set individual environment variables: `DB_HOST`, `DB_PORT`, `DB_DATABASE`, `DB_USERNAME`, `DB_PASSWORD`. If `DB_URL` is set, it takes precedence.

## Development

-   **Run locally**: `uv run python telescope_mcp_server.py`
-   **Test with inspector**: `uv run mcp dev telescope_mcp_server.py`
-   **Dependencies**: See `pyproject.toml`.
-   **Test with inspector (and DB_URL in ENV)**: 
```bash
DB_URL="mysql+pymysql://sail:password@127.0.0.1:3306/laravel" uv run python telescope_mcp_server.py
```

## Troubleshooting

-   **Connection Errors**: Ensure your database is running and accessible. Check your credentials and that the `telescope_entries` table exists.
-   **No Data**: Verify Telescope is enabled and recording data in your Laravel app (`TELESCOPE_ENABLED=true`).

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built for [Laravel Telescope](https://laravel.com/docs/telescope)
- Uses the [Model Context Protocol](https://modelcontextprotocol.io/)
- Compatible with [Claude Desktop](https://claude.ai/download) and [Claude Code](https://claude.ai/code)