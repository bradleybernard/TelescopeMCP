# Laravel Telescope MCP Server

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

Navigate to your **Laravel project's root directory** to perform this step, not the `TelescopeMCP` directory.

#### For Claude Code (Recommended)

Use the `claude mcp` command to connect the MCP server. This will create or update a `.claude/mcp.json` file in your Laravel project, telling your AI assistant how to communicate with the Telescope MCP server.

Run the following command after replacing the placeholder paths with the correct absolute paths for your system. This example uses database credentials for a default Laravel Sail setup.

```bash
# From your Laravel project root directory:
claude mcp add-json telescope '{
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
    "DB_URL": "mysql+pymysql://sail:password@127.0.0.1:3306/laravel",
    "LARAVEL_PROJECT_PATH": "/path/to/laravel-project"
  }
}'
```

-   **`/path/to/uv`**: Replace with the absolute path to the `uv` executable. Find it by running `which uv`.
-   **`/path/to/TelescopeMCP`**: Replace with the absolute path to the `TelescopeMCP` directory you cloned in Step 2.
-   **`/path/to/laravel-project`**: Replace with the absolute path to your Laravel project root directory.

After running the command, restart Claude Code to apply the changes. Once you boot up `claude` you should be able to run `/mcp` to see `telescope` connected and ready to use.

To reconfigure (ie: change the database URL or Laravel project path), run `claude mcp remove telescope` first from your Laravel project directory, then re-run the `claude mcp add-json telescope` command above.

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
        "DB_URL": "mysql+pymysql://user:pass@host:port/db",
        "LARAVEL_PROJECT_PATH": "/path/to/your/laravel/project"
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
6.  **get_request_jobs(batch_id, page=1)**: Get jobs for a request.
7.  **get_request_views(batch_id, page=1)**: Get view data for a request.
8.  **get_request_cache(batch_id, page=1)**: Get cache events for a request.
9.  **get_request_redis(batch_id, page=1)**: Get Redis commands for a request.
10. **get_request_notifications(batch_id, page=1)**: Get notifications for a request.
11. **get_request_timing(batch_id)**: Get timing breakdown from Laravel Debugbar.
12. **get_request_response(batch_id)**: Get the full response data.
13. **get_request_payload(batch_id)**: Get the request payload (form data or JSON body).
14. **slow_queries()**: Find all database queries exceeding a time threshold.

## Configuration

The server is configured through environment variables passed in your MCP setup.

### Database Connection

The recommended way to configure your database connection is with the `DB_URL` environment variable.

`DB_URL="mysql+pymysql://username:password@host:port/database"`

Example for Laravel Sail:
`DB_URL="mysql+pymysql://sail:password@127.0.0.1:3306/laravel"`

Alternatively, you can set individual environment variables: `DB_HOST`, `DB_PORT`, `DB_DATABASE`, `DB_USERNAME`, `DB_PASSWORD`. If `DB_URL` is set, it takes precedence.

### Laravel Project Path (for Debugbar)

To use the `get_request_timing` tool, you must provide the absolute path to your Laravel project root directory via the `LARAVEL_PROJECT_PATH` environment variable. The server uses this path to locate Laravel Debugbar's JSON storage files.

**Example:**
`LARAVEL_PROJECT_PATH="/Users/me/Code/my-laravel-app"`

Add this to your `env` configuration in your MCP setup:

```json
"env": {
  "DB_URL": "mysql+pymysql://sail:password@127.0.0.1:3306/laravel",
  "LARAVEL_PROJECT_PATH": "/path/to/your/laravel/project"
}
```

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