"""Laravel Telescope MCP server for exploring application requests and debugging data.

This server provides direct database access to Laravel Telescope data, enabling
AI agents to query debugging information from Laravel applications without accessing
the Telescope web UI.

The server connects to a MySQL database containing telescope_entries table and provides
tools for analyzing HTTP requests, database queries, model operations, and other
application debugging data.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from mcp.server.fastmcp import FastMCP

# Version
__version__ = "0.1.0"

################################################################################
# Constants                                                                    #
################################################################################

# Pagination settings
DEFAULT_PAGE_SIZE = 10
MAX_PAGE_SIZE = 100

# Performance thresholds
SLOW_QUERY_THRESHOLD_MS = 100
HIGH_QUERY_COUNT_THRESHOLD = 20
SLOW_REQUEST_THRESHOLD_MS = 500

# Database connection
DEFAULT_DB_HOST = "127.0.0.1"
DEFAULT_DB_PORT = "3306"
DEFAULT_DB_NAME = "laravel"
DEFAULT_DB_USER = "sail"
DEFAULT_DB_PASS = "password"

################################################################################
# Globals                                                                      #
################################################################################

LARAVEL_PROJECT_PATH = os.getenv("LARAVEL_PROJECT_PATH")

DB_URL = os.getenv(
    "DB_URL",
    "mysql+pymysql://{user}:{pwd}@{host}:{port}/{db}".format(
        user=os.getenv("DB_USERNAME", DEFAULT_DB_USER),
        pwd=os.getenv("DB_PASSWORD", DEFAULT_DB_PASS),
        host=os.getenv("DB_HOST", DEFAULT_DB_HOST),
        port=os.getenv("DB_PORT", DEFAULT_DB_PORT),
        db=os.getenv("DB_DATABASE", DEFAULT_DB_NAME),
    ),
)

try:
    ENGINE: Engine = create_engine(DB_URL, pool_pre_ping=True, pool_recycle=3600)
    # Test the connection immediately
    with ENGINE.connect() as conn:
        conn.execute(text("SELECT 1"))
    import sys
    sys.stderr.write(f"Database connection successful to: {DB_URL.split('@')[1] if '@' in DB_URL else DB_URL}\n")
except Exception as e:
    import sys
    sys.stderr.write(f"Database connection error: {e}\n")
    sys.stderr.write(f"Attempted URL: {DB_URL}\n")
    sys.exit(1)

################################################################################
# Server                                                                       #
################################################################################

mcp = FastMCP("Laravel Telescope Explorer", dependencies=["sqlalchemy", "pymysql"])

################################################################################
# Helpers                                                                      #
################################################################################

def paginate(items: List[Any], page: int = 1, limit: int = DEFAULT_PAGE_SIZE) -> Dict[str, Any]:
    """Paginate a list of items with metadata."""
    import math
    
    total_items = len(items)
    total_pages = math.ceil(total_items / limit) if total_items > 0 else 0
    
    # Ensure page is valid
    page = max(1, min(page, total_pages)) if total_pages > 0 else 1
    
    start = (page - 1) * limit
    end = start + limit
    
    return {
        "items": items[start:end],
        "pagination": {
            "page": page,
            "limit": limit,
            "total_pages": total_pages,
            "total_items": total_items,
            "has_more": end < total_items,
            "has_previous": page > 1
        }
    }


def analyze_queries(queries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze queries for patterns and potential issues."""
    import re
    from collections import defaultdict
    
    analysis = {
        "total_queries": len(queries),
        "slow_queries": 0,
        "duplicate_queries": 0,
        "n_plus_one_candidates": [],
        "tables_accessed": set(),
        "suggestions": []
    }
    
    # Track query patterns
    query_counts = defaultdict(int)
    table_pattern = re.compile(r'from\s+[`"]?(\w+)[`"]?', re.IGNORECASE)
    
    for query in queries:
        sql = query.get("sql", "").lower().strip()
        
        # Count slow queries
        if query.get("time_ms", 0) > SLOW_QUERY_THRESHOLD_MS:
            analysis["slow_queries"] += 1
        
        # Track duplicates
        query_counts[sql] += 1
        
        # Extract table names
        tables = table_pattern.findall(sql)
        analysis["tables_accessed"].update(tables)
        
    # Identify duplicates
    for sql, count in query_counts.items():
        if count > 1:
            analysis["duplicate_queries"] += count - 1
    
    # Detect N+1 patterns (multiple similar queries)
    for sql, count in query_counts.items():
        if count > 3 and "where" in sql and "id" in sql:
            # Extract table name for N+1 candidate
            tables = table_pattern.findall(sql)
            if tables:
                analysis["n_plus_one_candidates"].extend(tables)
    
    # Convert sets to lists for JSON serialization
    analysis["tables_accessed"] = list(analysis["tables_accessed"])
    analysis["n_plus_one_candidates"] = list(set(analysis["n_plus_one_candidates"]))
    
    # Generate suggestions
    if analysis["slow_queries"] > 0:
        analysis["suggestions"].append(f"Found {analysis['slow_queries']} slow queries (>{SLOW_QUERY_THRESHOLD_MS}ms). Consider adding indexes.")
    
    if analysis["duplicate_queries"] > 0:
        analysis["suggestions"].append(f"Found {analysis['duplicate_queries']} duplicate queries. Consider caching or query optimization.")
    
    if analysis["n_plus_one_candidates"]:
        analysis["suggestions"].append(f"Possible N+1 query pattern detected for tables: {', '.join(analysis['n_plus_one_candidates'])}. Consider eager loading.")
    
    if len(queries) > 50:
        analysis["suggestions"].append("High query count detected. Review data loading strategy.")
    
    return analysis


def clean_file_path(file_path: str) -> str:
    """Remove common prefixes from file paths for readability."""
    if file_path and file_path.startswith('/var/www/html/'):
        return file_path[14:]  # Remove prefix
    return file_path


def fix_escaped_json(obj: dict) -> dict:
    """Fix escaped quotes in JSON content."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            if isinstance(value, str) and value.startswith('"') and value.endswith('"'):
                try:
                    # Try to parse as JSON to remove extra quotes
                    obj[key] = json.loads(value)
                except:
                    pass
    return obj


def _format_request_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """Format a raw database row for a request into a consistent dictionary."""
    from datetime import datetime

    created = row["created_at"]
    if isinstance(created, str):
        created = datetime.fromisoformat(created.replace(" ", "T"))

    request_data = {
        "batch_id": row["batch_id"],
        "method": row.get("method") or "GET",
        "uri": row.get("uri") or "/",
        "status": int(row["status"]) if row["status"] else 200,
        "duration_ms": float(row["duration"]) if row["duration"] else 0,
        "timestamp": created.isoformat(),
    }

    # Optional fields
    if row.get("controller_action"):
        request_data["controller_action"] = row["controller_action"]

    return request_data


################################################################################
# SQL helpers                                                                  #
################################################################################

REQUEST_COMMON_SELECT_FIELDS = """
    batch_id,
    JSON_UNQUOTE(JSON_EXTRACT(content,'$.method')) as method,
    JSON_UNQUOTE(JSON_EXTRACT(content,'$.uri')) as uri,
    JSON_UNQUOTE(JSON_EXTRACT(content,'$.response_status')) as status,
    CAST(JSON_EXTRACT(content,'$.duration') AS DECIMAL(10,2)) as duration,
    created_at
"""

def _fetch(engine: Engine, sql: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """Execute a SQL query and return results as list of dictionaries."""
    with engine.connect() as conn:
        res = conn.execute(text(sql), params or {})
        cols = res.keys()
        return [dict(zip(cols, row)) for row in res]


################################################################################
# Tool                                                                         #
################################################################################

@mcp.tool(description="Find slowest queries in recent requests")
def slow_queries(time_threshold_ms: float = SLOW_QUERY_THRESHOLD_MS, limit: int = 20) -> dict:
    """Find database queries that exceed a time threshold. """
    try:
        sql = (
            "SELECT batch_id, "
            "JSON_UNQUOTE(JSON_EXTRACT(content,'$.sql')) as query, "
            "CAST(JSON_EXTRACT(content,'$.time') AS DECIMAL(10,2)) as time_ms, "
            "JSON_UNQUOTE(JSON_EXTRACT(content,'$.file')) as file, "
            "JSON_EXTRACT(content,'$.line') as line, "
            "created_at FROM telescope_entries "
            "WHERE type='query' AND CAST(JSON_EXTRACT(content,'$.time') AS DECIMAL(10,2)) > :threshold "
            "ORDER BY CAST(JSON_EXTRACT(content,'$.time') AS DECIMAL(10,2)) DESC "
            "LIMIT :limit"
        )
        rows = _fetch(ENGINE, sql, {"threshold": time_threshold_ms, "limit": limit})
        
        # Process results
        queries = []
        for row in rows:
            file_path = clean_file_path(row.get('file', ''))
                
            queries.append({
                "batch_id": row["batch_id"],
                "query": row["query"],
                "time_ms": float(row["time_ms"]),
                "file": file_path or None,
                "line": row.get("line"),
                "timestamp": str(row["created_at"])
            })
                    
        return {
            "slow_queries": queries,
            "threshold_ms": time_threshold_ms,
            "help": "Use get_request(batch_id) to see the full request context for any slow query."
        }
    except Exception as e:
        return {"error": f"Database query failed: {str(e)}"}


@mcp.tool(description="Search requests by multiple criteria. More powerful than basic requests().")
def search_requests(
    controller: Optional[str] = None,
    status: Optional[int] = None,
    min_duration_ms: Optional[float] = None,
    method: Optional[str] = None,
    uri_pattern: Optional[str] = None,
    page: int = 1
) -> dict:
    """Search requests with multiple filters."""
    try:
        # Build dynamic WHERE clause
        conditions = ["type='request'"]
        params = {}
        
        if controller:
            conditions.append("JSON_EXTRACT(content,'$.controller_action') LIKE :controller")
            params["controller"] = f"%{controller}%"
            
        if status is not None:
            conditions.append("JSON_EXTRACT(content,'$.response_status') = :status")
            params["status"] = str(status)
            
        if min_duration_ms is not None:
            conditions.append("CAST(JSON_EXTRACT(content,'$.duration') AS DECIMAL(10,2)) >= :min_duration")
            params["min_duration"] = min_duration_ms
            
        if method:
            conditions.append("JSON_EXTRACT(content,'$.method') = :method")
            params["method"] = method.upper()
            
        if uri_pattern:
            conditions.append("JSON_EXTRACT(content,'$.uri') LIKE :uri")
            params["uri"] = f"%{uri_pattern}%"
        
        where_clause = " AND ".join(conditions)
        
        sql = f"""
            SELECT
                {REQUEST_COMMON_SELECT_FIELDS},
                uuid,
                JSON_UNQUOTE(JSON_EXTRACT(content,'$.controller_action')) as controller_action
            FROM telescope_entries
            WHERE {where_clause}
            ORDER BY created_at DESC
        """
        
        rows = _fetch(ENGINE, sql, params)
        
        # Format results
        requests_list = [_format_request_row(row) for row in rows]
        
        # Paginate
        paginated = paginate(requests_list, page=page, limit=DEFAULT_PAGE_SIZE)
        
        result = {
            "requests": paginated["items"],
            "pagination": paginated["pagination"],
            "filters_applied": {
                "controller": controller,
                "status": status,
                "min_duration_ms": min_duration_ms,
                "method": method,
                "uri_pattern": uri_pattern
            },
            "help": "Use get_request(batch_id) for detailed analysis."
        }
        
        # Clean up filters_applied to show only what was used
        result["filters_applied"] = {k: v for k, v in result["filters_applied"].items() if v is not None}
        
        return result
    except Exception as e:
        return {"error": f"Search failed: {str(e)}"}


@mcp.tool(description="Show recent HTTP requests with pagination. Returns 10 per page.")
def requests(page: int = 1, search: Optional[str] = None) -> dict:
    """Show recent HTTP requests with pagination."""
    try:
        conditions = ["type='request'"]
        params = {}
        if search:
            conditions.append(
                "(JSON_EXTRACT(content,'$.uri') LIKE :search OR "
                "JSON_EXTRACT(content,'$.method') LIKE :search)"
            )
            params["search"] = f"%{search}%"

        where_clause = " AND ".join(conditions)

        sql = f"""
            SELECT {REQUEST_COMMON_SELECT_FIELDS}, uuid
            FROM telescope_entries
            WHERE {where_clause}
            ORDER BY created_at DESC
        """
        rows = _fetch(ENGINE, sql, params)

        # Format the requests for display
        requests_list = [_format_request_row(row) for row in rows]
        
        # Paginate the results
        paginated = paginate(requests_list, page=page, limit=DEFAULT_PAGE_SIZE)
        
        result = {
            "requests": paginated["items"],
            "pagination": paginated["pagination"],
            "help": "Use get_request(batch_id) for overview. Use page param to navigate results."
        }
        
        if search:
            result["search"] = search
            
        return result
    except Exception as e:
        return {"error": f"Database query failed: {str(e)}"}


@mcp.tool(description="Get request overview with entry counts. Use specific tools for detailed data.")
def get_request(batch_id: str) -> dict:
    """Get request overview and entry counts."""
    try:
        # Get the request entry first
        request_sql = (
            "SELECT content, created_at FROM telescope_entries "
            "WHERE batch_id = :bid AND type = 'request' LIMIT 1"
        )
        request_rows = _fetch(ENGINE, request_sql, {"bid": batch_id})
        
        if not request_rows:
            return {"error": f"No request found for batch_id: {batch_id}"}
        
        request_content = fix_escaped_json(json.loads(request_rows[0]["content"]))
        
        # Get entry counts for this batch
        count_sql = (
            "SELECT type, COUNT(*) as count FROM telescope_entries "
            "WHERE batch_id = :bid GROUP BY type ORDER BY count DESC"
        )
        counts = _fetch(ENGINE, count_sql, {"bid": batch_id})
        entry_counts = {row["type"]: row["count"] for row in counts}
        
        # Calculate performance breakdown if we have query data
        performance_breakdown = {}
        suggestions = []
        
        if "query" in entry_counts:
            # Get query timing data
            query_sql = (
                "SELECT SUM(CAST(JSON_EXTRACT(content, '$.time') AS DECIMAL(10,2))) as total_time "
                "FROM telescope_entries WHERE batch_id = :bid AND type = 'query'"
            )
            query_time = _fetch(ENGINE, query_sql, {"bid": batch_id})
            if query_time and query_time[0]["total_time"]:
                db_time = float(query_time[0]["total_time"])
                performance_breakdown["database_ms"] = round(db_time, 2)
                performance_breakdown["database_queries"] = entry_counts["query"]
                
                # Add suggestion if high query count
                if entry_counts["query"] > HIGH_QUERY_COUNT_THRESHOLD:
                    suggestions.append(f"High query count ({entry_counts['query']} queries). Use get_request_queries() to analyze.")
        
        # Add Redis time if present
        if "redis" in entry_counts:
            performance_breakdown["redis_commands"] = entry_counts["redis"]
        
        # Calculate total time and other metrics
        total_duration = float(request_content.get("duration", 0))
        if total_duration > 0:
            performance_breakdown["total_ms"] = total_duration
            if "database_ms" in performance_breakdown:
                performance_breakdown["other_ms"] = round(total_duration - performance_breakdown["database_ms"], 2)
        
        # Add performance suggestions
        if total_duration > SLOW_REQUEST_THRESHOLD_MS:
            suggestions.append(f"Slow request ({total_duration}ms). Consider performance optimization.")
        
        # Build the response with just overview data
        result = {
            "batch_id": batch_id,
            "request": {
                "method": request_content.get("method"),
                "uri": request_content.get("uri"),
                "controller_action": request_content.get("controller_action"),
                "middleware": request_content.get("middleware", []),
                "response_status": request_content.get("response_status"),
                "duration_ms": float(request_content.get("duration", 0)),
                "memory_mb": float(request_content.get("memory", 0)),
                "user": {
                    "id": request_content.get("user", {}).get("id"),
                    "name": request_content.get("user", {}).get("name")
                } if request_content.get("user") else None,
                "created_at": str(request_rows[0]["created_at"])
            },
            "entry_counts": entry_counts,
            "performance_breakdown": performance_breakdown,
            "suggestions": suggestions,
            "available_tools": {
                "queries": "get_request_queries(batch_id, page=1)" if "query" in entry_counts else None,
                "models": "get_request_models(batch_id, page=1)" if "model" in entry_counts else None,
                "views": "get_request_views(batch_id, page=1)" if "view" in entry_counts else None,
                "notifications": "get_request_notifications(batch_id, page=1)" if "notification" in entry_counts else None,
                "cache": "get_request_cache(batch_id, page=1)" if "cache" in entry_counts else None,
                "redis": "get_request_redis(batch_id, page=1)" if "redis" in entry_counts else None,
                "timing": "get_request_timing(batch_id)" if "debugbar" in entry_counts else None,
                "response": "get_request_response(batch_id)",
                # "headers": "get_request_headers(batch_id)",
                # "payload": "get_request_payload(batch_id)",
                # "session": "get_request_session(batch_id)"
            }
        }
        
        # Remove None tools
        result["available_tools"] = {k: v for k, v in result["available_tools"].items() if v is not None}
        
        return result
    except Exception as e:
        return {"error": f"Failed to fetch request details: {str(e)}"}

@mcp.tool(description="Get paginated models for a request. Shows 10 model operations per page.")
def get_request_models(batch_id: str, page: int = 1) -> dict:
    """Get model operations for a request with pagination."""
    try:
        # Get all model operations for this batch
        sql = (
            "SELECT content, created_at FROM telescope_entries "
            "WHERE batch_id = :bid AND type = 'model' "
            "ORDER BY sequence"
        )
        rows = _fetch(ENGINE, sql, {"bid": batch_id})
        
        if not rows:
            return {
                "batch_id": batch_id,
                "type": "models",
                "items": [],
                "pagination": {
                    "page": 1,
                    "limit": 10,
                    "total_pages": 0,
                    "total_items": 0,
                    "has_more": False,
                    "has_previous": False
                }
            }
        
        # Process models
        models = []
        for row in rows:
            content = fix_escaped_json(json.loads(row["content"]))
            models.append({
                "action": content.get("action"),
                "model": content.get("model"),
                "count": content.get("count", 1),
                "timestamp": str(row["created_at"])
            })
        
        # Paginate
        paginated = paginate(models, page=page, limit=DEFAULT_PAGE_SIZE)
        
        # Calculate summary stats
        model_stats = {}
        for model in models:
            model_name = model["model"]
            action = model["action"]
            count = model["count"]
            
            if model_name not in model_stats:
                model_stats[model_name] = {"retrieved": 0, "created": 0, "updated": 0, "deleted": 0}
            
            if action in model_stats[model_name]:
                model_stats[model_name][action] += count
        
        return {
            "batch_id": batch_id,
            "type": "models", 
            "summary": model_stats,
            "items": paginated["items"],
            "pagination": paginated["pagination"]
        }
    except Exception as e:
        return {"error": f"Failed to fetch models: {str(e)}"}


@mcp.tool(description="Get paginated views for a request. Shows 10 per page.")
def get_request_views(batch_id: str, page: int = 1) -> dict:
    """Get view data for a request with pagination."""
    try:
        sql = (
            "SELECT content, created_at FROM telescope_entries "
            "WHERE batch_id = :bid AND type = 'view' ORDER BY sequence"
        )
        rows = _fetch(ENGINE, sql, {"bid": batch_id})

        views = []
        for row in rows:
            content = fix_escaped_json(json.loads(row["content"]))

            composers = []
            for composer in content.get("composers", []):
                name = composer.get("name", "")
                if "/var/www/html/" in name:
                    name = name.replace("/var/www/html/", "")
                composers.append({"name": name, "type": composer.get("type")})

            views.append(
                {
                    "name": content.get("name"),
                    "path": clean_file_path(content.get("path", "")),
                    "data": content.get("data", []),
                    "composers": composers,
                    "timestamp": str(row["created_at"]),
                }
            )

        paginated = paginate(views, page=page, limit=DEFAULT_PAGE_SIZE)

        return {
            "batch_id": batch_id,
            "type": "views",
            "items": paginated["items"],
            "pagination": paginated["pagination"],
        }
    except Exception as e:
        return {"error": f"Failed to fetch views: {str(e)}"}


@mcp.tool(description="Get paginated cache events for a request. Shows 10 per page.")
def get_request_cache(batch_id: str, page: int = 1) -> dict:
    """Get cache operations for a request with pagination."""
    try:
        sql = (
            "SELECT content, created_at FROM telescope_entries "
            "WHERE batch_id = :bid AND type = 'cache' ORDER BY sequence"
        )
        rows = _fetch(ENGINE, sql, {"bid": batch_id})

        cache_events = []
        for row in rows:
            content = fix_escaped_json(json.loads(row["content"]))
            cache_events.append(
                {
                    "type": content.get("type"),  # hit, miss, set, forget
                    "key": content.get("key"),
                    "value": content.get("value"),
                    "timestamp": str(row["created_at"]),
                }
            )

        paginated = paginate(cache_events, page=page, limit=DEFAULT_PAGE_SIZE)

        return {
            "batch_id": batch_id,
            "type": "cache",
            "items": paginated["items"],
            "pagination": paginated["pagination"],
        }
    except Exception as e:
        return {"error": f"Failed to fetch cache events: {str(e)}"}


@mcp.tool(description="Get paginated Redis commands for a request. Shows 10 per page.")
def get_request_redis(batch_id: str, page: int = 1) -> dict:
    """Get Redis commands for a request with pagination."""
    try:
        sql = (
            "SELECT content, created_at FROM telescope_entries "
            "WHERE batch_id = :bid AND type = 'redis' ORDER BY sequence"
        )
        rows = _fetch(ENGINE, sql, {"bid": batch_id})

        redis_commands = []
        total_time = 0
        for row in rows:
            content = fix_escaped_json(json.loads(row["content"]))
            command_time = float(content.get("time", 0))
            total_time += command_time
            redis_commands.append(
                {
                    "connection": content.get("connection"),
                    "command": content.get("command"),
                    "time_ms": command_time,
                    "timestamp": str(row["created_at"]),
                }
            )

        paginated = paginate(redis_commands, page=page, limit=DEFAULT_PAGE_SIZE)

        return {
            "batch_id": batch_id,
            "type": "redis",
            "total_redis_time_ms": round(total_time, 2),
            "items": paginated["items"],
            "pagination": paginated["pagination"],
        }
    except Exception as e:
        return {"error": f"Failed to fetch redis commands: {str(e)}"}


@mcp.tool(description="Get paginated notifications for a request. Shows 10 per page.")
def get_request_notifications(batch_id: str, page: int = 1) -> dict:
    """Get notifications for a request with pagination."""
    try:
        sql = (
            "SELECT content, created_at FROM telescope_entries "
            "WHERE batch_id = :bid AND type = 'notification' ORDER BY sequence"
        )
        rows = _fetch(ENGINE, sql, {"bid": batch_id})

        notifications = []
        for row in rows:
            content = fix_escaped_json(json.loads(row["content"]))
            notifications.append(
                {
                    "notification": content.get("notification"),
                    "queued": content.get("queued"),
                    "notifiable": content.get("notifiable"),
                    "channel": content.get("channel"),
                    "response": content.get("response"),
                    "timestamp": str(row["created_at"]),
                }
            )

        paginated = paginate(notifications, page=page, limit=DEFAULT_PAGE_SIZE)

        return {
            "batch_id": batch_id,
            "type": "notifications",
            "items": paginated["items"],
            "pagination": paginated["pagination"],
        }
    except Exception as e:
        return {"error": f"Failed to fetch notifications: {str(e)}"}


@mcp.tool(description="Get paginated queries for a request. Shows 10 queries per page with full SQL.")
def get_request_queries(batch_id: str, page: int = 1) -> dict:
    """Get database queries for a request with pagination."""
    try:
        # Get all queries for this batch
        sql = (
            "SELECT content, created_at FROM telescope_entries "
            "WHERE batch_id = :bid AND type = 'query' "
            "ORDER BY sequence"
        )
        rows = _fetch(ENGINE, sql, {"bid": batch_id})
        
        if not rows:
            return {
                "batch_id": batch_id,
                "type": "queries",
                "items": [],
                "pagination": {
                    "page": 1,
                    "limit": 10,
                    "total_pages": 0,
                    "total_items": 0,
                    "has_more": False,
                    "has_previous": False
                }
            }
        
        # Process queries
        queries = []
        total_time = 0
        for row in rows:
            content = fix_escaped_json(json.loads(row["content"]))
            query_time = float(content.get("time", 0))
            total_time += query_time
            
            queries.append({
                "sql": content.get("sql", "").strip(),
                "time_ms": query_time,
                "file": clean_file_path(content.get("file", "")),
                "line": content.get("line"),
                "timestamp": str(row["created_at"])
            })
        
        # Paginate
        paginated = paginate(queries, page=page, limit=DEFAULT_PAGE_SIZE)
        
        # Analyze queries for patterns
        analysis = analyze_queries(queries)
        
        return {
            "batch_id": batch_id,
            "type": "queries",
            "total_query_time_ms": round(total_time, 2),
            "items": paginated["items"],
            "pagination": paginated["pagination"],
            "analysis": analysis
        }
    except Exception as e:
        return {"error": f"Failed to fetch queries: {str(e)}"}


@mcp.tool(description="Get the full response data for a request. Not paginated - returns complete response.")
def get_request_response(batch_id: str) -> dict:
    """Get the full response data for a request."""
    try:
        # Get the request entry
        sql = (
            "SELECT content FROM telescope_entries "
            "WHERE batch_id = :bid AND type = 'request' LIMIT 1"
        )
        rows = _fetch(ENGINE, sql, {"bid": batch_id})
        
        if not rows:
            return {"error": f"No request found for batch_id: {batch_id}"}
        
        content = fix_escaped_json(json.loads(rows[0]["content"]))
        response = content.get("response", {})
        
        return {
            "batch_id": batch_id,
            "type": "response",
            "response": response,
            "response_status": content.get("response_status"),
            "response_headers": content.get("response_headers", {})
        }
    except Exception as e:
        return {"error": f"Failed to fetch response: {str(e)}"}


@mcp.tool(description="Get the N most recent requests with optional URI filter")
def recent_requests(limit: int = 10, uri_pattern: Optional[str] = None) -> dict:
    """Get the most recent requests with optional URI filtering."""
    try:
        # Build query
        if uri_pattern:
            sql = (
                "SELECT batch_id, "
                "JSON_UNQUOTE(JSON_EXTRACT(content,'$.method')) as method, "
                "JSON_UNQUOTE(JSON_EXTRACT(content,'$.uri')) as uri, "
                "JSON_UNQUOTE(JSON_EXTRACT(content,'$.response_status')) as status, "
                "CAST(JSON_EXTRACT(content,'$.duration') AS DECIMAL(10,2)) as duration, "
                "created_at FROM telescope_entries "
                "WHERE type='request' AND JSON_EXTRACT(content,'$.uri') LIKE :pattern "
                "ORDER BY created_at DESC LIMIT :limit"
            )
            rows = _fetch(ENGINE, sql, {"pattern": f"%{uri_pattern}%", "limit": limit})
        else:
            sql = (
                "SELECT batch_id, "
                "JSON_UNQUOTE(JSON_EXTRACT(content,'$.method')) as method, "
                "JSON_UNQUOTE(JSON_EXTRACT(content,'$.uri')) as uri, "
                "JSON_UNQUOTE(JSON_EXTRACT(content,'$.response_status')) as status, "
                "CAST(JSON_EXTRACT(content,'$.duration') AS DECIMAL(10,2)) as duration, "
                "created_at FROM telescope_entries "
                "WHERE type='request' "
                "ORDER BY created_at DESC LIMIT :limit"
            )
            rows = _fetch(ENGINE, sql, {"limit": limit})
        
        # Format results
        requests = []
        for row in rows:
            requests.append({
                "batch_id": row["batch_id"],
                "method": row["method"] or "GET",
                "uri": row["uri"] or "/",
                "status": int(row["status"]) if row["status"] else 200,
                "duration_ms": float(row["duration"]) if row["duration"] else 0,
                "timestamp": str(row["created_at"])
            })
        
        result = {
            "requests": requests,
            "total": len(requests),
            "help": "Use get_request(batch_id) for detailed information about any request."
        }
        
        if uri_pattern:
            result["filter"] = uri_pattern
            
        return result
    except Exception as e:
        return {"error": f"Database query failed: {str(e)}"}


@mcp.tool(description="Return newest Telescope batch whose request URI contains the substring.")
def latest_request(uri_pattern: str) -> dict:
    """Get the newest request matching a URI pattern."""
    try:
        sql = f"""
            SELECT {REQUEST_COMMON_SELECT_FIELDS}
            FROM telescope_entries
            WHERE type='request' AND JSON_EXTRACT(content,'$.uri') LIKE :pattern
            ORDER BY created_at DESC LIMIT 1
        """
        rows = _fetch(ENGINE, sql, {"pattern": f"%{uri_pattern}%"})
        
        if not rows:
            return {"error": f"No request found matching URI pattern: {uri_pattern}"}
        
        row = rows[0]
        request_data = _format_request_row(row)
        del request_data["batch_id"]  # batch_id is at the top level

        return {
            "batch_id": row["batch_id"],
            "request": request_data,
            "help": f'Use get_request("{row["batch_id"]}") for full details.'
        }
    except Exception as e:
        return {"error": f"Database query failed: {str(e)}"}

@mcp.tool(description="Get Laravel Debugbar timing breakdown for a request if available")
def get_request_timing(batch_id: str) -> dict:
    """Extract timing breakdown from Laravel Debugbar for a request."""
    if not LARAVEL_PROJECT_PATH:
        return {
            "error": "LARAVEL_PROJECT_PATH environment variable not set.",
            "help": "Set the absolute path to your Laravel project root to use this tool."
        }
    try:
        # First, check if there's a debugbar entry for this batch
        sql = (
            "SELECT content FROM telescope_entries "
            "WHERE batch_id = :bid AND type = 'debugbar' LIMIT 1"
        )
        rows = _fetch(ENGINE, sql, {"bid": batch_id})

        if not rows:
            return {"error": "No debugbar data found for this request"}

        # Extract the debugbar request ID
        content = json.loads(rows[0]["content"])
        debugbar_id = content.get("requestId")

        if not debugbar_id:
            return {"error": "No requestId found in debugbar entry"}

        # Construct path to debugbar JSON file
        # Assuming we are running in a Laravel project
        debugbar_path = os.path.join(LARAVEL_PROJECT_PATH, f"storage/debugbar/{debugbar_id}.json")

        # Check if file exists and read it
        if not os.path.exists(debugbar_path):
            return {"error": f"Debugbar file not found: {debugbar_id}.json at path {debugbar_path}"}

        with open(debugbar_path, 'r') as f:
            debugbar_data = json.load(f)

        # Extract timing information
        time_data = debugbar_data.get("time", {})

        # Build response with timing breakdown
        timing_breakdown = []
        total_duration = time_data.get("duration", 0)

        for measure in time_data.get("measures", []):
            percentage = round((measure["duration"] / total_duration * 100), 1) if total_duration > 0 else 0
            timing_breakdown.append({
                "phase": measure["label"],
                "duration_ms": round(measure["duration"] * 1000, 2),
                "duration_str": measure["duration_str"],
                "start_ms": round(measure["relative_start"] * 1000, 2),
                "percentage": f"{percentage}%"
            })

        return {
            "batch_id": batch_id,
            "debugbar_id": debugbar_id,
            "total_duration": time_data.get("duration_str", "N/A"),
            "total_duration_ms": round(total_duration * 1000, 2),
            "timing_breakdown": timing_breakdown,
            "queries": {
                "count": debugbar_data.get("queries", {}).get("nb_statements", 0),
                "time": debugbar_data.get("queries", {}).get("accumulated_duration_str", "0ms")
            },
            "memory": {
                "peak_usage": debugbar_data.get("memory", {}).get("peak_usage_str", "N/A")
            }
        }

    except json.JSONDecodeError as e:
        return {"error": f"Invalid JSON in debugbar file: {str(e)}"}
    except Exception as e:
        return {"error": f"Failed to fetch timing data: {str(e)}"}


################################################################################
# Entrypoint                                                                   #
################################################################################

def main():
    """Main entry point for the telescope-mcp command."""
    import sys
    
    # Check for version flag
    if len(sys.argv) > 1 and sys.argv[1] in ["--version", "-v"]:
        print(f"telescope-mcp version {__version__}")
        sys.exit(0)
    
    # Run the MCP server
    mcp.run()


if __name__ == "__main__":
    main()