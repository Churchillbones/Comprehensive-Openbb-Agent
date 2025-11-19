"""
Context Manager for OpenBB Comprehensive Agent

This module manages conversation state and context across agent interactions.
Maintains history, caches data, and provides context to agents.
"""

from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import logging


# Configure logging
logger = logging.getLogger(__name__)


class ContextManager:
    """
    Manage conversation state and context across agent interactions

    Handles:
    - Conversation history tracking
    - Context caching for agent reuse
    - Session management
    - Widget data persistence
    - Query context building
    """

    def __init__(self, session_ttl: int = 3600):
        """
        Initialize the context manager

        Args:
            session_ttl: Session time-to-live in seconds (default 1 hour)
        """
        self.sessions: Dict[str, Dict] = {}
        self.session_ttl = session_ttl
        self.cache: Dict[str, Any] = {}
        self.cache_timestamps: Dict[str, datetime] = {}
        logger.info(f"Context manager initialized (session TTL: {session_ttl}s)")

    def create_session(self, session_id: str) -> Dict:
        """
        Create a new conversation session

        Args:
            session_id: Unique session identifier

        Returns:
            Session dictionary
        """
        if session_id in self.sessions:
            logger.warning(f"Session {session_id} already exists, resetting")

        self.sessions[session_id] = {
            "id": session_id,
            "created_at": datetime.now(),
            "last_accessed": datetime.now(),
            "history": [],
            "context": {},
            "widget_data": {},
            "uploaded_files": [],
            "preferences": {}
        }

        logger.info(f"Created session: {session_id}")
        return self.sessions[session_id]

    def get_session(self, session_id: str) -> Optional[Dict]:
        """
        Get an existing session

        Args:
            session_id: Session identifier

        Returns:
            Session dictionary or None if not found/expired
        """
        if session_id not in self.sessions:
            logger.debug(f"Session not found: {session_id}")
            return None

        session = self.sessions[session_id]

        # Check if session expired
        if self._is_session_expired(session):
            logger.info(f"Session expired: {session_id}")
            self.delete_session(session_id)
            return None

        # Update last accessed time
        session["last_accessed"] = datetime.now()
        return session

    def get_or_create_session(self, session_id: str) -> Dict:
        """
        Get existing session or create new one

        Args:
            session_id: Session identifier

        Returns:
            Session dictionary
        """
        session = self.get_session(session_id)
        if session is None:
            session = self.create_session(session_id)
        return session

    def delete_session(self, session_id: str) -> bool:
        """
        Delete a session

        Args:
            session_id: Session identifier

        Returns:
            True if session was deleted, False if not found
        """
        if session_id in self.sessions:
            del self.sessions[session_id]
            logger.info(f"Deleted session: {session_id}")
            return True
        return False

    def _is_session_expired(self, session: Dict) -> bool:
        """
        Check if session has expired

        Args:
            session: Session dictionary

        Returns:
            True if session is expired
        """
        last_accessed = session.get("last_accessed")
        if not last_accessed:
            return True

        expiry_time = last_accessed + timedelta(seconds=self.session_ttl)
        return datetime.now() > expiry_time

    def add_to_history(
        self,
        session_id: str,
        role: str,
        content: str,
        metadata: Optional[Dict] = None
    ) -> None:
        """
        Add a message to conversation history

        Args:
            session_id: Session identifier
            role: Message role (human, ai, system, tool)
            content: Message content
            metadata: Optional metadata for the message
        """
        session = self.get_or_create_session(session_id)

        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }

        session["history"].append(message)
        logger.debug(f"Added {role} message to session {session_id} history")

    def get_history(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """
        Get conversation history for a session

        Args:
            session_id: Session identifier
            limit: Maximum number of messages to return (most recent)

        Returns:
            List of message dictionaries
        """
        session = self.get_session(session_id)
        if not session:
            return []

        history = session.get("history", [])

        if limit:
            return history[-limit:]
        return history

    def update_context(
        self,
        session_id: str,
        context_updates: Dict[str, Any]
    ) -> None:
        """
        Update session context with new data

        Args:
            session_id: Session identifier
            context_updates: Dictionary of context updates
        """
        session = self.get_or_create_session(session_id)
        session["context"].update(context_updates)
        logger.debug(f"Updated context for session {session_id}: {list(context_updates.keys())}")

    def get_context(self, session_id: str) -> Dict[str, Any]:
        """
        Get current context for a session

        Args:
            session_id: Session identifier

        Returns:
            Context dictionary
        """
        session = self.get_session(session_id)
        if not session:
            return {}
        return session.get("context", {})

    def set_widget_data(
        self,
        session_id: str,
        widget_id: str,
        widget_data: Dict[str, Any]
    ) -> None:
        """
        Store widget data for a session

        Args:
            session_id: Session identifier
            widget_id: Widget identifier
            widget_data: Widget data dictionary
        """
        session = self.get_or_create_session(session_id)
        session["widget_data"][widget_id] = {
            "data": widget_data,
            "timestamp": datetime.now().isoformat()
        }
        logger.debug(f"Stored widget data for session {session_id}, widget {widget_id}")

    def get_widget_data(
        self,
        session_id: str,
        widget_id: Optional[str] = None
    ) -> Optional[Dict]:
        """
        Get widget data for a session

        Args:
            session_id: Session identifier
            widget_id: Optional specific widget ID, if None returns all widget data

        Returns:
            Widget data dictionary or None
        """
        session = self.get_session(session_id)
        if not session:
            return None

        widget_data = session.get("widget_data", {})

        if widget_id:
            widget_entry = widget_data.get(widget_id)
            return widget_entry.get("data") if widget_entry else None

        # Return all widget data
        return {
            wid: entry["data"]
            for wid, entry in widget_data.items()
        }

    def add_uploaded_file(
        self,
        session_id: str,
        filename: str,
        file_type: str,
        content: Any
    ) -> None:
        """
        Track uploaded file for a session

        Args:
            session_id: Session identifier
            filename: Name of uploaded file
            file_type: File type/extension
            content: File content or reference
        """
        session = self.get_or_create_session(session_id)
        session["uploaded_files"].append({
            "filename": filename,
            "file_type": file_type,
            "content": content,
            "uploaded_at": datetime.now().isoformat()
        })
        logger.debug(f"Added uploaded file to session {session_id}: {filename}")

    def get_uploaded_files(self, session_id: str) -> List[Dict]:
        """
        Get all uploaded files for a session

        Args:
            session_id: Session identifier

        Returns:
            List of uploaded file dictionaries
        """
        session = self.get_session(session_id)
        if not session:
            return []
        return session.get("uploaded_files", [])

    def build_query_context(
        self,
        session_id: str,
        query: str,
        additional_context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        Build complete context for a query

        Args:
            session_id: Session identifier
            query: User query
            additional_context: Optional additional context data

        Returns:
            Complete context dictionary for agent processing
        """
        session = self.get_or_create_session(session_id)

        context = {
            "query": query,
            "session_id": session_id,
            "history": self.get_history(session_id, limit=10),  # Last 10 messages
            "session_context": session.get("context", {}),
            "widget_data": self.get_widget_data(session_id),
            "uploaded_files": self.get_uploaded_files(session_id),
            "timestamp": datetime.now().isoformat()
        }

        # Merge additional context if provided
        if additional_context:
            context.update(additional_context)

        return context

    def cache_data(
        self,
        key: str,
        data: Any,
        ttl: Optional[int] = None
    ) -> None:
        """
        Cache data with optional TTL

        Args:
            key: Cache key
            data: Data to cache
            ttl: Time-to-live in seconds (None for indefinite)
        """
        self.cache[key] = data
        self.cache_timestamps[key] = datetime.now()

        if ttl:
            # Note: TTL enforcement happens on retrieval
            pass

        logger.debug(f"Cached data with key: {key}")

    def get_cached_data(
        self,
        key: str,
        ttl: Optional[int] = None
    ) -> Optional[Any]:
        """
        Get cached data

        Args:
            key: Cache key
            ttl: Time-to-live in seconds to check against

        Returns:
            Cached data or None if not found/expired
        """
        if key not in self.cache:
            return None

        # Check TTL if provided
        if ttl:
            timestamp = self.cache_timestamps.get(key)
            if timestamp:
                expiry_time = timestamp + timedelta(seconds=ttl)
                if datetime.now() > expiry_time:
                    # Expired, remove from cache
                    del self.cache[key]
                    del self.cache_timestamps[key]
                    logger.debug(f"Cache expired for key: {key}")
                    return None

        return self.cache[key]

    def clear_cache(self) -> None:
        """Clear all cached data"""
        self.cache.clear()
        self.cache_timestamps.clear()
        logger.info("Cache cleared")

    def cleanup_expired_sessions(self) -> int:
        """
        Remove expired sessions

        Returns:
            Number of sessions removed
        """
        expired_sessions = [
            session_id
            for session_id, session in self.sessions.items()
            if self._is_session_expired(session)
        ]

        for session_id in expired_sessions:
            self.delete_session(session_id)

        if expired_sessions:
            logger.info(f"Cleaned up {len(expired_sessions)} expired sessions")

        return len(expired_sessions)

    def get_stats(self) -> Dict[str, Any]:
        """
        Get context manager statistics

        Returns:
            Dictionary with stats about sessions and cache
        """
        return {
            "active_sessions": len(self.sessions),
            "cached_items": len(self.cache),
            "session_ttl": self.session_ttl,
            "sessions": [
                {
                    "id": session["id"],
                    "created_at": session["created_at"].isoformat(),
                    "last_accessed": session["last_accessed"].isoformat(),
                    "message_count": len(session.get("history", [])),
                    "widget_count": len(session.get("widget_data", {})),
                    "file_count": len(session.get("uploaded_files", []))
                }
                for session in self.sessions.values()
            ]
        }

    def __repr__(self) -> str:
        """String representation of the context manager"""
        return (
            f"<ContextManager sessions={len(self.sessions)} "
            f"cached_items={len(self.cache)}>"
        )
