"""
Core persistence layer with intelligent data routing and caching
Designed to scale from SQLite to PostgreSQL seamlessly
"""

import sqlite3
import json
import time
import asyncio
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
from datetime import datetime, timedelta
import hashlib
import pickle

class DatabaseManager:
    """
    Unified database manager with intelligent caching and routing
    Supports SQLite for development, PostgreSQL for production
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self.db_type = config.get('db_type', 'sqlite')  # 'sqlite' or 'postgresql'
        self.db_path = Path(config.get('db_path', 'data/chimera.db'))
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Connection pool for scaling
        self.connection_pool = []
        self.max_connections = config.get('max_connections', 5)
        
        # Intelligent caching layers
        self.hot_cache = {}  # Frequently accessed data (in-memory)
        self.warm_cache = {}  # Recent data
        self.cache_stats = defaultdict(int)  # Track access patterns
        
        # Write buffer for batch operations
        self.write_buffer = []
        self.buffer_size = config.get('buffer_size', 100)
        self.last_flush = time.time()
        
        # Initialize database
        self._init_database()
        
        # Start background tasks
        asyncio.create_task(self._cache_manager())
        asyncio.create_task(self._buffer_flush_manager())
        
    def _init_database(self):
        """Initialize database with optimized schema"""
        
        conn = self._get_connection()
        
        # Enable optimizations for SQLite
        if self.db_type == 'sqlite':
            conn.execute("PRAGMA journal_mode=WAL")  # Write-ahead logging
            conn.execute("PRAGMA synchronous=NORMAL")  # Balance safety/speed
            conn.execute("PRAGMA cache_size=10000")  # Larger cache
            conn.execute("PRAGMA temp_store=MEMORY")  # Use memory for temp tables
            
        # ========== THOUGHTS TABLE ==========
        # Core of the reasoning system
        conn.execute("""
            CREATE TABLE IF NOT EXISTS thoughts (
                id TEXT PRIMARY KEY,
                content TEXT NOT NULL,
                symbolic_form TEXT,
                confidence REAL DEFAULT 0.5,
                timestamp REAL NOT NULL,
                
                -- Connections stored as JSON array of thought IDs
                connections TEXT DEFAULT '[]',
                
                -- Vector embedding for similarity search (serialized numpy array)
                embedding BLOB,
                
                -- Which agent created this thought
                created_by TEXT,
                
                -- Resonance pattern for phase-locking
                resonance_pattern BLOB,
                
                -- How many times this thought has been activated
                activation_count INTEGER DEFAULT 0,
                last_activated REAL,
                
                -- Decay tracking for memory management
                strength REAL DEFAULT 1.0,
                
                -- Indexing for fast retrieval
                INDEX idx_thoughts_timestamp (timestamp DESC),
                INDEX idx_thoughts_confidence (confidence DESC),
                INDEX idx_thoughts_activation (activation_count DESC),
                INDEX idx_thoughts_strength (strength DESC)
            )
        """)
        
        # ========== CRYSTALLIZED INSIGHTS TABLE ==========
        # Permanent, verified knowledge
        conn.execute("""
            CREATE TABLE IF NOT EXISTS crystallized_insights (
                id TEXT PRIMARY KEY,
                timestamp REAL NOT NULL,
                
                -- The actual insight
                content TEXT NOT NULL,
                symbolic_representation TEXT,
                linguistic_expression TEXT,
                
                -- Confidence and verification
                confidence REAL DEFAULT 0.5,
                verification_count INTEGER DEFAULT 0,
                falsification_attempts INTEGER DEFAULT 0,
                empirical_support REAL DEFAULT 0.0,
                
                -- Source thoughts that led to this insight
                source_thoughts TEXT DEFAULT '[]',  -- JSON array
                
                -- Groundings in sensory/experiential data
                groundings TEXT DEFAULT '[]',  -- JSON array
                
                -- Which crystallization cycle created this
                crystallization_cycle INTEGER,
                
                INDEX idx_insights_confidence (confidence DESC),
                INDEX idx_insights_timestamp (timestamp DESC)
            )
        """)
        
        # ========== CONCEPTS TABLE ==========
        # Learned abstract concepts
        conn.execute("""
            CREATE TABLE IF NOT EXISTS concepts (
                id TEXT PRIMARY KEY,
                term TEXT UNIQUE NOT NULL,
                definition TEXT,
                examples TEXT,  -- JSON array
                
                -- Learning metadata
                first_learned REAL NOT NULL,
                last_reinforced REAL,
                reinforcement_count INTEGER DEFAULT 1,
                
                -- Confidence and understanding level
                confidence REAL DEFAULT 0.5,
                abstraction_level INTEGER DEFAULT 0,
                
                -- Related concepts (for semantic network)
                related_concepts TEXT DEFAULT '[]',  -- JSON array
                
                -- Which thoughts contributed to this concept
                contributing_thoughts TEXT DEFAULT '[]',  -- JSON array
                
                -- Usage statistics
                usage_count INTEGER DEFAULT 0,
                successful_applications INTEGER DEFAULT 0,
                
                INDEX idx_concepts_term (term),
                INDEX idx_concepts_confidence (confidence DESC),
                INDEX idx_concepts_usage (usage_count DESC)
            )
        """)
        
        # ========== CONVERSATIONS TABLE ==========
        # Complete interaction history
        conn.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp REAL NOT NULL,
                
                -- The exchange
                speaker TEXT NOT NULL,  -- 'user' or 'chimera'
                message TEXT NOT NULL,
                response TEXT,
                
                -- Understanding metrics
                understanding_confidence REAL,
                tokens_recognized INTEGER,
                tokens_unknown INTEGER,
                
                -- What happened internally
                thoughts_formed TEXT DEFAULT '[]',  -- JSON array of thought IDs
                concepts_activated TEXT DEFAULT '[]',  -- JSON array
                insights_triggered TEXT DEFAULT '[]',  -- JSON array
                
                -- Emotional/contextual metadata
                emotional_valence REAL DEFAULT 0.0,
                curiosity_level REAL,
                
                INDEX idx_conv_session (session_id),
                INDEX idx_conv_timestamp (timestamp DESC),
                INDEX idx_conv_speaker (speaker)
            )
        """)
        
        # ========== SEMANTIC MEMORY TABLE ==========
        # Episodic and semantic memory storage
        conn.execute("""
            CREATE TABLE IF NOT EXISTS semantic_memory (
                id TEXT PRIMARY KEY,
                memory_type TEXT NOT NULL,  -- 'episodic', 'semantic', 'procedural'
                
                -- The memory content
                content TEXT NOT NULL,
                context TEXT,  -- JSON object with context
                
                -- When and how it was formed
                formed_at REAL NOT NULL,
                last_accessed REAL,
                access_count INTEGER DEFAULT 0,
                
                -- Importance and decay
                importance REAL DEFAULT 0.5,
                emotional_weight REAL DEFAULT 0.0,
                decay_rate REAL DEFAULT 0.01,
                current_strength REAL DEFAULT 1.0,
                
                -- Associations
                associated_thoughts TEXT DEFAULT '[]',  -- JSON array
                associated_concepts TEXT DEFAULT '[]',  -- JSON array
                triggers TEXT DEFAULT '[]',  -- What triggers recall
                
                INDEX idx_memory_type (memory_type),
                INDEX idx_memory_importance (importance DESC),
                INDEX idx_memory_strength (current_strength DESC)
            )
        """)
        
        # ========== CURIOSITIES TABLE ==========
        # Autonomous interests and questions
        conn.execute("""
            CREATE TABLE IF NOT EXISTS curiosities (
                id TEXT PRIMARY KEY,
                target TEXT NOT NULL,  -- What we're curious about
                target_type TEXT,  -- 'entity', 'concept', 'pattern', etc.
                
                -- The curiosity itself
                questions TEXT NOT NULL,  -- JSON array of questions
                priority REAL DEFAULT 0.5,
                intensity REAL DEFAULT 1.0,
                
                -- Pursuit tracking
                created_at REAL NOT NULL,
                last_pursued REAL,
                exploration_attempts INTEGER DEFAULT 0,
                satisfaction_level REAL DEFAULT 0.0,
                
                -- Resolution
                resolved BOOLEAN DEFAULT FALSE,
                resolution TEXT,
                resolved_at REAL,
                
                -- What triggered this curiosity
                trigger_thought TEXT,
                trigger_context TEXT,  -- JSON
                
                INDEX idx_curiosity_priority (priority DESC),
                INDEX idx_curiosity_resolved (resolved),
                INDEX idx_curiosity_satisfaction (satisfaction_level)
            )
        """)
        
        # ========== AGENT STATES TABLE ==========
        # Preserve agent states for persistence
        conn.execute("""
            CREATE TABLE IF NOT EXISTS agent_states (
                agent_id TEXT PRIMARY KEY,
                agent_type TEXT NOT NULL,
                
                -- Serialized state
                state_data BLOB NOT NULL,  -- Pickled Python object
                
                -- Metadata
                last_updated REAL NOT NULL,
                tick_count INTEGER DEFAULT 0,
                total_messages_processed INTEGER DEFAULT 0,
                
                -- Performance metrics
                avg_processing_time REAL,
                error_count INTEGER DEFAULT 0,
                last_error TEXT
            )
        """)
        
        # ========== PHASE COHERENCE TABLE ==========
        # Track phase-locking patterns
        conn.execute("""
            CREATE TABLE IF NOT EXISTS phase_coherence (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                
                -- Coherence measurements
                global_coherence REAL,
                agent_phases TEXT,  -- JSON object {agent_id: phase}
                
                -- Which agents were synchronized
                synchronized_agents TEXT,  -- JSON array
                coherence_strength REAL,
                
                -- What emerged from this coherence
                emergent_pattern TEXT,
                resulted_in_insight BOOLEAN DEFAULT FALSE,
                insight_id TEXT,
                
                INDEX idx_coherence_timestamp (timestamp DESC),
                INDEX idx_coherence_strength (coherence_strength DESC)
            )
        """)
        
        # ========== PERFORMANCE METRICS TABLE ==========
        # Track system performance for optimization
        conn.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                metric_type TEXT NOT NULL,
                
                -- The metric value
                value REAL NOT NULL,
                unit TEXT,
                
                -- Context
                agent_id TEXT,
                operation TEXT,
                
                -- For identifying bottlenecks
                duration_ms REAL,
                memory_usage_mb REAL,
                
                INDEX idx_metrics_timestamp (timestamp DESC),
                INDEX idx_metrics_type (metric_type)
            )
        """)
        
        conn.commit()
        self._return_connection(conn)
        
    def _get_connection(self):
        """Get a database connection from pool"""
        if self.connection_pool:
            return self.connection_pool.pop()
        
        if self.db_type == 'sqlite':
            conn = sqlite3.connect(
                self.db_path,
                check_same_thread=False,
                isolation_level=None  # Autocommit mode
            )
            conn.row_factory = sqlite3.Row  # Enable column access by name
        else:
            # PostgreSQL connection would go here
            # import psycopg2
            # conn = psycopg2.connect(self.config['pg_connection_string'])
            pass
            
        return conn
        
    def _return_connection(self, conn):
        """Return connection to pool"""
        if len(self.connection_pool) < self.max_connections:
            self.connection_pool.append(conn)
        else:
            conn.close()