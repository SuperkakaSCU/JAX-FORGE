"""SQLite data storage for forging VAE training data.

Follows the cogging DBMS pattern: one DB file, start_series / record_hit API.
Stores per-hit (input_state, hit_params) → output_state training pairs.
"""

import json
import sqlite3
import uuid
import time
from pathlib import Path


class ForgingDBMS:
    """SQLite database manager for forging simulation training data.

    Schema
    ------
    series
        id              TEXT PRIMARY KEY   — UUID
        sequence_type   TEXT               — e.g. 'one_press', 'progressive'
        num_hits        INTEGER            — expected total hits in this series
        status          TEXT               — 'running' | 'completed' | 'failed_at_hit_N'
        time_start      TEXT               — ISO timestamp
        time_end        TEXT               — ISO timestamp (NULL until finished)

    hits
        id              INTEGER PRIMARY KEY AUTOINCREMENT
        series_id       TEXT               — FK → series.id
        step_number     INTEGER            — 1-based hit index within series
        x_min_band      REAL               — band lower bound (fraction of H)
        x_max_band      REAL               — band upper bound (fraction of H)
        compression_displacement REAL      — platen displacement (mm)
        rotation_euler_x REAL              — rotation angle (degrees)
        total_time       REAL              — hit duration (seconds)
        input_displacement  TEXT           — JSON: flat list, (N*3,)
        input_temperature   TEXT           — JSON: flat list, (N*1,)
        result_displacement TEXT           — JSON: flat list, (N*3,)
        result_temperature  TEXT           — JSON: flat list, (N*1,)
        solver_steps    INTEGER            — adaptive stepper step count
        wall_time       REAL               — wall-clock seconds for this hit
    """

    def __init__(self, db_path, new=False):
        """Open (or create) the database.

        Parameters
        ----------
        db_path : str or Path
            Path to the SQLite database file.
        new : bool
            If True, drop existing tables and recreate.
        """
        self.db_path = str(db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        if new:
            self._drop_tables()
        self._create_tables()

    # ------------------------------------------------------------------
    # Schema management
    # ------------------------------------------------------------------
    def _drop_tables(self):
        cur = self.conn.cursor()
        cur.execute("DROP TABLE IF EXISTS hits")
        cur.execute("DROP TABLE IF EXISTS series")
        self.conn.commit()

    def _create_tables(self):
        cur = self.conn.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS series (
                id              TEXT PRIMARY KEY,
                sequence_type   TEXT NOT NULL,
                num_hits        INTEGER NOT NULL,
                status          TEXT NOT NULL DEFAULT 'running',
                time_start      TEXT NOT NULL,
                time_end        TEXT
            )
        """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS hits (
                id                      INTEGER PRIMARY KEY AUTOINCREMENT,
                series_id               TEXT NOT NULL,
                step_number             INTEGER NOT NULL,
                x_min_band              REAL NOT NULL,
                x_max_band              REAL NOT NULL,
                compression_displacement REAL NOT NULL,
                rotation_euler_x        REAL NOT NULL,
                total_time              REAL NOT NULL,
                input_displacement      TEXT NOT NULL,
                input_temperature       TEXT NOT NULL,
                result_displacement     TEXT NOT NULL,
                result_temperature      TEXT NOT NULL,
                solver_steps            INTEGER,
                wall_time               REAL,
                FOREIGN KEY (series_id) REFERENCES series(id)
            )
        """)
        self.conn.commit()

    # ------------------------------------------------------------------
    # Series lifecycle
    # ------------------------------------------------------------------
    def start_series(self, sequence_type, num_hits):
        """Begin a new series. Returns the series_id (UUID string)."""
        series_id = str(uuid.uuid4())
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO series (id, sequence_type, num_hits, status, time_start) "
            "VALUES (?, ?, ?, 'running', datetime('now'))",
            (series_id, sequence_type, num_hits),
        )
        self.conn.commit()
        return series_id

    def complete_series(self, series_id):
        """Mark a series as successfully completed."""
        cur = self.conn.cursor()
        cur.execute(
            "UPDATE series SET status='completed', time_end=datetime('now') WHERE id=?",
            (series_id,),
        )
        self.conn.commit()

    def fail_series(self, series_id, failed_at_hit):
        """Mark a series as failed at a specific hit number.

        Partial data (hits before the failure) is preserved in the DB.
        """
        status = f"failed_at_hit_{failed_at_hit}"
        cur = self.conn.cursor()
        cur.execute(
            "UPDATE series SET status=?, time_end=datetime('now') WHERE id=?",
            (status, series_id),
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    # Hit recording
    # ------------------------------------------------------------------
    def record_hit(self, series_id, step_number, hit, input_u, input_T,
                   result_u, result_T, solver_steps=None, wall_time=None):
        """Record one hit's input/output pair.

        Parameters
        ----------
        series_id : str
        step_number : int
            1-based hit index within the series.
        hit : Hit
            Hit dataclass instance with band, displacement, rotation, time.
        input_u : array-like, shape (N, 3)
            Displacement field before the hit.
        input_T : array-like, shape (N, 1)
            Temperature field before the hit.
        result_u : array-like, shape (N, 3)
            Displacement field after the hit.
        result_T : array-like, shape (N, 1)
            Temperature field after the hit.
        solver_steps : int or None
        wall_time : float or None
        """
        import numpy as onp

        def _to_json(arr):
            return json.dumps(onp.asarray(arr).flatten().tolist())

        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO hits "
            "(series_id, step_number, x_min_band, x_max_band, "
            " compression_displacement, rotation_euler_x, total_time, "
            " input_displacement, input_temperature, "
            " result_displacement, result_temperature, "
            " solver_steps, wall_time) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                series_id,
                step_number,
                float(hit.x_min_band),
                float(hit.x_max_band),
                float(hit.compression_displacement),
                float(hit.rotation_euler_x),
                float(hit.total_time),
                _to_json(input_u),
                _to_json(input_T),
                _to_json(result_u),
                _to_json(result_T),
                solver_steps,
                wall_time,
            ),
        )
        self.conn.commit()

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------
    def get_series_count(self, status=None):
        """Return number of series, optionally filtered by status."""
        cur = self.conn.cursor()
        if status:
            cur.execute("SELECT COUNT(*) FROM series WHERE status=?", (status,))
        else:
            cur.execute("SELECT COUNT(*) FROM series")
        return cur.fetchone()[0]

    def get_hit_count(self, series_id=None):
        """Return total number of hits, optionally for one series."""
        cur = self.conn.cursor()
        if series_id:
            cur.execute("SELECT COUNT(*) FROM hits WHERE series_id=?", (series_id,))
        else:
            cur.execute("SELECT COUNT(*) FROM hits")
        return cur.fetchone()[0]

    def get_series_hits(self, series_id):
        """Return all hit rows for a series, ordered by step_number."""
        cur = self.conn.cursor()
        cur.execute(
            "SELECT * FROM hits WHERE series_id=? ORDER BY step_number",
            (series_id,),
        )
        columns = [desc[0] for desc in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]

    def get_all_series(self, status=None):
        """Return all series records."""
        cur = self.conn.cursor()
        if status:
            cur.execute("SELECT * FROM series WHERE status=?", (status,))
        else:
            cur.execute("SELECT * FROM series")
        columns = [desc[0] for desc in cur.description]
        return [dict(zip(columns, row)) for row in cur.fetchall()]

    def summary(self):
        """Print a summary of the database contents."""
        total_series = self.get_series_count()
        completed = self.get_series_count("completed")
        failed = total_series - completed - self.get_series_count("running")
        total_hits = self.get_hit_count()
        print(f"Database: {self.db_path}")
        print(f"  Series: {total_series} (completed={completed}, failed={failed})")
        print(f"  Hits:   {total_hits}")

    def close(self):
        """Close the database connection."""
        self.conn.close()
