import sqlite3
import uuid
import time
import re
from pathlib import Path
from config import settings


class UserProfile:

    # Initialize class state.
    def __init__(self, db_path=None):
        """
        Initialize the instance state.
        
        This method implements the init step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        self.db_path = str(db_path or settings.USERS_DB_PATH)
        self._init_db()

    # Internal helper to init db.
    def _init_db(self):
        """
        Initialize db.
        
        This method implements the init db step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "\n            CREATE TABLE IF NOT EXISTS user_likes (\n                user_id TEXT,\n                song_id TEXT,\n                liked_at REAL,\n                PRIMARY KEY (user_id, song_id)\n            )\n        "
        )
        conn.execute(
            "\n            CREATE TABLE IF NOT EXISTS user_dislikes (\n                user_id TEXT,\n                song_id TEXT,\n                disliked_at REAL,\n                PRIMARY KEY (user_id, song_id)\n            )\n        "
        )
        conn.execute(
            "\n            CREATE TABLE IF NOT EXISTS user_plays (\n                user_id TEXT,\n                song_id TEXT,\n                played_at REAL\n            )\n        "
        )
        conn.execute(
            "\n            CREATE TABLE IF NOT EXISTS user_skip_events (\n                user_id TEXT,\n                song_id TEXT,\n                event TEXT,\n                position_sec REAL,\n                duration_sec REAL,\n                event_at REAL\n            )\n        "
        )
        conn.execute(
            "\n            CREATE TABLE IF NOT EXISTS user_playlists (\n                playlist_id TEXT PRIMARY KEY,\n                user_id TEXT,\n                name TEXT,\n                cover_image TEXT,\n                created_at REAL,\n                updated_at REAL\n            )\n        "
        )
        conn.execute(
            "\n            CREATE TABLE IF NOT EXISTS user_playlist_tracks (\n                playlist_id TEXT,\n                user_id TEXT,\n                song_id TEXT,\n                added_at REAL,\n                PRIMARY KEY (playlist_id, song_id)\n            )\n        "
        )
        conn.execute(
            "\n            CREATE TABLE IF NOT EXISTS user_accounts (\n                user_id TEXT PRIMARY KEY,\n                username TEXT,\n                username_norm TEXT UNIQUE,\n                password_hash TEXT,\n                avatar_image TEXT,\n                banner_image TEXT,\n                created_at REAL,\n                updated_at REAL\n            )\n        "
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_likes_user ON user_likes(user_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_dislikes_user ON user_dislikes(user_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_plays_user ON user_plays(user_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_skip_events_user ON user_skip_events(user_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_skip_events_song ON user_skip_events(song_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_playlists_user ON user_playlists(user_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_playlist_tracks_user ON user_playlist_tracks(user_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_playlist_tracks_playlist ON user_playlist_tracks(playlist_id)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_accounts_username_norm ON user_accounts(username_norm)"
        )
        # Lightweight schema migration for existing DBs created before playlist cover support.
        try:
            playlist_cols = {
                str(row[1] or "").strip()
                for row in conn.execute("PRAGMA table_info(user_playlists)").fetchall()
            }
            if "cover_image" not in playlist_cols:
                conn.execute("ALTER TABLE user_playlists ADD COLUMN cover_image TEXT")
        except Exception:
            pass
        # Lightweight schema migration for existing DBs created before account normalization.
        try:
            account_cols = {
                str(row[1] or "").strip()
                for row in conn.execute("PRAGMA table_info(user_accounts)").fetchall()
            }
            if "username_norm" not in account_cols:
                conn.execute("ALTER TABLE user_accounts ADD COLUMN username_norm TEXT")
                rows = conn.execute("SELECT user_id, username FROM user_accounts").fetchall()
                for row in rows:
                    uid = str(row[0] or "").strip()
                    uname = str(row[1] or "").strip()
                    conn.execute(
                        "UPDATE user_accounts SET username_norm=? WHERE user_id=?",
                        (self._normalize_username(uname), uid),
                    )
                conn.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS idx_accounts_username_norm ON user_accounts(username_norm)"
                )
            if "updated_at" not in account_cols:
                conn.execute("ALTER TABLE user_accounts ADD COLUMN updated_at REAL")
                conn.execute(
                    "UPDATE user_accounts SET updated_at=COALESCE(updated_at, created_at, ?)",
                    (time.time(),),
                )
            if "avatar_image" not in account_cols:
                conn.execute("ALTER TABLE user_accounts ADD COLUMN avatar_image TEXT")
                conn.execute(
                    "UPDATE user_accounts SET avatar_image=COALESCE(avatar_image, '')"
                )
            if "banner_image" not in account_cols:
                conn.execute("ALTER TABLE user_accounts ADD COLUMN banner_image TEXT")
                conn.execute(
                    "UPDATE user_accounts SET banner_image=COALESCE(banner_image, '')"
                )
        except Exception:
            pass
        conn.commit()
        conn.close()

    # Internal helper to conn.
    def _conn(self):
        """
        Execute conn.
        
        This method implements the conn step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        return sqlite3.connect(self.db_path)

    # Internal helper to check duplicate playlist names (case-insensitive per user).
    def _playlist_name_exists(self, conn, user_id, name, exclude_playlist_id=None):
        """
        Execute playlist name exists.
        
        This method implements the playlist name exists step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        playlist_name = str(name or "").strip()
        if not playlist_name:
            return False
        sql = (
            "SELECT 1 FROM user_playlists "
            "WHERE user_id=? AND lower(trim(name)) = lower(trim(?))"
        )
        params = [user_id, playlist_name]
        excluded = str(exclude_playlist_id or "").strip()
        if excluded:
            sql += " AND playlist_id<>?"
            params.append(excluded)
        row = conn.execute(sql, tuple(params)).fetchone()
        return bool(row)

    # Handle generate user id.
    @staticmethod
    def generate_user_id():
        """
        Execute generate user id.
        
        This method implements the generate user id step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        return str(uuid.uuid4())

    # Normalize username for case-insensitive lookups.
    @staticmethod
    def _normalize_username(username):
        """
        Normalize username.
        
        This method implements the normalize username step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        return str(username or "").strip().lower()

    # Validate username policy.
    @staticmethod
    def _validate_username(username):
        """
        Validate username.
        
        This method implements the validate username step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        text = str(username or "").strip()
        if len(text) < 3:
            raise ValueError("username must be at least 3 characters")
        if len(text) > 32:
            raise ValueError("username must be at most 32 characters")
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", text):
            raise ValueError("username can only use letters, numbers, _, -, and .")
        return text

    # Create account (permanent user identity).
    def create_account(self, username, password_hash, avatar_image="", banner_image=""):
        """
        Create account.
        
        This method implements the create account step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        uname = self._validate_username(username)
        norm = self._normalize_username(uname)
        pwhash = str(password_hash or "").strip()
        avatar = str(avatar_image or "").strip()
        banner = str(banner_image or "").strip()
        if not pwhash:
            raise ValueError("password hash required")
        user_id = self.generate_user_id()
        now = time.time()
        conn = self._conn()
        exists = conn.execute(
            "SELECT 1 FROM user_accounts WHERE username_norm=?",
            (norm,),
        ).fetchone()
        if exists:
            conn.close()
            raise ValueError("username already exists")
        conn.execute(
            "INSERT INTO user_accounts (user_id, username, username_norm, password_hash, avatar_image, banner_image, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (user_id, uname, norm, pwhash, avatar, banner, now, now),
        )
        conn.commit()
        conn.close()
        return {
            "user_id": user_id,
            "username": uname,
            "avatar_image": avatar,
            "banner_image": banner,
            "created_at": now,
            "updated_at": now,
        }

    # Fetch account by case-insensitive username.
    def get_account_by_username(self, username):
        """
        Get account by username.
        
        This method implements the get account by username step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        norm = self._normalize_username(username)
        if not norm:
            return None
        conn = self._conn()
        row = conn.execute(
            "SELECT user_id, username, username_norm, password_hash, avatar_image, banner_image, created_at, updated_at FROM user_accounts WHERE username_norm=?",
            (norm,),
        ).fetchone()
        conn.close()
        if not row:
            return None
        return {
            "user_id": str(row[0] or ""),
            "username": str(row[1] or ""),
            "username_norm": str(row[2] or ""),
            "password_hash": str(row[3] or ""),
            "avatar_image": str(row[4] or ""),
            "banner_image": str(row[5] or ""),
            "created_at": float(row[6] or 0.0),
            "updated_at": float(row[7] or 0.0),
        }

    # Fetch account by user id.
    def get_account_by_user_id(self, user_id):
        """
        Get account by user id.
        
        This method implements the get account by user id step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        uid = str(user_id or "").strip()
        if not uid:
            return None
        conn = self._conn()
        row = conn.execute(
            "SELECT user_id, username, username_norm, password_hash, avatar_image, banner_image, created_at, updated_at FROM user_accounts WHERE user_id=?",
            (uid,),
        ).fetchone()
        conn.close()
        if not row:
            return None
        return {
            "user_id": str(row[0] or ""),
            "username": str(row[1] or ""),
            "username_norm": str(row[2] or ""),
            "password_hash": str(row[3] or ""),
            "avatar_image": str(row[4] or ""),
            "banner_image": str(row[5] or ""),
            "created_at": float(row[6] or 0.0),
            "updated_at": float(row[7] or 0.0),
        }

    # Update account avatar image URL/data URI.
    def set_account_avatar(self, user_id, avatar_image):
        """
        Set account avatar.
        
        This method implements the set account avatar step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        uid = str(user_id or "").strip()
        avatar = str(avatar_image or "").strip()
        if not uid:
            raise ValueError("user_id required")
        conn = self._conn()
        conn.execute(
            "UPDATE user_accounts SET avatar_image=?, updated_at=? WHERE user_id=?",
            (avatar, time.time(), uid),
        )
        updated = conn.total_changes > 0
        conn.commit()
        conn.close()
        return bool(updated)

    # Update account banner image URL/data URI.
    def set_account_banner(self, user_id, banner_image):
        """
        Set account banner.
        
        This method implements the set account banner step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        uid = str(user_id or "").strip()
        banner = str(banner_image or "").strip()
        if not uid:
            raise ValueError("user_id required")
        conn = self._conn()
        conn.execute(
            "UPDATE user_accounts SET banner_image=?, updated_at=? WHERE user_id=?",
            (banner, time.time(), uid),
        )
        updated = conn.total_changes > 0
        conn.commit()
        conn.close()
        return bool(updated)

    # Add like.
    def add_like(self, user_id, song_id):
        """
        Add like.
        
        This method implements the add like step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        conn = self._conn()
        # Keep explicit feedback consistent: a track cannot be both liked and disliked.
        conn.execute(
            "DELETE FROM user_dislikes WHERE user_id=? AND song_id=?",
            (user_id, str(song_id)),
        )
        conn.execute(
            "INSERT OR IGNORE INTO user_likes (user_id, song_id, liked_at) VALUES (?, ?, ?)",
            (user_id, str(song_id), time.time()),
        )
        conn.commit()
        conn.close()

    # Remove like.
    def remove_like(self, user_id, song_id):
        """
        Remove like.
        
        This method implements the remove like step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        conn = self._conn()
        conn.execute(
            "DELETE FROM user_likes WHERE user_id=? AND song_id=?", (user_id, str(song_id))
        )
        conn.commit()
        conn.close()

    # Add dislike.
    def add_dislike(self, user_id, song_id):
        """
        Add dislike.
        
        This method implements the add dislike step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        conn = self._conn()
        # Keep explicit feedback consistent: a track cannot be both liked and disliked.
        conn.execute(
            "DELETE FROM user_likes WHERE user_id=? AND song_id=?",
            (user_id, str(song_id)),
        )
        conn.execute(
            "INSERT OR IGNORE INTO user_dislikes (user_id, song_id, disliked_at) VALUES (?, ?, ?)",
            (user_id, str(song_id), time.time()),
        )
        conn.commit()
        conn.close()

    # Remove dislike.
    def remove_dislike(self, user_id, song_id):
        """
        Remove dislike.
        
        This method implements the remove dislike step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        conn = self._conn()
        conn.execute(
            "DELETE FROM user_dislikes WHERE user_id=? AND song_id=?",
            (user_id, str(song_id)),
        )
        conn.commit()
        conn.close()

    # Add play.
    def add_play(self, user_id, song_id):
        """
        Add play.
        
        This method implements the add play step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        conn = self._conn()
        conn.execute(
            "INSERT INTO user_plays (user_id, song_id, played_at) VALUES (?, ?, ?)",
            (user_id, str(song_id), time.time()),
        )
        conn.commit()
        conn.close()

    # Add skip event (next/prev button behavior).
    def add_skip_event(self, user_id, song_id, event="next", position_sec=None, duration_sec=None):
        """
        Add skip event.
        
        This method implements the add skip event step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        ev = str(event or "").strip().lower()
        if ev not in {"next", "prev"}:
            raise ValueError("event must be next or prev")
        sid = str(song_id or "").strip()
        if not sid:
            raise ValueError("song_id required")
        try:
            pos = float(position_sec) if position_sec is not None else None
        except Exception:
            pos = None
        try:
            dur = float(duration_sec) if duration_sec is not None else None
        except Exception:
            dur = None
        conn = self._conn()
        conn.execute(
            "INSERT INTO user_skip_events (user_id, song_id, event, position_sec, duration_sec, event_at) VALUES (?, ?, ?, ?, ?, ?)",
            (user_id, sid, ev, pos, dur, time.time()),
        )
        conn.commit()
        conn.close()

    # Get likes.
    def get_likes(self, user_id):
        """
        Get likes.
        
        This method implements the get likes step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        conn = self._conn()
        rows = conn.execute(
            "SELECT song_id FROM user_likes WHERE user_id=? ORDER BY liked_at DESC", (user_id,)
        ).fetchall()
        conn.close()
        return [r[0] for r in rows]

    # Get dislikes.
    def get_dislikes(self, user_id):
        """
        Get dislikes.
        
        This method implements the get dislikes step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        conn = self._conn()
        rows = conn.execute(
            "SELECT song_id FROM user_dislikes WHERE user_id=? ORDER BY disliked_at DESC",
            (user_id,),
        ).fetchall()
        conn.close()
        return [r[0] for r in rows]

    # Get plays.
    def get_plays(self, user_id, limit=50):
        """
        Get plays.
        
        This method implements the get plays step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        conn = self._conn()
        rows = conn.execute(
            "SELECT DISTINCT song_id FROM user_plays WHERE user_id=? ORDER BY played_at DESC LIMIT ?",
            (user_id, limit),
        ).fetchall()
        conn.close()
        return [r[0] for r in rows]

    # Get summarized skip/replay signals from recent skip events.
    def get_skip_summary(self, user_id, limit=600, early_threshold=0.35):
        """
        Get skip summary.
        
        This method implements the get skip summary step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        conn = self._conn()
        rows = conn.execute(
            "SELECT song_id, event, position_sec, duration_sec FROM user_skip_events WHERE user_id=? ORDER BY event_at DESC LIMIT ?",
            (user_id, int(limit or 600)),
        ).fetchall()
        conn.close()
        next_counts = {}
        prev_counts = {}
        early_next_counts = {}
        for raw_song_id, raw_event, raw_pos, raw_dur in rows:
            sid = str(raw_song_id or "").strip()
            ev = str(raw_event or "").strip().lower()
            if not sid or ev not in {"next", "prev"}:
                continue
            if ev == "prev":
                prev_counts[sid] = int(prev_counts.get(sid, 0)) + 1
                continue
            next_counts[sid] = int(next_counts.get(sid, 0)) + 1
            try:
                pos = float(raw_pos) if raw_pos is not None else None
                dur = float(raw_dur) if raw_dur is not None else None
            except Exception:
                pos, dur = None, None
            if pos is None or dur is None or dur <= 0:
                continue
            ratio = pos / max(1.0, dur)
            if ratio <= float(early_threshold):
                early_next_counts[sid] = int(early_next_counts.get(sid, 0)) + 1
        return {
            "next_counts": next_counts,
            "prev_counts": prev_counts,
            "early_next_counts": early_next_counts,
            "total_events": int(len(rows)),
        }

    # Get interaction count.
    def get_interaction_count(self, user_id):
        """
        Get interaction count.
        
        This method implements the get interaction count step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        likes = len(self.get_likes(user_id))
        plays = len(self.get_plays(user_id))
        dislikes = len(self.get_dislikes(user_id))
        playlist_tracks = len(self.get_all_playlist_track_ids(user_id, limit=None))
        skip_events = int((self.get_skip_summary(user_id, limit=300) or {}).get("total_events", 0))
        return likes + plays + dislikes + playlist_tracks + skip_events

    # Create playlist.
    def create_playlist(self, user_id, name, cover_image=None):
        """
        Create playlist.
        
        This method implements the create playlist step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        playlist_name = str(name or "").strip()
        if not playlist_name:
            raise ValueError("playlist name required")
        playlist_id = str(uuid.uuid4())
        now = time.time()
        cover = str(cover_image or "").strip()
        conn = self._conn()
        if self._playlist_name_exists(conn, user_id, playlist_name):
            conn.close()
            raise ValueError("playlist name already exists")
        conn.execute(
            "INSERT INTO user_playlists (playlist_id, user_id, name, cover_image, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
            (playlist_id, user_id, playlist_name, cover, now, now),
        )
        conn.commit()
        conn.close()
        return playlist_id

    # Rename playlist.
    def rename_playlist(self, user_id, playlist_id, name):
        """
        Execute rename playlist.
        
        This method implements the rename playlist step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        playlist_name = str(name or "").strip()
        if not playlist_name:
            raise ValueError("playlist name required")
        pid = str(playlist_id or "").strip()
        if not pid:
            raise ValueError("playlist_id required")
        conn = self._conn()
        exists = conn.execute(
            "SELECT 1 FROM user_playlists WHERE user_id=? AND playlist_id=?",
            (user_id, pid),
        ).fetchone()
        if not exists:
            conn.close()
            return False
        if self._playlist_name_exists(conn, user_id, playlist_name, exclude_playlist_id=pid):
            conn.close()
            raise ValueError("playlist name already exists")
        conn.execute(
            "UPDATE user_playlists SET name=?, updated_at=? WHERE user_id=? AND playlist_id=?",
            (playlist_name, time.time(), user_id, pid),
        )
        updated = conn.total_changes > 0
        conn.commit()
        conn.close()
        return bool(updated)

    # Set playlist cover image (data URL or URL).
    def set_playlist_cover(self, user_id, playlist_id, cover_image):
        """
        Set playlist cover.
        
        This method implements the set playlist cover step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        pid = str(playlist_id or "").strip()
        cover = str(cover_image or "").strip()
        conn = self._conn()
        conn.execute(
            "UPDATE user_playlists SET cover_image=?, updated_at=? WHERE user_id=? AND playlist_id=?",
            (cover, time.time(), user_id, pid),
        )
        updated = conn.total_changes > 0
        conn.commit()
        conn.close()
        return bool(updated)

    # Delete playlist.
    def delete_playlist(self, user_id, playlist_id):
        """
        Delete playlist.
        
        This method implements the delete playlist step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        pid = str(playlist_id or "")
        conn = self._conn()
        conn.execute(
            "DELETE FROM user_playlist_tracks WHERE user_id=? AND playlist_id=?",
            (user_id, pid),
        )
        conn.execute(
            "DELETE FROM user_playlists WHERE user_id=? AND playlist_id=?",
            (user_id, pid),
        )
        conn.commit()
        conn.close()

    # Add track to playlist.
    def add_playlist_track(self, user_id, playlist_id, song_id):
        """
        Add playlist track.
        
        This method implements the add playlist track step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        pid = str(playlist_id or "").strip()
        sid = str(song_id or "").strip()
        if not pid or not sid:
            raise ValueError("playlist_id and song_id required")
        conn = self._conn()
        now = time.time()
        exists = conn.execute(
            "SELECT 1 FROM user_playlists WHERE user_id=? AND playlist_id=?",
            (user_id, pid),
        ).fetchone()
        if not exists:
            conn.close()
            raise ValueError("playlist not found")
        before_changes = int(conn.total_changes or 0)
        conn.execute(
            "INSERT OR IGNORE INTO user_playlist_tracks (playlist_id, user_id, song_id, added_at) VALUES (?, ?, ?, ?)",
            (pid, user_id, sid, now),
        )
        added = int(conn.total_changes or 0) > before_changes
        if added:
            conn.execute(
                "UPDATE user_playlists SET updated_at=? WHERE user_id=? AND playlist_id=?",
                (now, user_id, pid),
            )
        conn.commit()
        conn.close()
        return added

    # Remove track from playlist.
    def remove_playlist_track(self, user_id, playlist_id, song_id):
        """
        Remove playlist track.
        
        This method implements the remove playlist track step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        pid = str(playlist_id or "").strip()
        sid = str(song_id or "").strip()
        conn = self._conn()
        conn.execute(
            "DELETE FROM user_playlist_tracks WHERE user_id=? AND playlist_id=? AND song_id=?",
            (user_id, pid, sid),
        )
        conn.execute(
            "UPDATE user_playlists SET updated_at=? WHERE user_id=? AND playlist_id=?",
            (time.time(), user_id, pid),
        )
        conn.commit()
        conn.close()

    # Clear playlist tracks.
    def clear_playlist(self, user_id, playlist_id):
        """
        Clear playlist.
        
        This method implements the clear playlist step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        pid = str(playlist_id or "").strip()
        conn = self._conn()
        conn.execute(
            "DELETE FROM user_playlist_tracks WHERE user_id=? AND playlist_id=?",
            (user_id, pid),
        )
        conn.execute(
            "UPDATE user_playlists SET updated_at=? WHERE user_id=? AND playlist_id=?",
            (time.time(), user_id, pid),
        )
        conn.commit()
        conn.close()

    # Get playlists for a user.
    def get_playlists(self, user_id):
        """
        Get playlists.
        
        This method implements the get playlists step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        conn = self._conn()
        rows = conn.execute(
            "SELECT playlist_id, name, cover_image, created_at, updated_at FROM user_playlists WHERE user_id=? ORDER BY updated_at DESC, created_at DESC",
            (user_id,),
        ).fetchall()
        conn.close()
        return [
            {
                "playlist_id": str(r[0]),
                "name": str(r[1] or ""),
                "cover_image": str(r[2] or ""),
                "created_at": float(r[3] or 0.0),
                "updated_at": float(r[4] or 0.0),
            }
            for r in rows
        ]

    # Get track ids in a playlist.
    def get_playlist_track_ids(self, user_id, playlist_id, limit=None):
        """
        Get playlist track ids.
        
        This method implements the get playlist track ids step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        pid = str(playlist_id or "").strip()
        conn = self._conn()
        sql = (
            "SELECT song_id FROM user_playlist_tracks WHERE user_id=? AND playlist_id=? "
            "ORDER BY added_at DESC"
        )
        params = [user_id, pid]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        rows = conn.execute(sql, tuple(params)).fetchall()
        conn.close()
        return [r[0] for r in rows]

    # Get all playlist track ids across a user (deduped, newest first).
    def get_all_playlist_track_ids(self, user_id, limit=200):
        """
        Get all playlist track ids.
        
        This method implements the get all playlist track ids step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        conn = self._conn()
        sql = (
            "SELECT song_id FROM user_playlist_tracks WHERE user_id=? "
            "ORDER BY added_at DESC"
        )
        params = [user_id]
        if limit is not None:
            sql += " LIMIT ?"
            params.append(int(limit))
        rows = conn.execute(sql, tuple(params)).fetchall()
        conn.close()
        seen = set()
        out = []
        for row in rows:
            sid = str(row[0] or "").strip()
            if not sid or sid in seen:
                continue
            seen.add(sid)
            out.append(sid)
        return out

    # Get profile.
    def get_profile(self, user_id):
        """
        Get profile.
        
        This method implements the get profile step for this module.
        It is used to keep the broader workflow readable and easier to maintain.
        """
        playlists = self.get_playlists(user_id)
        skip_summary = self.get_skip_summary(user_id, limit=300)
        account = self.get_account_by_user_id(user_id) or {}
        return {
            "user_id": user_id,
            "username": str(account.get("username") or ""),
            "avatar_image": str(account.get("avatar_image") or ""),
            "banner_image": str(account.get("banner_image") or ""),
            "likes": self.get_likes(user_id),
            "dislikes": self.get_dislikes(user_id),
            "recent_plays": self.get_plays(user_id, limit=20),
            "playlist_count": len(playlists),
            "playlist_track_count": len(self.get_all_playlist_track_ids(user_id, limit=None)),
            "skip_next_count": int(sum((skip_summary or {}).get("next_counts", {}).values())),
            "skip_prev_count": int(sum((skip_summary or {}).get("prev_counts", {}).values())),
            "skip_early_count": int(sum((skip_summary or {}).get("early_next_counts", {}).values())),
            "interaction_count": self.get_interaction_count(user_id),
        }