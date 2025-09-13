# -*- coding: utf-8 -*-
"""
    Created on Tue Aug 29 17:49:23 2025

    @author: Johnson
"""

from contextlib import asynccontextmanager
from typing import Optional, Tuple, Pattern, List
import pandas as pd
import logging
import asyncpg
import asyncio
import os
import re

"""
    In this project, using asyncpg is not strictly necessary,
    as it could just as well be implemented with psycopg in a synchronous manner.

    The database here does not require concurrency or parallel processing,
    and if such patterns appear, they are intentional rather than required.

    The choice of asyncpg is mainly to prepare for future projects,
    so that when an asynchronous architecture becomes essential, the groundwork will already be in place.
"""

"""
    load_dotenv() is invoked globally by the main program,
    so when the main program imports this module, it does not need to call load_dotenv() again.
        # from dotenv import load_dotenv
        # load_dotenv()
"""

CONSOLE_LOG = logging.getLogger("Console_log")

class MyPsql:

    def __init__(self):

        self._conn: Optional[asyncpg.Connection] = None
        self._pool: Optional[asyncpg.Pool]       = None
        self._init_lock                          = asyncio.Lock()
        self._initialized: bool                  = False
        self._min_size: int                      = 1
        self._max_size: int                      = 5
        self._target_db: str                     = os.getenv("TARGET_DB")
        self._target_user: str                   = os.getenv("TARGET_USER")
        self._host: str                          = os.getenv("HOST")
        self._target_pwd: str                    = os.getenv("TARGET_PWD")
        self._target_sch: str                    = os.getenv("TARGET_SCHEMA")
        self._target_tb: str                     = os.getenv("TARGET_TB")

    async def _ensure_initialized(self) -> None:

        try:
            """
                Double-check "self._initialized" inside the lock to prevent race conditions.
                This ensures that even if multiple coroutines try to acquire connectionsat the same time,
                only one of them will perform the pool initialization.
            """
            if self._initialized:
                return

            async with self._init_lock:
                if self._initialized:
                    return

                await self._checking_sql()
                await self._connect_pool()
                self._initialized = True

        except Exception as e:
            CONSOLE_LOG.error(f"Ensure initialized fail : {e}")

    async def _checking_sql(self) -> None:

        """
            Use superuser privileges to verify the existence of the target database and user,
            performing basic initialization tasks.
            User privileges are scoped to avoid unnecessary operations outside of their intended responsibilities.

            NOTE: 
                Within this project, such an approach is not strictly required;
                it is implemented primarily as preparation for future projects.
        """
        try:
            await self._supperuser_conn_pdb()
            await self._supperuser_switch_conn_tdb()

        except Exception as e:
            CONSOLE_LOG.error(f"Checking sql fail : {e}")

        finally:
            await self._conn_close()

    async def _supperuser_conn_pdb(self) -> None:

        try:
            await self._connect_to(obj = os.getenv("DB_USER"))
            await self._ensure_target_db_and_user()

        except Exception as e:
            CONSOLE_LOG.error(f"Supperuser conn pdb fail : {e}")

    async def _supperuser_switch_conn_tdb(self) -> None:

        try:
            await self._connect_to(obj = os.getenv("TARGET_TB"))
            await self._target_schema_exists()
            await self._target_table_exists()
            await self._grant_user()
            await self._set_trigger()

        except Exception as e:
            CONSOLE_LOG.error(f"Supperuser conn pdb fail : {e}")

    async def _connect_pool(self) -> None:

        try:
            self._pool = await asyncpg.create_pool(
                    dsn                              = await self._connect_to(),
                    min_size                         = self._min_size,
                    max_size                         = self._max_size,
                    timeout                          = 30,
                    max_inactive_connection_lifetime = 300,
                    statement_cache_size             = 10
                )

        except Exception as e:
            CONSOLE_LOG.error(f"Connect pool fail : {e}")

    async def _connect_to(self, obj: Optional[str] = None) -> Optional[None]:

        try:
            if obj == os.getenv("DB_USER"):
                user, password, database = self._env_var_parsing()
                login_url: str = f"postgresql://{user}:{password}@{self._host}:5432/{database}"

            elif obj == os.getenv("TARGET_TB"):
                user, password, _ = self._env_var_parsing()
                login_url: str = f"postgresql://{user}:{password}@{self._host}:5432/{self._target_db}"

            else:
                """
                    This way is used for user.
                """
                login_url: str = f"postgresql://{self._target_user}:{self._target_pwd}@{self._host}:5432/{self._target_db}"
                return login_url

            self._conn = await asyncpg.connect(login_url)

        except ValueError as ve:
            CONSOLE_LOG.error(ve)
        except Exception as e:
            CONSOLE_LOG.error(f"Connect to fail : {e}")

    def _env_var_parsing(self) -> Tuple[str, str, str]:

        user: str     = os.getenv("DB_USER")
        password: str = os.getenv("DB_PWD")
        database: str = os.getenv("DB_NAME")

        if not all((user, password, database)):
            raise ValueError("Please input database info in \".env\".")

        return user, password, database

    async def _ensure_target_db_and_user(self) -> None:

        await self._target_db_exists()
        await self._target_user_exists()

    async def _target_db_exists(self) -> None:

        try:
            tdb_exists: bool = await self._conn.fetchval(f"select * from pg_database \
                                                            where datname = '{self._target_db}';") is not None

            if not tdb_exists:
                CONSOLE_LOG.info(f"{self._target_db} does not exist, create...")
                await self._conn.execute(f"create database \"{self._target_db}\"")

        except Exception as e:
            CONSOLE_LOG.error(f"Target db exists fail : {e}")

    async def _target_user_exists(self) -> None:

        try:
            tuser_exists: bool = await self._conn.fetchval(f"select * from pg_roles \
                                                            where rolname = '{self._target_user}';") is not None

            if not tuser_exists:
                CONSOLE_LOG.info(f"{self._target_user} does not exist, create...")
                await self._conn.execute(f"create role \"{self._target_user}\" with login password '{self._target_pwd}' \
                                            nosuperuser nocreatedb nocreaterole noinherit")

        except Exception as e:
            CONSOLE_LOG.error(f"Target user exists fail : {e}")

    async def _target_schema_exists(self) -> None:

        try:
            tsch_exists: bool = await self._conn.fetchval(f"select to_regnamespace('{self._target_sch}');") is not None

            if not tsch_exists:
                await self._conn.execute(f"create schema if not exists \"{self._target_sch}\" authorization \"{self._target_user}\"")
                CONSOLE_LOG.info(f"Schema {self._target_sch} ensured (owner = {self._target_user}).")

        except Exception as e:
            CONSOLE_LOG.error(f"Target schema exists fail : {e}")

    async def _target_table_exists(self) -> None:

        try:
            ttable_exists: bool = await self._conn.fetchval(f"select to_regclass('{self._target_sch}.{self._target_tb}')") is not None

            if not ttable_exists:
                await self._conn.execute(f"""
                                        create table if not exists \"{self._target_sch}\".\"{self._target_tb}\" (
                                                id          smallint primary key generated by default as identity,
                                                term        varchar(10) not null check (term ~ '^[0-9]{{3}}-[12]$'),
                                                time        varchar(15) not null,
                                                "Mon"       varchar(100),
                                                "Tue"       varchar(100),
                                                "Wed"       varchar(100),
                                                "Thr"       varchar(100),
                                                "Fri"       varchar(100),
                                                created_at  timestamptz not null default now(),
                                                updated_at  timestamptz not null default now(),
                                                is_del      bool default false,
                                                constraint  unique_term_time unique (term, time)
                                        );""")

                await self._conn.execute(f"alter table \"{self._target_sch}\".\"{self._target_tb}\" owner to \"{self._target_user}\";")
                CONSOLE_LOG.info(f"Table {self._target_sch}.{self._target_tb} ensured (owner = {self._target_user})")

        except Exception as e:
            CONSOLE_LOG.error(f"Target table exists fail : {e}")

    async def _grant_user(self) -> None:

        try:
            await self._conn.execute(f"grant connect on database \"{self._target_db}\" to \"{self._target_user}\"")
            CONSOLE_LOG.info(f"Granted privileges to {self._target_user}")

        except Exception as e:
            CONSOLE_LOG.error(f"Grant user fail : {e}")

    async def _set_trigger(self) -> None:

        """
            This function _set_trigger() ensures that whenever a table is updated,
            the updated_at column is automatically set to the current time,
            without requiring manual handling.
        """
        try:
            await self._conn.execute(f"""
                create or replace function "{self._target_sch}".set_updated_at()
                returns trigger as $$
                begin
                    new.updated_at := now();
                    return new;
                end;
                $$ language plpgsql;

                drop trigger if exists trg_set_updated_at on "{self._target_sch}"."{self._target_tb}";
                
                create trigger trg_set_updated_at
                before update on "{self._target_sch}"."{self._target_tb}"
                for each row when (old is distinct from new)
                execute function "{self._target_sch}".set_updated_at();
            """)
            CONSOLE_LOG.info("Set trigger success.")

        except Exception as e:
            CONSOLE_LOG.error(f"Set trigger fail : {e}")

    async def _conn_close(self):

        if self._conn:
            await self._conn.close()
            self._conn = None

    @asynccontextmanager
    async def _transaction(self, isolation: str = "read_committed"):

        await self._ensure_initialized()

        async with self._pool.acquire() as conn:
            tx = conn.transaction(isolation = isolation)
            await tx.start()

            try:
                yield conn

            except Exception as e:
                CONSOLE_LOG.error(f"Transaction rollback due to error: {e}")
                await tx.rollback()
                raise

            else:
                CONSOLE_LOG.info("Transaction committed successfully.")
                await tx.commit()

    ################################################################################
    """
        The following functions form the user-facing API and may be called directly.
    """
    ################################################################################

    async def close(self):
        if self._pool:
            await self._pool.close()
            self._pool = None

    async def upsert_sql(self, academic_term: str, courses: Tuple[Tuple[str]]):

        try:
            sql: str = f"""
                            insert into {self._target_sch}.{self._target_tb}
                            (term, time, "Mon", "Tue", "Wed", "Thr", "Fri")
                            values ($1, $2, $3, $4, $5, $6, $7)
                            on conflict (term, time) 
                            do update set
                                "Mon" = excluded."Mon",
                                "Tue" = excluded."Tue", 
                                "Wed" = excluded."Wed",
                                "Thr" = excluded."Thr",
                                "Fri" = excluded."Fri",
                                updated_at = now()
                            where (
                                "{self._target_sch}"."{self._target_tb}"."Mon" is distinct from excluded."Mon" or
                                "{self._target_sch}"."{self._target_tb}"."Tue" is distinct from excluded."Tue" or
                                "{self._target_sch}"."{self._target_tb}"."Wed" is distinct from excluded."Wed" or
                                "{self._target_sch}"."{self._target_tb}"."Thr" is distinct from excluded."Thr" or
                                "{self._target_sch}"."{self._target_tb}"."Fri" is distinct from excluded."Fri"
                            )
                        """

            async with self._transaction() as conn:
                await conn.executemany(sql, courses)

            CONSOLE_LOG.info(f"Upsert success for term {academic_term}: {len(courses)} rows.")

        except Exception as e:
            CONSOLE_LOG.error(f"Upsert sql fail : {e}")

    async def fetch_sql(self) -> pd.DataFrame:
        
        try:
            sql: str = f"""
                            select unnest(array["Mon", "Tue", "Wed", "Thr", "Fri"])
                            from \"{self._target_sch}\".\"{self._target_tb}\"
                        """
            async with self._transaction() as conn:
                rows = await conn.fetch(sql)

            pattern: Pattern[str]         = re.compile(r'\(.{1,3}\)')
            courses: List[str]            = [pattern.sub('', row[0]).strip() for row in rows]
            
            counts_courses: pd.DataFrame  = pd.Series(courses).value_counts().reset_index()
            return counts_courses

        except Exception as e:
            CONSOLE_LOG.error(f"Fetch sql fail : {e}")