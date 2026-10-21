"""SQLAlchemy 中的记录管理层实现。

该管理层使用 SQLAlchemy 来跟踪插入或更新的记录。

Currently, this layer only works with SQLite; hopwever, should be adaptable
to other SQL implementations with minimal effort.

Currently, includes an implementation that uses SQLAlchemy which should
allow it to work with a variety of SQL as a backend.

* 每个键都关联一个 updated_at 字段。
* 该字段在键被更新时自动更新。
* 可根据 updated at 字段列出键。
* 可删除键。
"""

import contextlib
import decimal
import uuid
from collections.abc import AsyncGenerator, Generator, Sequence
from typing import Any

from langchain_core.indexing import RecordManager
from sqlalchemy import (
    Column,
    Float,
    Index,
    String,
    UniqueConstraint,
    and_,
    create_engine,
    delete,
    select,
    text,
)
from sqlalchemy.engine import URL, Engine
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    create_async_engine,
)
from sqlalchemy.orm import Query, Session, declarative_base, sessionmaker

try:
    from sqlalchemy.ext.asyncio import async_sessionmaker
except ImportError:
        # dummy for sqlalchemy < 2
    async_sessionmaker = type("async_sessionmaker", (type,), {})  # type: ignore[assignment,misc]

Base = declarative_base()


class UpsertionRecord(Base):  # type: ignore[valid-type,misc]
    """用于记录键上次更新时间的表。"""

        # 注意：
        # 在修改此表之前，请确定是否
        # 应为此表创建迁移，以确保
        # 用户不会遭遇数据丢失。
    __tablename__ = "upsertion_record"

    uuid = Column(
        String,
        index=True,
        default=lambda: str(uuid.uuid4()),
        primary_key=True,
        nullable=False,
    )
    key = Column(String, index=True)
        # 使用非规范化表示来处理 `namespace` 属性。
        # 如果需要，此属性稍后可提取到独立的 Collection 表中。
        # 与最后一次记录插入相关的 timestamp。
    namespace = Column(String, index=True, nullable=False)
    group_id = Column(String, index=True, nullable=True)

        # * SQLite 特定实现，可根据方言进行更改。
    updated_at = Column(Float, index=True)

    __table_args__ = (
        UniqueConstraint("key", "namespace", name="uix_key_namespace"),
        Index("ix_key_namespace", "key", "namespace"),
    )


class SQLRecordManager(RecordManager):
    """基于 SQLAlchemy 的记录管理器实现。"""

    def __init__(
        self,
        namespace: str,
        *,
        engine: Engine | AsyncEngine | None = None,
        db_url: None | str | URL = None,
        engine_kwargs: dict[str, Any] | None = None,
        async_mode: bool = False,
    ) -> None:
        """初始化 SQLRecordManager。

        This class serves as a manager persistence layer that uses an SQL
        backend to track upserted records. You should specify either a `db_url`
        to create an engine or provide an existing engine.

        Args:
            namespace: The namespace associated with this record manager.
            engine: An already existing SQL Alchemy engine.
            db_url: A database connection string used to create an SQL Alchemy engine.
            engine_kwargs: Additional keyword arguments to be passed when creating the
                engine.
            async_mode: Whether to create an async engine. Driver should support async
                operations. It only applies if `db_url` is provided.

        Raises:
            ValueError: 如果同时提供了 db_url 和 engine，或两者均未提供。
            AssertionError: 在引擎配置过程中发生意外情况时。
        """
        super().__init__(namespace=namespace)
        if db_url is None and engine is None:
            msg = "Must specify either db_url or engine"
            raise ValueError(msg)

        if db_url is not None and engine is not None:
            msg = "Must specify either db_url or engine, not both"
            raise ValueError(msg)

        _engine: Engine | AsyncEngine
        if db_url:
            if async_mode:
                _engine = create_async_engine(db_url, **(engine_kwargs or {}))
            else:
                _engine = create_engine(db_url, **(engine_kwargs or {}))
        elif engine:
            _engine = engine

        else:
            msg = "Something went wrong with configuration of engine."
            raise AssertionError(msg)

        _session_factory: sessionmaker[Session] | async_sessionmaker[AsyncSession]
        if isinstance(_engine, AsyncEngine):
            _session_factory = async_sessionmaker(bind=_engine)
        else:
            _session_factory = sessionmaker(bind=_engine)

        self.engine = _engine
        self.dialect = _engine.dialect.name
        self.session_factory = _session_factory

    def create_schema(self) -> None:
        """创建数据库模式。"""
        if isinstance(self.engine, AsyncEngine):
            msg = "This method is not supported for async engines."
            raise AssertionError(msg)  # noqa: TRY004

        Base.metadata.create_all(self.engine)

    async def acreate_schema(self) -> None:
        """创建数据库模式。"""
        if not isinstance(self.engine, AsyncEngine):
            msg = "This method is not supported for sync engines."
            raise AssertionError(msg)  # noqa: TRY004

        async with self.engine.begin() as session:
            await session.run_sync(Base.metadata.create_all)

    @contextlib.contextmanager
    def _make_session(self) -> Generator[Session, None, None]:
        """创建会话并在使用后关闭。"""
        if isinstance(self.session_factory, async_sessionmaker):
            msg = "This method is not supported for async engines."
            raise AssertionError(msg)  # noqa: TRY004

        session = self.session_factory()
        try:
            yield session
        finally:
            session.close()

    @contextlib.asynccontextmanager
    async def _amake_session(self) -> AsyncGenerator[AsyncSession, None]:
        """创建会话并在使用后关闭。"""
        if not isinstance(self.session_factory, async_sessionmaker):
            msg = "This method is not supported for sync engines."
            raise AssertionError(msg)  # noqa: TRY004

        async with self.session_factory() as session:
            yield session

    def get_time(self) -> float:
        """以时间戳形式获取当前服务器时间。

        Please note it's critical that time is obtained from the server since
        we want a monotonic clock.
        """
        with self._make_session() as session:
                        # * 对于 SQLite，与 unixepoch 不同，它可兼容较旧版本的 SQLite。
                        # * 对于 SQLite，与 unixepoch 不同，它可在较旧版本的 SQLite 上运行。
            # ----
                        # julianday('now'): 当前日期和时间的儒略日数。
                        # 儒略日是从一个参考日期（儒略日数为 0）开始的连续天数计数。
                        # 儒略日是从一个参考日期（儒略日数为 0）开始的连续天数计数。
                        # 2440587.5 - 常量，表示 1970 年 1 月 1 日的儒略日数
                        # 86400.0 - 常量，表示一天中的秒数（24 小时 * 60 分钟 * 60 秒）
                        # 86400.0 - 常量，表示一天中的秒数（24 小时 * 60 分钟 * 60 秒）
            if self.dialect == "sqlite":
                query = text("SELECT (julianday('now') - 2440587.5) * 86400.0;")
            elif self.dialect == "postgresql":
                query = text("SELECT EXTRACT (EPOCH FROM CURRENT_TIMESTAMP);")
            else:
                msg = f"Not implemented for dialect {self.dialect}"
                raise NotImplementedError(msg)

            dt = session.execute(query).scalar()
            if isinstance(dt, decimal.Decimal):
                dt = float(dt)
            if not isinstance(dt, float):
                msg = f"Unexpected type for datetime: {type(dt)}"
                raise AssertionError(msg)  # noqa: TRY004
            return dt

    async def aget_time(self) -> float:
        """以时间戳形式获取当前服务器时间。

        Please note it's critical that time is obtained from the server since
        we want a monotonic clock.
        """
        async with self._amake_session() as session:
                        # * SQLite 特定实现，可根据方言进行更改。
                        # * 对于 SQLite，与 unixepoch 不同，它可在较旧版本的 SQLite 上使用。
            # ----
                        # julianday('now'): 当前日期和时间的儒略日数。
                        # 儒略日是从一个参考日期（儒略日数为 0）开始的连续天数计数。
                        # 儒略日是从一个参考日期（儒略日数为 0）开始的连续天数计数。
                        # 2440587.5 - 常量，表示 1970 年 1 月 1 日的儒略日数
                        # 86400.0 - 常量，表示一天中的秒数（24 小时 * 60 分钟 * 60 秒）
                        # 86400.0 - 常量，表示一天中的秒数（24 小时 * 60 分钟 * 60 秒）
            if self.dialect == "sqlite":
                query = text("SELECT (julianday('now') - 2440587.5) * 86400.0;")
            elif self.dialect == "postgresql":
                query = text("SELECT EXTRACT (EPOCH FROM CURRENT_TIMESTAMP);")
            else:
                msg = f"Not implemented for dialect {self.dialect}"
                raise NotImplementedError(msg)

            dt = (await session.execute(query)).scalar_one_or_none()

            if isinstance(dt, decimal.Decimal):
                dt = float(dt)
            if not isinstance(dt, float):
                msg = f"Unexpected type for datetime: {type(dt)}"
                raise AssertionError(msg)  # noqa: TRY004
            return dt

    def update(
        self,
        keys: Sequence[str],
        *,
        group_ids: Sequence[str | None] | None = None,
        time_at_least: float | None = None,
    ) -> None:
        """将记录插入或更新到 SQLite 数据库中。"""
        if group_ids is None:
            group_ids = [None] * len(keys)

        if len(keys) != len(group_ids):
            msg = (
                f"Number of keys ({len(keys)}) does not match number of "
                f"group_ids ({len(group_ids)})"
            )
            raise ValueError(msg)

                # 从服务器获取当前时间。
                # 这会增加一次额外的往返通信，但如果批处理大小足够大，影响不大。
                # 这会增加一次额外的往返通信，但如果批处理大小足够大，影响不大。
                # 在此获取时间有助于我们将其与 time_at_least 进行比较
                # 并在存在时间同步问题时抛出错误。
                # 这里我们只是格外小心，以最小化因错误删除记录而导致的数据丢失风险。
                # 这里我们只是格外小心，以最小化因错误删除记录而导致的数据丢失风险。
        update_time = self.get_time()

        if time_at_least and update_time < time_at_least:
                        # 防范时间同步问题
            msg = f"Time sync issue: {update_time} < {time_at_least}"
            raise AssertionError(msg)

        records_to_upsert = [
            {
                "key": key,
                "namespace": self.namespace,
                "updated_at": update_time,
                "group_id": group_id,
            }
            for key, group_id in zip(keys, group_ids, strict=False)
        ]

        with self._make_session() as session:
            if self.dialect == "sqlite":
                from sqlalchemy.dialects.sqlite import Insert as SqliteInsertType
                from sqlalchemy.dialects.sqlite import insert as sqlite_insert

                                # 注意：使用 SQLite insert 以使 on_conflict_do_update 生效。
                                # 这段代码需要稍作泛化，以便支持更多方言。
                sqlite_insert_stmt: SqliteInsertType = sqlite_insert(
                    UpsertionRecord,
                ).values(records_to_upsert)
                stmt = sqlite_insert_stmt.on_conflict_do_update(
                    [UpsertionRecord.key, UpsertionRecord.namespace],
                    set_={
                        "updated_at": sqlite_insert_stmt.excluded.updated_at,
                        "group_id": sqlite_insert_stmt.excluded.group_id,
                    },
                )
            elif self.dialect == "postgresql":
                from sqlalchemy.dialects.postgresql import Insert as PgInsertType
                from sqlalchemy.dialects.postgresql import insert as pg_insert

                                # 注意：使用 postgresql insert 以使 on_conflict_do_update 生效。
                                # 这段代码需要稍作泛化，以便支持更多方言。
                pg_insert_stmt: PgInsertType = pg_insert(UpsertionRecord).values(
                    records_to_upsert,
                )
                stmt = pg_insert_stmt.on_conflict_do_update(  # type: ignore[assignment]
                    constraint="uix_key_namespace",  # 约束名称
                    set_={
                        "updated_at": pg_insert_stmt.excluded.updated_at,
                        "group_id": pg_insert_stmt.excluded.group_id,
                    },
                )
            else:
                msg = f"Unsupported dialect {self.dialect}"
                raise NotImplementedError(msg)

            session.execute(stmt)
            session.commit()

    async def aupdate(
        self,
        keys: Sequence[str],
        *,
        group_ids: Sequence[str | None] | None = None,
        time_at_least: float | None = None,
    ) -> None:
        """将记录插入或更新到 SQLite 数据库中。"""
        if group_ids is None:
            group_ids = [None] * len(keys)

        if len(keys) != len(group_ids):
            msg = (
                f"Number of keys ({len(keys)}) does not match number of "
                f"group_ids ({len(group_ids)})"
            )
            raise ValueError(msg)

                # 从服务器获取当前时间。
                # 这会增加一次额外的往返通信，但如果批处理大小足够大，影响不大。
                # 这会增加一次额外的往返通信，但如果批处理大小足够大，影响不大。
                # 在此获取时间有助于我们将其与 time_at_least 进行比较
                # 并在存在时间同步问题时抛出错误。
                # 这里我们只是格外小心，以最小化因错误删除记录而导致的数据丢失风险。
                # 这里我们只是格外小心，以最小化因错误删除记录而导致的数据丢失风险。
        update_time = await self.aget_time()

        if time_at_least and update_time < time_at_least:
                        # 防范时间同步问题
            msg = f"Time sync issue: {update_time} < {time_at_least}"
            raise AssertionError(msg)

        records_to_upsert = [
            {
                "key": key,
                "namespace": self.namespace,
                "updated_at": update_time,
                "group_id": group_id,
            }
            for key, group_id in zip(keys, group_ids, strict=False)
        ]

        async with self._amake_session() as session:
            if self.dialect == "sqlite":
                from sqlalchemy.dialects.sqlite import Insert as SqliteInsertType
                from sqlalchemy.dialects.sqlite import insert as sqlite_insert

                                # 注意：使用 SQLite insert 以使 on_conflict_do_update 生效。
                                # 这段代码需要稍作泛化，以便支持更多方言。
                sqlite_insert_stmt: SqliteInsertType = sqlite_insert(
                    UpsertionRecord,
                ).values(records_to_upsert)
                stmt = sqlite_insert_stmt.on_conflict_do_update(
                    [UpsertionRecord.key, UpsertionRecord.namespace],
                    set_={
                        "updated_at": sqlite_insert_stmt.excluded.updated_at,
                        "group_id": sqlite_insert_stmt.excluded.group_id,
                    },
                )
            elif self.dialect == "postgresql":
                from sqlalchemy.dialects.postgresql import Insert as PgInsertType
                from sqlalchemy.dialects.postgresql import insert as pg_insert

                                # 注意：使用 SQLite insert 以使 on_conflict_do_update 生效。
                                # 这段代码需要稍作泛化，以便支持更多方言。
                pg_insert_stmt: PgInsertType = pg_insert(UpsertionRecord).values(
                    records_to_upsert,
                )
                stmt = pg_insert_stmt.on_conflict_do_update(  # type: ignore[assignment]
                    constraint="uix_key_namespace",  # 约束名称
                    set_={
                        "updated_at": pg_insert_stmt.excluded.updated_at,
                        "group_id": pg_insert_stmt.excluded.group_id,
                    },
                )
            else:
                msg = f"Unsupported dialect {self.dialect}"
                raise NotImplementedError(msg)

            await session.execute(stmt)
            await session.commit()

    def exists(self, keys: Sequence[str]) -> list[bool]:
        """检查给定的键是否存在于 SQLite 数据库中。"""
        session: Session
        with self._make_session() as session:
            filtered_query: Query = session.query(UpsertionRecord.key).filter(
                and_(
                    UpsertionRecord.key.in_(keys),
                    UpsertionRecord.namespace == self.namespace,
                ),
            )
            records = filtered_query.all()
        found_keys = {r.key for r in records}
        return [k in found_keys for k in keys]

    async def aexists(self, keys: Sequence[str]) -> list[bool]:
        """检查给定的键是否存在于 SQLite 数据库中。"""
        async with self._amake_session() as session:
            records = (
                (
                    await session.execute(
                        select(UpsertionRecord.key).where(
                            and_(
                                UpsertionRecord.key.in_(keys),
                                UpsertionRecord.namespace == self.namespace,
                            ),
                        ),
                    )
                )
                .scalars()
                .all()
            )
        found_keys = set(records)
        return [k in found_keys for k in keys]

    def list_keys(
        self,
        *,
        before: float | None = None,
        after: float | None = None,
        group_ids: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> list[str]:
        """根据提供的日期范围列出 SQLite 数据库中的记录。"""
        session: Session
        with self._make_session() as session:
            query: Query = session.query(UpsertionRecord).filter(
                UpsertionRecord.namespace == self.namespace,
            )

            if after:
                query = query.filter(UpsertionRecord.updated_at > after)
            if before:
                query = query.filter(UpsertionRecord.updated_at < before)
            if group_ids:
                query = query.filter(UpsertionRecord.group_id.in_(group_ids))

            if limit:
                query = query.limit(limit)
            records = query.all()
        return [r.key for r in records]

    async def alist_keys(
        self,
        *,
        before: float | None = None,
        after: float | None = None,
        group_ids: Sequence[str] | None = None,
        limit: int | None = None,
    ) -> list[str]:
        """根据提供的日期范围列出 SQLite 数据库中的记录。"""
        session: AsyncSession
        async with self._amake_session() as session:
            query: Query = select(UpsertionRecord.key).filter(  # type: ignore[assignment]
                UpsertionRecord.namespace == self.namespace,
            )

                        # mypy 无法识别 .all() 或 .filter()
            if after:
                query = query.filter(UpsertionRecord.updated_at > after)
            if before:
                query = query.filter(UpsertionRecord.updated_at < before)
            if group_ids:
                query = query.filter(UpsertionRecord.group_id.in_(group_ids))

            if limit:
                query = query.limit(limit)
            records = (await session.execute(query)).scalars().all()
        return list(records)

    def delete_keys(self, keys: Sequence[str]) -> None:
        """从 SQLite 数据库中删除记录。"""
        session: Session
        with self._make_session() as session:
            filtered_query: Query = session.query(UpsertionRecord).filter(
                and_(
                    UpsertionRecord.key.in_(keys),
                    UpsertionRecord.namespace == self.namespace,
                ),
            )

            filtered_query.delete()
            session.commit()

    async def adelete_keys(self, keys: Sequence[str]) -> None:
        """从 SQLite 数据库中删除记录。"""
        async with self._amake_session() as session:
            await session.execute(
                delete(UpsertionRecord).where(
                    and_(
                        UpsertionRecord.key.in_(keys),
                        UpsertionRecord.namespace == self.namespace,
                    ),
                ),
            )

            await session.commit()
