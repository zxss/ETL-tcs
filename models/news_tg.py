"""
Схема хранения новостей из публичных Telegram-каналов.

Отдельная схема `news`, а не отдельная база: изоляция та же (свои права,
`pg_dump -n news` для отдельного бэкапа), но соединение и пул общие, а новости
остаются джойнимыми со свечами из public — ради чего их и собирают.

ВАЖНО: это контур СБОРА, а не торговли. Ничто здесь не участвует в скоринге,
отборе бумаг и постановке заявок. Прошлый новостной контур был сознательно
вынесен из пайплайна (contrib/experimental_news), и это решение не отменяется:
данные складываются, выводы из них человек делает сам.
"""

CREATE_NEWS_SCHEMA_SQL = """
CREATE SCHEMA IF NOT EXISTS news;
"""

CREATE_TG_POSTS_SQL = """
CREATE TABLE IF NOT EXISTS news.tg_posts (
    id          BIGSERIAL PRIMARY KEY,
    channel     VARCHAR(64)  NOT NULL,
    message_id  BIGINT       NOT NULL,
    posted_at   TIMESTAMPTZ  NOT NULL,
    text        TEXT,
    views       INTEGER,
    links       TEXT[],
    has_media   BOOLEAN      NOT NULL DEFAULT FALSE,
    fetched_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMPTZ  NOT NULL DEFAULT NOW(),
    UNIQUE (channel, message_id)
);
CREATE INDEX IF NOT EXISTS idx_tg_posts_posted   ON news.tg_posts (posted_at DESC);
CREATE INDEX IF NOT EXISTS idx_tg_posts_ch_posted ON news.tg_posts (channel, posted_at DESC);
"""

# Полнотекстовый поиск по русской морфологии: без него любой отбор новостей по
# тикеру или теме вырождается в ILIKE '%...%' с полным сканом таблицы.
CREATE_TG_POSTS_FTS_SQL = """
CREATE INDEX IF NOT EXISTS idx_tg_posts_fts
    ON news.tg_posts USING GIN (to_tsvector('russian', coalesce(text, '')));
"""

# Журнал опросов. Нужен, чтобы молчащий сбор был отличим от молчащего канала:
# без него «новостей нет» и «парсер сломался» выглядят одинаково.
CREATE_TG_FETCH_RUNS_SQL = """
CREATE TABLE IF NOT EXISTS news.tg_fetch_runs (
    id            BIGSERIAL PRIMARY KEY,
    channel       VARCHAR(64) NOT NULL,
    started_at    TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    finished_at   TIMESTAMPTZ,
    pages         INTEGER     NOT NULL DEFAULT 0,
    posts_seen    INTEGER     NOT NULL DEFAULT 0,
    posts_new     INTEGER     NOT NULL DEFAULT 0,
    posts_updated INTEGER     NOT NULL DEFAULT 0,
    http_status   INTEGER,
    ok            BOOLEAN     NOT NULL DEFAULT FALSE,
    error         TEXT
);
CREATE INDEX IF NOT EXISTS idx_tg_fetch_runs_started
    ON news.tg_fetch_runs (channel, started_at DESC);
"""

ALL_NEWS_DDL = (CREATE_NEWS_SCHEMA_SQL, CREATE_TG_POSTS_SQL,
                CREATE_TG_POSTS_FTS_SQL, CREATE_TG_FETCH_RUNS_SQL)

# Правка поста в Telegram меняет текст при том же message_id, поэтому DO UPDATE,
# а не DO NOTHING. updated_at двигается только при реальном изменении ТЕКСТА:
# просмотры растут при каждом опросе, и если двигать updated_at по ним, колонка
# перестанет отвечать на вопрос «этот пост правили?».
#
# xmax = 0 в RETURNING — признак того, что строка вставлена, а не обновлена:
# единственный способ отличить новый пост от перечитанного, не делая SELECT.
UPSERT_TG_POST_SQL = """
INSERT INTO news.tg_posts (channel, message_id, posted_at, text, views,
                           links, has_media)
VALUES (%(channel)s, %(message_id)s, %(posted_at)s, %(text)s, %(views)s,
        %(links)s, %(has_media)s)
ON CONFLICT (channel, message_id) DO UPDATE SET
    text       = EXCLUDED.text,
    views      = GREATEST(COALESCE(news.tg_posts.views, 0), COALESCE(EXCLUDED.views, 0)),
    links      = EXCLUDED.links,
    has_media  = EXCLUDED.has_media,
    updated_at = CASE WHEN news.tg_posts.text IS DISTINCT FROM EXCLUDED.text
                      THEN NOW() ELSE news.tg_posts.updated_at END
RETURNING (xmax = 0) AS inserted;
"""
