-- Журнал аудита: docs/common/права-и-аудит.md §6.4 и §6.6, ML/INTEGRATION.md §13.4.
-- Прогоняет административная учётка (владелец схемы), один раз и повторно без вреда:
--
--   psql -v ON_ERROR_STOP=1 -f schema.sql
--
-- Паролей здесь нет. Роль audit_writer создаётся без пароля, пароль ставит администратор из Vault
-- (secret/tf/audit, поле db_password): psql -v pw="$(vault kv get -field=db_password secret/tf/audit)"
-- -c "alter role audit_writer password :'pw'".
--
-- Отличия от §6.4: у события есть event_id — повтор той же записи из потока (перезапуск сервиса
-- аудита, досылка из файла tfkit) отбрасывается уникальным ключом, а не пишется второй раз.

create schema if not exists audit;

do $$ begin
    if not exists (select from pg_roles where rolname = 'audit_writer') then
        create role audit_writer login;
    end if;
end $$;

create table if not exists audit.events (
    id          bigint generated always as identity,
    event_id    uuid not null,                  -- от источника: tfkit.Audit ставит uuid4
    occurred_at timestamptz not null,           -- когда случилось, по часам сервиса
    received_at timestamptz not null default now(),
    service     text not null,                  -- auth, bff, dispatch, funnel, ml, notify
    event_type  text not null,                  -- ticket.decided
    outcome     text not null check (outcome in ('success', 'denied', 'error')),
    actor_kind  text not null check (actor_kind in ('user', 'service', 'anonymous')),
    actor_id    uuid,                           -- sub из токена
    actor_login text,                           -- логин на момент события
    request_id  text,                           -- X-Request-ID от nginx: связывает цепочку вызовов
    ip          inet,
    object_type text,                           -- ticket, sensor, group, forecast
    object_id   text,
    area_id     bigint,                         -- узел дерева объектов
    details     jsonb not null default '{}',
    primary key (id, occurred_at),
    unique (event_id, occurred_at)
) partition by range (occurred_at);

create index if not exists events_occurred_brin on audit.events using brin (occurred_at);
create index if not exists events_actor on audit.events (actor_id, occurred_at);
create index if not exists events_object on audit.events (object_type, object_id);
create index if not exists events_type on audit.events (event_type, occurred_at);

create table if not exists audit.requests (
    occurred_at timestamptz not null,
    service     text     not null,
    method      text     not null,
    route       text     not null,             -- шаблон маршрута: /tickets/{id}/take
    status      smallint not null,
    duration_ms integer  not null,
    actor_kind  text     not null,
    actor_id    uuid,
    request_id  text,
    ip          inet
) partition by range (occurred_at);

create index if not exists requests_occurred_brin on audit.requests using brin (occurred_at);

-- Партиции по месяцам. Функция работает от имени владельца схемы: сервис аудита вызывает её при
-- старте и раз в сутки, сам создавать таблицы не может. Создаёт только недостающие месяцы.
create or replace function audit.ensure_partitions(months_back int default 1, months_ahead int default 3)
returns int language plpgsql security definer set search_path = pg_catalog, audit as $$
declare
    m date;
    made int := 0;
    t text;
begin
    for m in select generate_series(date_trunc('month', now() - make_interval(months => months_back)),
                                    date_trunc('month', now() + make_interval(months => months_ahead)),
                                    interval '1 month')::date loop
        foreach t in array array['events', 'requests'] loop
            if to_regclass(format('audit.%s_%s', t, to_char(m, 'YYYY_MM'))) is null then
                execute format('create table audit.%I partition of audit.%I for values from (%L) to (%L)',
                               t || '_' || to_char(m, 'YYYY_MM'), t, m, (m + interval '1 month')::date);
                made := made + 1;
            end if;
        end loop;
    end loop;
    return made;
end $$;

-- с начала работ по проекту и на квартал вперёд
select audit.ensure_partitions(
    (extract(year from age(date_trunc('month', now()), date '2026-01-01')) * 12
     + extract(month from age(date_trunc('month', now()), date '2026-01-01')))::int, 3);

-- Журнал запросов хранится 90 дней (§6.1): месяц, которому вышел срок, удаляется целиком. Только
-- администратор (роль владельца): сервису аудита право на функцию не выдаётся.
create or replace function audit.drop_old_requests(keep_days int default 90)
returns int language plpgsql set search_path = pg_catalog, audit as $$
declare
    r record;
    dropped int := 0;
begin
    for r in select c.relname from pg_inherits i
             join pg_class c on c.oid = i.inhrelid
             join pg_class p on p.oid = i.inhparent
             join pg_namespace n on n.oid = p.relnamespace
             where n.nspname = 'audit' and p.relname = 'requests' loop
        if to_date(right(r.relname, 7), 'YYYY_MM') + interval '1 month' <= now() - make_interval(days => keep_days) then
            execute format('drop table audit.%I', r.relname);
            dropped := dropped + 1;
        end if;
    end loop;
    return dropped;
end $$;

-- Правка и удаление строк журнала запрещены всем, включая владельца: задним числом журнал не
-- меняется (§6.6). Удалить старый месяц целиком (drop table партиции) по-прежнему можно.
create or replace function audit.forbid_change() returns trigger language plpgsql as $$
begin
    raise exception 'журнал аудита не меняется: % запрещён', tg_op using errcode = 'insufficient_privilege';
end $$;

drop trigger if exists events_no_change on audit.events;
create trigger events_no_change before update or delete on audit.events
    for each row execute function audit.forbid_change();
drop trigger if exists events_no_truncate on audit.events;
create trigger events_no_truncate before truncate on audit.events
    for each statement execute function audit.forbid_change();
drop trigger if exists requests_no_change on audit.requests;
create trigger requests_no_change before update or delete on audit.requests
    for each row execute function audit.forbid_change();

-- §6.6: сервис аудита добавляет и читает, но не меняет и не удаляет
revoke all on schema audit from public;
revoke all on all tables in schema audit from public;
revoke all on function audit.ensure_partitions(int, int), audit.drop_old_requests(int), audit.forbid_change() from public;
grant usage on schema audit to audit_writer;
grant insert, select on audit.events, audit.requests to audit_writer;
revoke update, delete, truncate on all tables in schema audit from audit_writer;
grant execute on function audit.ensure_partitions(int, int) to audit_writer;
