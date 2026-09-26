# Воронка показаний (Go). Контекст сборки — docs/backend, как у остальных наших сервисов:
#   docker build -f docs/backend/funnel/Dockerfile -t tf-funnel docs/backend
# Пароль Kafka (tf-funnel) и техучётки шины — из Vault (secret/tf/kafka, secret/tf/funnel).
FROM golang:1.27 AS build
WORKDIR /src
COPY funnel/go.mod funnel/go.sum ./
RUN go mod download
COPY funnel/*.go ./
RUN go test ./... && CGO_ENABLED=0 go build -trimpath -ldflags="-s -w" -o /tf-funnel .

# статический бинарь: без оболочки и пакетов, с корневыми сертификатами и /tmp
FROM gcr.io/distroless/static-debian12
COPY --from=build /tf-funnel /tf-funnel
USER 10004:10004
ENV TF_ENV=prod GOMEMLIMIT=400MiB
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 CMD ["/tf-funnel", "-health"]
ENTRYPOINT ["/tf-funnel"]
