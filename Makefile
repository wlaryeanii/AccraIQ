# Makefile

.PHONY: dev-up dev-down dev-logs dev-rebuild dev-restart
.PHONY: up down logs rebuild restart

# Targets for development environment
dev-up:
	docker compose -f docker-compose.yml up --build -d

dev-down:
	docker compose -f docker-compose.yml down

dev-logs:
	docker compose -f docker-compose.yml logs -f

dev-rebuild:
	docker compose -f docker-compose.yml up --build -d --force-recreate

dev-restart: dev-down dev-up

# Targets for production environment
up:
	docker compose -f docker-compose-prod.yml up --build -d

down:
	docker compose -f docker-compose-prod.yml down

logs:
	docker compose -f docker-compose-prod.yml logs

rebuild:
	docker compose -f docker-compose-prod.yml up --build -d --force-recreate

restart: down up