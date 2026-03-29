.PHONY: all build build-daemon build-app install install-daemon install-app \
       uninstall uninstall-daemon uninstall-app \
       dev dev-daemon dev-app test lint clean status help

SHELL := /bin/bash

APP_NAME     := Memory
APP_DIR      := gui
TAURI_DIR    := $(APP_DIR)/src-tauri
BUNDLE_DIR   := $(TAURI_DIR)/target/release/bundle/macos
APP_BUNDLE   := $(BUNDLE_DIR)/$(APP_NAME).app
INSTALL_DIR  := $(HOME)/Applications
PLIST_LABEL  := com.memask.daemon
PLIST_PATH   := $(HOME)/Library/LaunchAgents/$(PLIST_LABEL).plist
UID          := $(shell id -u)

# ── top-level targets ──────────────────────────────────

all: build ## Build everything

build: build-daemon build-app ## Build daemon + desktop app

install: install-daemon install-app ## Install everything and start on login
	@echo ""
	@echo "✓ memask fully installed"
	@echo "  Daemon:  auto-starts on login (launchd)"
	@echo "  App:     $(INSTALL_DIR)/$(APP_NAME).app"
	@echo ""
	@echo "To start now without rebooting:"
	@echo "  make start"

uninstall: uninstall-app uninstall-daemon ## Remove everything
	@echo ""
	@echo "✓ memask fully uninstalled"

start: start-daemon start-app ## Start daemon + open app now

stop: stop-app stop-daemon ## Stop daemon + quit app

status: ## Show current install/run status
	@echo "── daemon ──"
	@if launchctl print gui/$(UID)/$(PLIST_LABEL) &>/dev/null; then \
		echo "  launchd: loaded"; \
	else \
		echo "  launchd: not loaded"; \
	fi
	@if curl -sf http://127.0.0.1:7394/health >/dev/null 2>&1; then \
		echo "  http:    running"; \
		curl -sf http://127.0.0.1:7394/health | python3 -m json.tool 2>/dev/null; \
	else \
		echo "  http:    not responding"; \
	fi
	@echo ""
	@echo "── app ──"
	@if [ -d "$(INSTALL_DIR)/$(APP_NAME).app" ]; then \
		echo "  installed: $(INSTALL_DIR)/$(APP_NAME).app"; \
	else \
		echo "  installed: no"; \
	fi
	@if pgrep -f "$(APP_NAME)" >/dev/null 2>&1; then \
		echo "  running: yes"; \
	else \
		echo "  running: no"; \
	fi

# ── build ──────────────────────────────────────────────

build-daemon: ## Install Python package (editable)
	uv sync
	@echo "✓ daemon built (memask cli available)"

build-app: ## Build Tauri desktop app
	cd $(TAURI_DIR) && cargo tauri build
	@echo "✓ app built: $(APP_BUNDLE)"

# ── install ────────────────────────────────────────────

install-daemon: build-daemon ## Install daemon autostart (launchd)
	memask install
	@echo "✓ daemon autostart installed"

install-app: build-app ## Copy .app to ~/Applications and add to Login Items
	@mkdir -p "$(INSTALL_DIR)"
	@rm -rf "$(INSTALL_DIR)/$(APP_NAME).app"
	cp -R "$(APP_BUNDLE)" "$(INSTALL_DIR)/$(APP_NAME).app"
	@echo "✓ app installed: $(INSTALL_DIR)/$(APP_NAME).app"
	@# Add to Login Items via osascript (macOS)
	@osascript -e 'tell application "System Events" to make login item at end with properties {path:"$(INSTALL_DIR)/$(APP_NAME).app", hidden:false}' 2>/dev/null || true
	@echo "✓ app added to Login Items"

# ── uninstall ──────────────────────────────────────────

uninstall-daemon: stop-daemon ## Remove daemon autostart
	memask uninstall
	@echo "✓ daemon autostart removed"

uninstall-app: stop-app ## Remove .app and Login Items entry
	@osascript -e 'tell application "System Events" to delete login item "$(APP_NAME)"' 2>/dev/null || true
	@rm -rf "$(INSTALL_DIR)/$(APP_NAME).app"
	@echo "✓ app uninstalled"

# ── start / stop ───────────────────────────────────────

start-daemon: ## Start daemon now via launchd
	@if launchctl print gui/$(UID)/$(PLIST_LABEL) &>/dev/null; then \
		echo "daemon already loaded"; \
	else \
		launchctl bootstrap gui/$(UID) $(PLIST_PATH); \
		echo "✓ daemon started"; \
	fi

stop-daemon: ## Stop daemon via launchd
	@launchctl bootout gui/$(UID)/$(PLIST_LABEL) 2>/dev/null || true
	@echo "✓ daemon stopped"

start-app: ## Launch the desktop app
	@open "$(INSTALL_DIR)/$(APP_NAME).app" 2>/dev/null || \
		open "$(APP_BUNDLE)" 2>/dev/null || \
		echo "app not found — run 'make build-app' first"

stop-app: ## Quit the desktop app
	@osascript -e 'tell application "$(APP_NAME)" to quit' 2>/dev/null || true

# ── development ────────────────────────────────────────

dev: ## Run daemon + app in dev mode (two terminals recommended)
	@echo "Run in separate terminals:"
	@echo "  make dev-daemon"
	@echo "  make dev-app"

dev-daemon: ## Run daemon in foreground
	memask serve

dev-app: ## Run Tauri app in dev mode (hot reload)
	cd $(TAURI_DIR) && cargo tauri dev

# ── test / lint ────────────────────────────────────────

test: ## Run all Python tests
	cd src && python -m pytest -v

test-fast: ## Run tests excluding slow model tests
	cd src && python -m pytest -v -m "not real_model"

lint: ## Run ruff linter
	ruff check src/

# ── model ──────────────────────────────────────────────

model-download: ## Download the LLM model
	memask model download

model-status: ## Show LLM model status
	memask model status

# ── clean ──────────────────────────────────────────────

clean: ## Remove build artifacts
	rm -rf $(TAURI_DIR)/target
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@echo "✓ cleaned"

# ── help ───────────────────────────────────────────────

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'
