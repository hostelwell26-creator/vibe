```python
# vibely5.1.py
# -*- coding: utf-8 -*-
# set CAMOUFOX_WINDOW=1280x800 set CAMOUFOX_VIEWPORT=1280x720
# CAMOUFOX_HEADLESS
"""
CHANGELOG
v5.1 (2025-10-04)
- Добавлена поддержка MetaMask: функция setup_metamask для установки и активации расширения.
- Добавлено поле use_metamask в Account для активации MetaMask per-аккаунт.
- Вызов setup_metamask в run_for_account если use_metamask=True.
- Модифицирован VIS_PATCH_JS с исключением для MetaMask попапа.
- Удалены проблемные flags для no_throttle, чтобы избежать сворачивания попапов.
- Добавлен handler попапа MetaMask в get_or_create_main_page для удержания фокуса.
- Добавлена опция allow_addon_new_tab=True в launch_persistent_context_adapter для разрешения new tabs от addons.

v5.0 (2025-10-04)
- Исправлены все отступы: строго 4 пробела на уровень, устранено смешивание пробелов и табуляций.
- Исправлена синтаксическая ошибка в send_one (удалён старый код после патча).
- Добавлен asyncio.Lock для _PERSIST, чтобы предотвратить race conditions в многопоточном режиме.
- Оптимизирован wait_new_partner_message: заменён поллинг на wait_for_selector.
- Заменена рекурсия на цикл в run_for_account для избежания stack overflow.
- Исправлена логика is_login_gate_present для корректной обработки нулевых балансов.
- Добавлена обработка ValueError в _parse_header_numbers.
- Добавлен вызов save_reports при пустом ACCOUNTS.
- Обновлены все вызовы get_sent_so_far/inc_sent на асинхронные с lock.

v1.5.1 (2025-09-29)
- Синхронизация reload/кликов: единый page._action_lock. Любые goto/reload/критичные клики
  выполняются последовательно; флаг _nav_busy расставлен корректно.
- "Чат"-клик: 3 попытки кликнуть ИМЕННО кнопку; затем fallback на span
  <span class="text-[14px] whitespace-nowrap text-[#8F8F9A]">Чат</span>;
  дальше — общий текстовый поиск. Дебаунс ≥4s.
- Остальной функционал без изменений.

v1.5.0 (2025-09-29)
- "Чат"-клик: устойчивые селекторы, дебаунс (≥4s), запрет клика во время _nav_busy,
  проверка, не открыт ли уже чат; единый open_chat_tab().
- Баннер входа ("Продолжить"): детектор + режим «СТОП-ВСЁ» в этом профиле.
  Каждые 5с лог: "Нужно залогиниться…". Продолжение после появления textarea.
- Губернатор reload'ов: page_reload_governed() с лимитом N/2мин, экспоненц. пауза.
  Используется в ожиданиях чата и EQ. Watchdog больше не делает агрессивных goto/перезагрузок,
  если виден логин-баннер/регистрация.
- Снижен шум: watchdog уважает _nav_busy и login gate; force-click "Чат" вызывается реже.
- Мелкие правки: безопасные вызовы, дополнительные логи причин отказа клика.

v1.4.2 (2025-09-28)
- Мини-баннер аккаунта на всех вкладках/попапах…

v1.4.1 (2025-09-28)
- Фоновая работа при свёрнутых/неактивных вкладках…
"""

import asyncio
import random
import re
import csv
import os
import json
import time
import requests
import sys
import logging
import logging.handlers
import traceback
from pathlib import Path
import shutil

# Добавлен lock для _PERSIST чтобы избежать race conditions в multi-worker
_PERSIST_LOCK = asyncio.Lock()

def heal_profile_dir(profile_path: str) -> bool:
    p = Path(profile_path)
    try:
        p.mkdir(parents=True, exist_ok=True)
    except Exception:
        return False
    healed = False
    # снять блокировки
    for name in ("parent.lock", "lock"):
        try:
            f = p / name
            if f.exists():
                f.unlink(missing_ok=True)
                healed = True
        except Exception:
            pass
    # безопасно убрать кеш запуска и compatibility.ini — они пересоберутся
    try:
        sc = p / "startupCache"
        if sc.exists():
            shutil.rmtree(sc, ignore_errors=True)
            healed = True
    except Exception:
        pass
    try:
        ci = p / "compatibility.ini"
        if ci.exists():
            ci.unlink(missing_ok=True)
            healed = True
    except Exception:
        pass
    return healed

def quarantine_profile(profile_path: str) -> str | None:
    try:
        src = Path(profile_path)
        if not src.exists():
            return None
        stamp = time.strftime("%Y%m%d_%H%M%S")
        dst = src.with_name(src.name + f"-corrupt-{stamp}")
        shutil.move(str(src), str(dst))
        src.mkdir(parents=True, exist_ok=True)  # чистый каталог
        return str(dst)
    except Exception:
        return None

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict

from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout

__version__ = "5.1"

# ================== КАСТОМНЫЕ ИСКЛЮЧЕНИЯ ==================
# Исключение, возникающее, когда кнопка 'Чат/Chat' не найдена селекторами
class ChatNotFoundError(Exception):
    pass

# Исключение, обозначающее тайм‑аут перезагрузки страницы. Используется для
# переключения в видимый режим при повторных неудачах reload.
class ReloadTimeoutError(Exception):
    pass

# ================== ЛОГИРОВАНИЕ ==================
def setup_logger() -> logging.Logger:
    Path("logs").mkdir(exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = f"logs/run_{ts}.log"

    logger = logging.getLogger("vibely")
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.setLevel(logging.INFO)

    fh = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
    )
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)

    if not logger.handlers:
        logger.addHandler(sh)
        logger.addHandler(fh)

    logger.info("Лог запущен. Файл: %s", log_path)
    logger.info("Версия скрипта: %s", __version__)
    return logger

log = setup_logger()

# ================== НАСТРОЙКИ ==================
DELAY_SECONDS = 30
JITTER_SECONDS = (1, 10)
MAX_LEN = 300

DEFAULT_MESSAGES_PER_ACCOUNT = 20
POST_SWITCH_WAIT_SECONDS = 60
HEADLESS = False  # можно переопределить env CAMOUFOX_HEADLESS=1

# УСТОЙЧИВЫЕ СЕЛЕКТОРЫ ДЛЯ ПОЛЯ ВВОДА ЧАТА
TEXTAREA_SELS = [
    "form textarea",                       # универсально внутри формы
    "textarea[placeholder*='сообщ']",      # русские варианты («сообщение» и т.п.)
    "textarea[placeholder*='message' i]",  # английская локаль
    "textarea[placeholder*='напиш']",      # «напишите», «напиши»
]
# ---- [PATCH] selectors alias -----------------------------------------------
# Алиас одной строкой: годится и для :has(...), и для обычных локаторов
TEXTAREA_SEL = ", ".join(TEXTAREA_SELS)

def _textarea_any(page):
    # Единая точка правды: любые изменения селекторов подтянутся сюда
    return page.locator(TEXTAREA_SEL)
# ---- [/PATCH] ---------------------------------------------------------------

LIMIT_MESSAGE = "Ваши попытки обучения чату исчерпаны. Пожалуйста, посетите магазин, чтобы приобрести ещё."

WAIT_ENABLE_SEND_MS = 12000
WAIT_AFTER_CLICK_MS = 1100
WAIT_POST_APPEAR_MS = 6000
NAVIGATION_TIMEOUT_MS = 80000

BALANCE_TIMEOUT_MS = 10000
BALANCE_RELOADS = 0
BALANCE_STABLE_DELAY_MS = 600
BALANCE_ZERO_BOTH_INVALID = False

ATTEMPTS_GRACE_MS = 1500
ATTEMPTS_GRACE_STEP_MS = 200
ATTEMPTS_ZERO_STABILITY_MS = 2500
ATTEMPTS_ZERO_ACCEPT_AFTER_RELOADS = 0

EQ_TIMEOUT_MS = 80000
EQ_RELOADS = 3
REEVAL_TEXT = "Переоценка"

WORKERS = 1

GLOBAL_MIN_SEND_GAP_S = (1, 10)
GLOBAL_BURST = 2
GLOBAL_WINDOW_S = 10
BACKOFF_BASE_S = 25
BACKOFF_MAX_S = 180

BASE_CHAT_URL = "https://vibely.chat/chat"
START_CHAT_TEXT = "Начать чат"
START_REG_TEXT = "Начать регистрацию"
CHAT_TAB_TEXTS = ("Чат", "Chat")

LOGIN_GATE_TEXT = (
    "Продолжить",
    "Continue"  # баннер входа
)

RETURN_TO_EVENT_TEXTS = (
    "Вернуться к событию",
    "Вернуться к событью",
)
CREATE_COMPANION_TEXTS = (
    "Создать моего ИИ Компаньона",
    "Создать моего компаньона",
)

REPORT_CSV = "vibely_report.csv"
REPORT_TXT = "vibely_report.txt"

AI_UNAVAILABLE_TEXT = "Извините, в данный момент искусственный интеллект не может ответить. Попробуйте позже."
AI_UNAVAILABLE_COOLDOWN_S = 10

# -------- OPENAI / API ----------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
OPENAI_MODEL   = os.getenv("OPENAI_MODEL", "qwen/qwen-2.5-72b-instruct").strip()
OPENAI_BASE    = os.getenv("OPENAI_BASE_URL", "https://openrouter.ai/api").rstrip("/")
OPENAI_CC_PATH = "/v1/chat/completions"
OPENAI_RSP_PATH= "/v1/responses"
ALT_MODEL      = os.getenv("ALT_MODEL", "").strip() or None

# -------- RETRY ДЛЯ OPENAI ----------
OPENAI_RETRIES = 3
OPENAI_RETRY_BASE_S = 0.7
OPENAI_RETRY_JITTER_S = (0.0, 0.6)
OPENAI_MAX_OUTPUT_TOKENS_CAP = 2048

DEBUG = True  # включает подробные логи OpenAI вызовов

# -------- ГУБЕРНАТОР ПЕРЕЗАГРУЗОК --------
RELOAD_LIMIT_PER_2MIN = 3
RELOAD_WINDOW_S = 120
RELOAD_BACKOFF_S = (6, 18, 45, 90)  # экспоненциальные паузы при превышении

# ================ ДАННЫЕ ================
@dataclass
class Account:
    user_data_dir: str
    chat_url: str
    messages_limit: Optional[int] = None
    attempts_cached: int = 0
    tokens_cached: float = 0.0
    # Добавлены для автоматизации логина
    email: str = ""
    password: str = ""
    # Добавлено для MetaMask
    use_metamask: bool = False
    metamask_seed: str = ""  # Seed phrase for import (небезопасно, используйте осторожно)

@dataclass
class AccountReport:
    user_data_dir: str
    chat_url: str
    start_tokens: float = 0.0
    start_attempts: int = 0
    end_tokens: float = 0.0
    end_attempts: int = 0
    sent: int = 0
    limit_hit: bool = False
    status: str = "pending"
    errors: List[str] = field(default_factory=list)

# --- ВСТАВЬ СВОИ АККАУНТЫ ТУТ (добавьте email/password/metamask_seed для авто-логина) ---
ACCOUNTS: List[Account] = [
    Account(user_data_dir="vibely-profile-1",  chat_url="https://vibely.chat/chat/Aleftina", messages_limit=20),
            email="your@email.com", password="yourpassword", use_metamask=True, metamask_seed="your seed phrase here"),
]

REPORTS: Dict[str, AccountReport] = {}

# ================== PERSISTENT MESSAGE CAP ==================
COUNTERS_PATH = "vibely_counters.json"
GLOBAL_HARD_CAP = 900  # жёсткий потолок сообщений на аккаунт (персистентно)

# мини-хранилище в памяти
_PERSIST: Dict[str, int] = {}

def _acc_key(acc: "Account") -> str:
    """
    Уникальный ключ для счётчика. Предпочитаем chat_url (стабильный),
    иначе user_data_dir.
    """
    key = (acc.chat_url or "").strip()
    if not key:
        key = (acc.user_data_dir or "").strip()
    return key or "unknown"

def _load_counters() -> None:
    global _PERSIST
    try:
        if os.path.exists(COUNTERS_PATH):
            with open(COUNTERS_PATH, "r", encoding="utf-8") as f:
                data = json.load(f) or {}
                if isinstance(data, dict):
                    # нормализуем в int
                    _PERSIST = {str(k): max(0, int(v or 0)) for k, v in data.items()}
                    log.info("[CAP] Загружен счётчик: %s записей", len(_PERSIST))
                    return
    except Exception as e:
        log.warning("[CAP] Не удалось прочитать %s: %s", COUNTERS_PATH, e)
    _PERSIST = {}

def _save_counters() -> None:
    try:
        tmp = COUNTERS_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(_PERSIST, f, ensure_ascii=False, indent=2)
        os.replace(tmp, COUNTERS_PATH)  # атомарно
    except Exception as e:
        log.warning("[CAP] Не удалось сохранить счётчик: %s", e)

async def get_sent_so_far(acc: "Account") -> int:
    async with _PERSIST_LOCK:
        return max(0, int(_PERSIST.get(_acc_key(acc), 0)))

async def inc_sent(acc: "Account", delta: int = 1) -> int:
    async with _PERSIST_LOCK:
        k = _acc_key(acc)
        cur = max(0, int(_PERSIST.get(k, 0)))
        cur += max(0, int(delta))
        _PERSIST[k] = cur
        _save_counters()
        return cur

# загружаем при импорте
_load_counters()

# ================== УТИЛИТЫ ==================
def clip300(s: str) -> str:
    s = (s or "").strip()
    return s if len(s) <= MAX_LEN else s[: MAX_LEN - 1] + "…"

def _norm(s: str) -> str:
    s = clip300(s)
    s = " ".join(s.split())
    return s.strip()

def _env_flag(name: str, default: bool = False) -> bool:
    v = os.getenv(name, "").strip().lower()
    if v in ("1", "true", "yes", "on"): return True
    if v in ("0", "false", "no", "off"): return False
    return default

async def remaining_cap_for(acc: "Account") -> int:
    """
    Сколько ещё можно отправить для этого аккаунта по жёсткому лимиту.
    """
    used = await get_sent_so_far(acc)
    return max(0, int(GLOBAL_HARD_CAP) - used)

# ====== VIEWPORT HELPERS (NEW) ======
def _parse_whx(s: str) -> Optional[Tuple[int, int]]:
    try:
        s = (s or "").lower().replace(" ", "").replace("×", "x")
        if "x" not in s:
            return None
        w, h = s.split("x", 1)
        return int(w), int(h)
    except Exception:
        return None

def _env_viewport() -> Optional[Tuple[int, int]]:
    return _parse_whx(os.getenv("CAMOUFOX_VIEWPORT", ""))

# ====== ЕДИНЫЙ ЗАМОК ДЛЯ НАВИГАЦИИ/КЛИКОВ ======
def _get_action_lock(page):
    if not hasattr(page, "_action_lock"):
        setattr(page, "_action_lock", asyncio.Lock())
    return getattr(page, "_action_lock")

class _ActionScope:
    def __init__(self, page, label="nav"):
        self.page = page
        self.label = label
        self.lock = _get_action_lock(page)
    async def __aenter__(self):
        await self.lock.acquire()
        setattr(self.page, "_nav_busy", True)
    async def __aexit__(self, exc_type, exc, tb):
        try:
            setattr(self.page, "_nav_busy", False)
        finally:
            self.lock.release()

# ================== VISIBILITY / KEEP-ALIVE ПАТЧ ==================
VIS_PATCH_JS = r"""
(() => {
  try {
    // Skip for MetaMask popup to avoid conflict
    if (location.href.includes('nkbihfbeogaeaoehlefnkodbefgpgknn')) return; 
    const d = document;
    const makeConst = (obj, prop, value) => {
      try { Object.defineProperty(obj, prop, { configurable: true, get: () => value }); } catch(e) {}
    };
    makeConst(d, 'hidden', false);
    makeConst(d, 'webkitHidden', false);
    makeConst(d, 'visibilityState', 'visible');
    const origHasFocus = d.hasFocus ? d.hasFocus.bind(d) : null;
    d.hasFocus = () => true;

    const dropTypes = new Set(['visibilitychange','pagehide','blur']);
    const ET = window.EventTarget && window.EventTarget.prototype;
    if (ET && ET.addEventListener) {
      const origAdd = ET.addEventListener;
      ET.addEventListener = function(type, listener, options) {
        if (dropTypes.has(String(type))) { return; }
        return origAdd.call(this, type, listener, options);
      };
    }

    setInterval(() => { void 0; }, 30000);
  } catch(e) {}
})();
"""

# ================== MINI-BANNER (NEW) ==================
def _banner_js(label: str) -> str:
    text = json.dumps(label)
    return r"""
(() => {
  try {
    const TXT = %s;
    const ID = '__acc_banner__';
    function mk() {
      const el = document.createElement('div');
      el.id = ID;
      try {
        const sr = el.attachShadow({mode:'open'});
        const st = document.createElement('style');
        st.textContent = `
          :host { all: initial; position: fixed !important; top: 8px !important; left: 8px !important;
                  z-index: 2147483647 !important; pointer-events: none !important; }
          #__lbl { all: initial; display:inline-block; padding:2px 8px; font:600 12px/1.3 ui-sans-serif,system-ui;
                   background:rgba(0,0,0,.6); color:#fff; border-radius:8px; letter-spacing:.2px; }`;
        const wrap = document.createElement('div');
        wrap.id='__lbl'; wrap.textContent = TXT;
        sr.appendChild(st); sr.appendChild(wrap);
      } catch (_e) {
        el.textContent = TXT;
        Object.assign(el.style, { position:'fixed', top:'8px', left:'8px', padding:'2px 8px',
          font:'600 12px/1.3 ui-sans-serif,system-ui', background:'rgba(0,0,0,.6)', color:'#fff',
          borderRadius:'8px', pointerEvents:'none'
        });
      }
      (document.documentElement || document.body).appendChild(el);
      return el;
    }
    function ensure() {
      let el = document.getElementById(ID) || mk();
      el.style.setProperty('position','fixed','important');
      el.style.setProperty('top','8px','important');
      el.style.setProperty('left','8px','important');
      el.style.setProperty('z-index','2147483647','important');
      el.style.setProperty('pointer-events','none','important');
      try {
        const lbl = el.shadowRoot && el.shadowRoot.getElementById('__lbl');
        if (lbl && lbl.textContent !== TXT) lbl.textContent = TXT;
      } catch (_e) {}
      return el;
    }
    ensure();
    if (!window.__acc_banner_obs) {
      window.__acc_banner_obs = new MutationObserver(() => { if (!document.getElementById(ID)) ensure(); });
      window.__acc_banner_obs.observe(document.documentElement, { childList:true, subtree:true });
    }
    if (!window.__acc_banner_tick) { window.__acc_banner_tick = setInterval(ensure, 3000); }
    window.addEventListener('fullscreenchange', ensure, { passive:true });
  } catch (_e) {}
})();
""" % text


async def install_visibility_hacks(page):
    try:
        await page.add_init_script(script=VIS_PATCH_JS)
    except Exception:
        pass
    try:
        await page.evaluate(VIS_PATCH_JS)
    except Exception:
        pass

async def install_banner(page, label: str):
    if not label:
        return
    js = _banner_js(label)
    try:
        await page.context.add_init_script(script=js)
    except Exception:
        pass
    try:
        await page.add_init_script(script=js)
    except Exception:
        pass
    try:
        await page.evaluate(js)
    except Exception:
        pass
    try:
        for fr in page.frames:
            try:
                await fr.evaluate(js)
            except Exception:
                pass
    except Exception:
        pass
    def _on_frame_nav(frame):
        async def _reinject():
            try:
                await frame.evaluate(js)
            except Exception:
                pass
        asyncio.create_task(_reinject())
    try:
        page.on("framenavigated", _on_frame_nav)
        page.on("domcontentloaded", lambda: asyncio.create_task(page.evaluate(js)))
    except Exception:
        pass

# ================== CAMOUFOX LAUNCHER ==================
async def launch_persistent_context_adapter(playwright, *, user_data_dir, headless=False, args=None, proxy=None, **kwargs):
    args = list(args or [])
    try:
        from camoufox.async_api import AsyncNewBrowser
    except Exception as e:
        log.error("Camoufox import failed: %s", e)
        raise RuntimeError(
            "Camoufox не установлен. Установите: pip install -U camoufox[geoip] && python -m camoufox fetch"
        ) from e

    headless_opt = "virtual" if (headless and sys.platform.startswith("linux")) else headless

    def env_flag(name, default=False):
        v = os.getenv(name, "").strip().lower()
        if v in ("1","true","yes","on"): return True
        if v in ("0","false","no","off"): return False
        return default

    def env_float(name, default):
        v = os.getenv(name, "").strip()
        try: return float(v) if v else default
        except: return default

    locale_opt   = os.getenv("CAMOUFOX_LOCALE", "ru-RU,en-US")
    humanize_opt = env_float("CAMOUFOX_HUMANIZE", 1.6)
    geoip_opt    = env_flag("CAMOUFOX_GEOIP", False)
    coop_opt     = env_flag("CAMOUFOX_DISABLE_COOP", False)
    env_headless = _env_flag("CAMOUFOX_HEADLESS", False)
    if env_headless:
        headless_opt = "virtual" if sys.platform.startswith("linux") else True

    # Определяем необходимость отключать таймеры и рендеринг в фоне. Если явно
    # задан CAMOUFOX_NO_THROTTLE=1, используем это значение. В противном случае
    # когда headless_opt=False (видимый браузер), то принудительно отключаем
    # фоновое торможение, чтобы сценарии продолжали работать, даже если окно
    # находится не в фокусе.
    no_throttle_env = env_flag("CAMOUFOX_NO_THROTTLE", False)
    no_throttle = no_throttle_env or (headless_opt is False)
    if no_throttle:
        def _ensure(flag: str):
            if not any(flag == a or flag in str(a) for a in args):
                args.append(flag)
        # Эти параметры отключают экономию ресурсов в фоновых вкладках и
        # позволяют скрипту выполнять действия в неактивном окне
        _ensure("--disable-background-timer-throttling")
        _ensure("--disable-renderer-backgrounding")
        # Удалены проблемные flags для MetaMask
        # _ensure("--disable-backgrounding-occluded-windows")
        _ensure("--autoplay-policy=no-user-gesture-required")
        _ensure("--mute-audio")

    window_opt = None
    win_str = os.getenv("CAMOUFOX_WINDOW", "").lower().replace(" ", "x")
    if "x" in win_str:
        try:
            w, h = win_str.split("x", 1)
            window_opt = (int(w), int(h))
        except Exception:
            window_opt = None
 # ↓ ДОБАВЬ ЭТО (дефолт если переменная не задана/парсинг не удался)
    if not window_opt:
        window_opt = (1280, 800)

    proxy_opt = proxy if proxy else os.getenv("CAMOUFOX_PROXY", "").strip()
    if isinstance(proxy_opt, str) and proxy_opt:
        if proxy_opt.startswith("{") and proxy_opt.endswith("}"):
            try: proxy_opt = json.loads(proxy_opt)
            except: proxy_opt = {"server": proxy_opt}
        else:
            proxy_opt = {"server": proxy_opt}
    elif not proxy_opt:
        proxy_opt = None

    prof_dir = str(Path(user_data_dir).resolve())
    cf_kwargs = dict(
        persistent_context=True,
        user_data_dir=prof_dir,
        headless=headless_opt,
        locale=locale_opt,
        humanize=humanize_opt,
    )

    if geoip_opt: cf_kwargs["geoip"] = True
    if coop_opt: cf_kwargs["disable_coop"] = True
    if window_opt: cf_kwargs["window"] = window_opt
    if proxy_opt: cf_kwargs["proxy"] = proxy_opt

    if no_throttle:
        try:
            ff_prefs = {
                "dom.min_background_timeout_value": 0,
                "dom.timeout.enable_budget_timer_throttling": False,
                "dom.timeout.set_timeout_without_clamp": True,
            }
            cf_kwargs["firefox_user_prefs"] = ff_prefs
        except Exception:
            pass

    bad_flags = {
        '--disable-blink-features=AutomationControlled',
        '--no-sandbox', '--disable-dev-shm-usage', '--remote-debugging-port=9222'
    }
    safe_args = [a for a in list(args) if str(a).strip() not in bad_flags]
    if safe_args:
        cf_kwargs['args'] = safe_args
        log.info("[ADAPTER] args: %s", " ".join(safe_args))

    cf_kwargs.update(kwargs or {})
    prof_dir = str(user_data_dir)
    heal_profile_dir(prof_dir)
    # ---- [PATCH] launch_persistent_context_adapter: resilient launch ------------
    try:
        ctx = await AsyncNewBrowser(playwright, **cf_kwargs)
    except Exception as e1:
        msg = (str(e1) or "").lower()
        # 1) Если у тебя уже есть логика «карантина» профиля — оставляем её первой веткой
        if (("target page" in msg and "closed" in msg)
            or "targetclosederror" in msg
            or "process did exit" in msg
            or "temporary directories cleanup" in msg):
            q = quarantine_profile(prof_dir)
            log.error(
                "[ADAPTER] профиль '%s' помечен как corrupt (%s). Карантин: %s → retry...",
                prof_dir,
                e1.__class__.__name__,
                q or "-",
            )
            heal_profile_dir(prof_dir)
            # повторная попытка
            ctx = await AsyncNewBrowser(playwright, **cf_kwargs)
        else:
            # 2) Подозрение на SWGL/headless-глюк (в т.ч. банальный таймаут при headless)
            swgl_hint = (
                ("rendercompositorswgl" in msg)
                or ("failed mapping default framebuffer" in msg)
                or ("timeout" in msg and "180000ms" in msg)
            )
            if swgl_hint or True:
                try:
                    cf_kwargs_fallback = dict(cf_kwargs)
                    # отключаем headless — на Windows это наиболее надёжный обход
                    cf_kwargs_fallback["headless"] = False
                    # доп. безопасные Firefox-параметры против проблемного рендера
                    ff_prefs = dict(cf_kwargs_fallback.get("firefox_user_prefs") or {})
                    ff_prefs.update(
                        {
                            "webgl.disabled": True,
                            "gfx.canvas.accelerated": False,
                            "layers.acceleration.force-enabled": False,
                            "gfx.webrender.force-disabled": True,
                            "layers.acceleration.disabled": True,
                        }
                    )
                    cf_kwargs_fallback["firefox_user_prefs"] = ff_prefs
                    log.warning(
                        "[ADAPTER] headless launch failed (%s). Fallback → headless=False + safe prefs",
                        e1.__class__.__name__,
                    )
                    ctx = await AsyncNewBrowser(playwright, **cf_kwargs_fallback)
                except Exception:
                    # если и так не взлетело — бросаем исходную ошибку, чтобы её не потерять
                    raise e1
            else:
                raise
    # ---- [/PATCH] ---------------------------------------------------------------

    # сохраняем флаг no_throttle в контекст браузера, чтобы другие части кода могли
    # определить, нужно ли выводить вкладку на передний план. Используется в
    # watchdog для отказа от bring_to_front() в автономном режиме.
    try:
        setattr(ctx, "_no_throttle", no_throttle)
    except Exception:
        pass
    log.info(
        "[ADAPTER] Camoufox persistent context запущен: profile=%s headless=%s no_throttle=%s",
        str(user_data_dir),
        headless_opt,
        no_throttle,
    )
    return ctx

# ================== КОНТРОЛЬ ВКЛАДОК ==================
def _now() -> float:
    try: return asyncio.get_running_loop().time()
    except: return time.time()

async def get_or_create_main_page(context, target_url: Optional[str] = None):
    try:
        for p in list(context.pages):
            url = (getattr(p, "url", "") or "").strip()
            if url in ("about:blank", "about:newtab", "") and len(context.pages) > 1:
                try:
                    await p.close()
                    log.info("[TABS] Закрыта лишняя пустая вкладка")
                except Exception:
                    pass

        page = None
        if context.pages:
            for cand in context.pages:
                u = (getattr(cand, "url", "") or "").strip()
                page = cand
                if u in ("about:blank", "about:newtab", "") and len(context.pages) == 1:
                    break
                if u not in ("about:blank", "about:newtab", ""):
                    break

        if not page:
            page = await context.new_page()
            log.info("[TABS] Создана новая вкладка")

    # --- Настройка viewport ---
    # Берём размеры из CAMOUFOX_VIEWPORT или CAMOUFOX_WINDOW,
    # если не заданы — выставляем дефолт 1280x720,
    # чтобы в headless не включалась «мобильная» вёрстка и кнопка Чат была доступна.
        try:
            vp = _env_viewport()
            if not vp: vp = _parse_whx(os.getenv("CAMOUFOX_WINDOW", ""))
            if vp:
                try: await page.set_viewport_size({"width": vp[0], "height": vp[1]})
                except Exception: pass
                try: setattr(context, "_fixed_vp", vp)
                except Exception: pass
                log.info("[VIEWPORT] установлен %sx%s (контекст)", vp[0], vp[1])
        except Exception as e:
            log.info("[VIEWPORT] подготовка viewport fail: %s", e)


        await install_visibility_hacks(page)
        try:
            lbl = getattr(context, "_banner_label", None) or getattr(page, "_banner_label", None)
            if lbl: await install_banner(page, lbl)
        except Exception:
            pass

        def _on_new_page(p):
            async def _handle():
                try:
                    await install_visibility_hacks(p)
                    try:
                        vp2 = getattr(context, "_fixed_vp", None)
                        if vp2:
                            try: await p.set_viewport_size({"width": vp2[0], "height": vp2[1]})
                            except Exception: pass
                    except Exception: pass
                    try:
                        lbl2 = getattr(context, "_banner_label", None)
                        if lbl2: await install_banner(p, lbl2)
                    except Exception: pass
                    await asyncio.sleep(0.1)
                    u = (getattr(p, "url", "") or "").strip()
                    if u in ("about:blank", "about:newtab", ""):
                        await p.close()
                        log.info("[TABS] Закрыл всплывшую пустую вкладку")
                except Exception:
                    pass
            asyncio.create_task(_handle())
        context.on("page", _on_new_page)

        async def _on_popup(p):
            try:
                await install_visibility_hacks(p)
                try:
                    vp2 = getattr(context, "_fixed_vp", None)
                    if vp2:
                        try: await p.set_viewport_size({"width": vp2[0], "height": vp2[1]})
                        except Exception: pass
                    except Exception: pass
                try:
                    lbl3 = getattr(context, "_banner_label", None)
                    if lbl3: await install_banner(p, lbl3)
                except Exception: pass
                u = (getattr(p, "url", "") or "").strip()
                if u in ("about:blank", "about:newtab", ""):
                    await p.close()
                    log.info("[TABS] Закрыл popup about:blank")
                # Добавлено: Обработка попапа MetaMask (если открылся)
                if 'nkbihfbeogaeaoehlefnkodbefgpgknn' in u:
                    log.info("[METAMASK] Detected MetaMask popup: %s", u)
                    # Фокус на попапе, чтобы не сворачивался
                    await p.bring_to_front()
                    await asyncio.sleep(0.5)  # Дебанс
                    await p.click('body', force=True)  # Клик в тело попапа
            except Exception:
                pass
        page.on("popup", _on_popup)

        if target_url:
            cur = (getattr(page, "url", "") or "")
            if not cur or "vibely.chat" not in cur:
                try:
                    await page.goto(target_url, wait_until="domcontentloaded", timeout=NAVIGATION_TIMEOUT_MS)
                except Exception as e:
                    log.warning("[TABS] Ошибка первичной навигации: %s", e)
        return page
    except Exception as e:
        log.error("[TABS] Критическая ошибка контроля вкладок: %s", e, exc_info=True)
        try:
            page = await context.new_page()
            await install_visibility_hacks(page)
            try:
                lbl = getattr(context, "_banner_label", None)
                if lbl: await install_banner(page, lbl)
            except Exception: pass
            return page
        except Exception:
            raise

# ================== OPENAI (без изменений) ==================
def _post(url: str, headers: dict, payload: dict, timeout=40) -> requests.Response:
    if DEBUG:
        log.info("[OPENAI] POST %s model=%s", url, payload.get('model'))
    return requests.post(url, headers=headers, json=payload, timeout=40)

def _parse_chat_completions(data: dict) -> Optional[str]:
    try:
        return (data["choices"][0]["message"]["content"] or "").strip()
    except Exception:
        return None

def _parse_responses(data: dict) -> Optional[str]:
    out = data.get("output")
    if not isinstance(out, list):
        choices = data.get("choices")
        if isinstance(choices, list) and choices:
            return (choices[0].get("message", {}).get("content", "") or "").strip()
        return None
    texts: List[str] = []
    for item in out:
        itype = item.get("type")
        if itype == "message":
            content = item.get("content", [])
            if isinstance(content, list):
                for piece in content:
                    if isinstance(piece, dict):
                        t = piece.get("text")
                        if t and isinstance(t, str):
                            texts.append(t.strip())
                        if not t and "output_text" in piece and isinstance(piece["output_text"], str):
                            texts.append(piece["output_text"].strip())
            elif isinstance(content, str) and content.strip():
                texts.append(content.strip())
        elif itype == "reasoning":
            continue
        else:
            if "output_text" in item and isinstance(item["output_text"], str):
                texts.append(item["output_text"].strip())
    if texts:
        return "\n".join([t for t in texts if t])[:2000].strip()
    return None

def _jitter_sleep(base: float, attempt: int):
    delay = base * (1.6 ** attempt) + random.uniform(*OPENAI_RETRY_JITTER_S)
    time.sleep(delay)

def _should_retry_http(resp: requests.Response) -> bool:
    if resp is None:
        return True
    if resp.status_code in (408, 409, 425, 429, 500, 502, 503, 504):
        return True
    return False

def _should_retry_rsp_json(data: dict) -> bool:
    if not isinstance(data, dict):
        return True
    if data.get("status") == "incomplete":
        return True
    inc = data.get("incomplete_details") or {}
    if inc.get("reason") in ("max_output_tokens", "rate_limit_exceeded", "server_error"):
        return True
    return _parse_responses(data) in (None, "", " ")

def openai_romantic_300(src_text: str) -> str:
    if not OPENAI_API_KEY:
        if DEBUG:
            log.error("[OPENAI] ERROR: OPENAI_API_KEY пуст")
        return "Я рядом и слушаю тебя сердцем. Давай держаться вместе, хорошо?"

    sys_prompt = (
        "Ты — влюблённый рассказчик и собеседник. Отвечай на сообщения в тёплой, человечной форме. "
        "Длина ответа 120–260 символов. Каждый ответ уникальный, даже при одинаковом или пустом запросе. "
        "Стиль: Романтично-эмоциональный простая мысль + один образ + короткий вывод. "
        "Пиши от первого лица к любимой. Без списков и длинных тире. Даты/EXP — не воспринимаем за текст"
    )

    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }

    def _call_cc(model_name: str) -> Optional[str]:
        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": str(src_text)[:4000]}
            ],
            "max_tokens": 180
        }
        last_err = None
        for attempt in range(OPENAI_RETRIES):
            try:
                r = _post(OPENAI_BASE + OPENAI_CC_PATH, headers, payload, timeout=40)
                if DEBUG:
                    log.info("[OPENAI/CC] status=%s attempt=%s", r.status_code, attempt + 1)
                if r.status_code == 200:
                    data = r.json()
                    msg = _parse_chat_completions(data)
                    if msg:
                        return clip300(msg)
                if not _should_retry_http(r):
                    break
            except Exception as e:
                last_err = e
                if DEBUG:
                    log.exception("[OPENAI/CC] exception")
            _jitter_sleep(OPENAI_RETRY_BASE_S, attempt)
        if last_err and DEBUG:
            log.error("[OPENAI/CC] failed after retries: %s", last_err)
        return None

    def _call_rsp(model_name: str) -> Optional[str]:
        payload = {
            "model": model_name,
            "input": [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": str(src_text)[:4000]}
            ],
            "max_output_tokens": 480,
            "reasoning": {"effort": "low"}
        }
        last_err = None
        for attempt in range(OPENAI_RETRIES):
            try:
                r = _post(OPENAI_BASE + OPENAI_RSP_PATH, headers, payload, timeout=45)
                if DEBUG:
                    log.info("[OPENAI/RSP] status=%s attempt=%s", r.status_code, attempt + 1)
                if r.status_code == 200:
                    data = r.json()
                    txt = _parse_responses(data)
                    if txt and txt.strip():
                        return clip300(txt)
                    if DEBUG and (data.get("status") == "incomplete" or _should_retry_rsp_json(data)):
                        log.info("[OPENAI/RSP] no text or incomplete, preview: %s", json.dumps(data)[:220])
                    payload["max_output_tokens"] = min(
                        int(payload.get("max_output_tokens", 480)) + 180,
                        OPENAI_MAX_OUTPUT_TOKENS_CAP
                    )
                else:
                    if not _should_retry_http(r):
                        if DEBUG:
                            log.info("[OPENAI/RSP] body: %s", r.text[:600])
                        break
            except Exception as e:
                last_err = e
                if DEBUG:
                    log.exception("[OPENAI/RSP] exception")
            _jitter_sleep(OPENAI_RETRY_BASE_S, attempt)
        if last_err and DEBUG:
            log.error("[OPENAI/RSP] failed after retries: %s", last_err)
        return None

    msg = _call_cc(OPENAI_MODEL)
    if msg:
        return msg

    msg = _call_rsp(OPENAI_MODEL)
    if msg:
        return msg

    if ALT_MODEL:
        if DEBUG:
            log.info("[OPENAI] Trying ALT_MODEL via CC")
        msg = _call_cc(ALT_MODEL)
        if msg:
            return msg
        if DEBUG:
            log.info("[OPENAI] Trying ALT_MODEL via RSP")
        msg = _call_rsp(ALT_MODEL)
        if msg:
            return msg

    return "Я рядом и слушаю тебя сердцем. Давай держаться вместе, хорошо?"


# === ASYNC-ОБЁРТКА ДЛЯ OPENAI (не блокирует event loop) ===
OPENAI_CONC = int(os.getenv("OPENAI_CONC", "5"))
_OPENAI_SEM = asyncio.Semaphore(OPENAI_CONC)

async def aopenai_romantic_300(src_text: str) -> str:
    """
    Асинхронная обёртка: выполняет openai_romantic_300 в отдельном потоке,
    чтобы не блокировать event loop Playwright/asyncio.
    """
    async with _OPENAI_SEM:
        return await asyncio.to_thread(openai_romantic_300, src_text)

# ================== BACKOFF-МОНИТОР ==================
class GlobalSendCoordinator:
    def __init__(self, burst: int, window_s: int):
        self._lock = asyncio.Lock()
        self._last_send_ts: float = 0.0
        self._burst = burst
        self._window_s = window_s
        self._timestamps: List[float] = []
        self._backoff_until: float = 0.0
    def _rand_gap(self) -> float:
        lo, hi = GLOBAL_MIN_SEND_GAP_S
        return random.uniform(lo, hi)
    async def _enforce_gap_and_burst(self):
        now = asyncio.get_running_loop().time()
        if now < self._backoff_until:
            await asyncio.sleep(self._backoff_until - now)
        gap = self._rand_gap()
        wait_needed = max(0.0, self._last_send_ts + gap - now)
        if wait_needed > 0:
            await asyncio.sleep(wait_needed)
        now = asyncio.get_running_loop().time()
        self._timestamps = [t for t in self._timestamps if now - t <= self._window_s]
        if len(self._timestamps) >= self._burst:
            earliest = min(self._timestamps)
            wait_more = self._window_s - (now - earliest) + 0.01
            if wait_more > 0:
                await asyncio.sleep(wait_more)
    async def acquire_slot(self):
        await self._lock.acquire()
        try:
            await self._enforce_gap_and_burst()
        except:
            self._lock.release()
            raise
    def release_slot(self):
        now = asyncio.get_running_loop().time()
        self._last_send_ts = now
        self._timestamps.append(now)
        self._lock.release()
    def apply_backoff(self, seconds: float):
        now = asyncio.get_running_loop().time()
        self._backoff_until = max(self._backoff_until, now + min(seconds, BACKOFF_MAX_S))

GLOBAL_SEND_COORD = GlobalSendCoordinator(GLOBAL_BURST, GLOBAL_WINDOW_S)

def attach_block_watchers(page):
    async def _tab_space_once(p):
        setattr(p, "_backoff_recovery_running", True)
        try:
            await asyncio.sleep(15)
            try: await p.keyboard.press("Tab")
            except Exception: pass
            await asyncio.sleep(2)
            try: await p.keyboard.press("Space")
            except Exception: pass
            log.info("[BACKOFF] recovery: TAB затем SPACE")
        except Exception as e:
            log.error("[BACKOFF] recovery error: %s", e)
        finally:
            setattr(p, "_backoff_recovery_running", False)

    def _on_response(resp):
        try:
            st = resp.status
            url = resp.url or ""
            if st == 403 and "vibely.chat" in url:
                log.warning("[BACKOFF] HTTP %s на %s → backoff", st, url)
                GLOBAL_SEND_COORD.apply_backoff(BACKOFF_BASE_S * 1.5)
                if not getattr(page, "_backoff_recovery_running", False):
                    asyncio.create_task(_tab_space_once(page))
        except Exception:
            pass

    page.on("response", _on_response)

# ======= ГУБЕРНАТОР ПЕРЕЗАГРУЗОК =======
def _reload_bucket(page):
    if not hasattr(page, "_reload_ts"):
        setattr(page, "_reload_ts", [])
    return getattr(page, "_reload_ts")

async def page_reload_governed(page, *, reason: str = "", wait_until="domcontentloaded"):
    ts = _reload_bucket(page)
    now = _now()
    ts[:] = [t for t in ts if now - t <= RELOAD_WINDOW_S]
    if len(ts) >= RELOAD_LIMIT_PER_2MIN:
        idx = min(len(ts)-RELOAD_LIMIT_PER_2MIN, len(RELOAD_BACKOFF_S)-1)
        backoff = RELOAD_BACKOFF_S[idx]
        log.info("[RELOAD-GOV] лимит: откладываю reload на %ss (reason=%s)", backoff, reason or "-")
        await asyncio.sleep(backoff)
    try:
        async with _ActionScope(page, "reload"):
            await page.reload(wait_until=wait_until, timeout=NAVIGATION_TIMEOUT_MS)
            ts.append(_now())
    except Exception as e:
        log.info("[RELOAD-GOV] reload fail (%s): %s", reason or "-", e)
        try:
            url = (page.url or "").strip()
            if url:
                async with _ActionScope(page, "goto_after_reload_fail"):
                    await page.goto(url, wait_until=wait_until, timeout=NAVIGATION_TIMEOUT_MS)
                    ts.append(_now())
                    log.info("[RELOAD-GOV] fallback goto after reload fail (%s) succeeded", reason or "-")
                    return
        except Exception as e2:
            log.info("[RELOAD-GOV] goto fallback fail (%s): %s", reason or "-", e2)
            raise ReloadTimeoutError(f"reload and goto failed ({reason or '-'})")

# ======= WATCHDOG =======
async def start_user_interference_watchdog(page):
    t = getattr(page, "_watchdog_task", None)
    if t and hasattr(t, "done") and not t.done():
        return
    async def _watch():
        try:
            while True:
                try:
                    if page.is_closed():
                        break
                except Exception:
                    break

                if getattr(page, "_nav_busy", False):
                    await asyncio.sleep(1.0)
                    continue

                if await is_login_gate_present(page):
                    await asyncio.sleep(2.0)
                    continue

                try:
                    if await detect_generic_challenge(page):
                        await asyncio.sleep(2.0)
                        continue
                except Exception:
                    pass

                current_url = getattr(page, "url", "") or ""
                textarea_exists = False
                try:
                    textarea_exists = (await _textarea_any(page).count()) > 0
                except Exception as e:
                    msg = str(e)
                    if "has been closed" in msg or "Target page" in msg or "browser has been closed" in msg:
                        break
                    await asyncio.sleep(1.5)
                    continue

                if ("vibely.chat" not in current_url) or (not textarea_exists):
                    log.info("[GUARD] Восстанавливаю фокус чата…")
                    try:
                        await smart_goto(page, BASE_CHAT_URL)
                        await escape_register_if_needed(page)
                    except Exception:
                        pass
                else:
                    try:
                        no_thr = False
                        try:
                            no_thr = getattr(page.context, "_no_throttle", False)
                        except Exception:
                            no_thr = getattr(page.context.browser, "_no_throttle", False)
                    except Exception:
                        no_thr = False
                    if not no_thr:
                        try:
                            await page.bring_to_front()
                        except Exception:
                            pass
                await asyncio.sleep(2.0)
        finally:
            try:
                setattr(page, "_watchdog_task", None)
            except Exception:
                pass

    setattr(page, "_watchdog_task", asyncio.create_task(_watch()))

# ================== НАВИГАЦИЯ ==================
async def detect_generic_challenge(page) -> bool:
    try:
        if await page.locator("iframe[src*='cdn-cgi/challenge-platform']").count() > 0:
            return True
        html = await page.content()
        if "data-cf-beacon" in html or "/cdn-cgi/challenge-platform" in html:
            return True
    except:
        pass
    return False

def stable_phase(obj_key: str, max_shift_s: float = 10.0) -> float:
    h = abs(hash(obj_key)) % 1000003
    return (h % int(max_shift_s * 1000)) / 1000.0

async def detect_ai_unavailable(page) -> bool:
    try:
        loc = page.get_by_text(AI_UNAVAILABLE_TEXT, exact=False)
        if await loc.count() > 0:
            try:
                await loc.first.wait_for(state="visible", timeout=400)
                return True
            except PlaywrightTimeout:
                pass
        body = await page.locator("body").inner_text()
        return AI_UNAVAILABLE_TEXT in (body or "")
    except Exception:
        return False

async def cooldown_and_reload(page, reason: str = "AI unavailable"):
    try:
        log.warning("[AI-UNAVAILABLE] %s → пауза %ss и reload", reason, AI_UNAVAILABLE_COOLDOWN_S)
        await asyncio.sleep(AI_UNAVAILABLE_COOLDOWN_S)
        await page_reload_governed(page, reason=reason)
        await escape_register_if_needed(page)
    except Exception as e:
        log.error("[AI-UNAVAILABLE] reload error: %s", e)

# ======= ЕДИНЫЙ КЛИКЕР «ЧАТ» С ДЕБАУНСОМ И ПРИОРИТЕТОМ КНОПКИ =======
async def open_chat_tab(page) -> bool:
    """Открывает вкладку 'Чат/Chat' один раз, безопасно."""
    try:
        cur_url = (page.url or "").strip()
        if cur_url and "vibely.chat" not in cur_url:
            log.info("[NAV] Открыта чужая страница (%s) — переход на %s", cur_url, BASE_CHAT_URL)
            try:
                await smart_goto(page, BASE_CHAT_URL)
            except Exception as e:
                log.warning("[NAV] Не удалось перейти на базовую страницу чата: %s", e)
    except Exception:
        pass

    try:
        if (await _textarea_any(page).count()) > 0:
            return False
    except Exception:
        pass

    last = getattr(page, "_last_chat_click_ts", 0.0)
    if _now() - last < 4.0:
        log.info("[NAV] дебаунс клика 'Чат/Chat' (%.1fs)", _now() - last)
        return False

    if getattr(page, "_nav_busy", False):
        log.info("[NAV] пропуск клика 'Чат/Chat' — _nav_busy")
        return False

    # 0) Оранжевая круглая кнопка с chat_icon
    chat_orange_button_candidates = [
        page.locator(
            "button.w-\\[56px\\].h-\\[56px\\].cursor-pointer.bg-\\[\\#FE5626\\].rounded-full.items-center.justify-center:has(img[alt='chat_icon'])"
        ),
        page.locator("button:has(img[alt='chat_icon'])"),
        page.locator("button:has(img[alt='chat_icon'][src*='chat_foc_icon.svg'])"),
    ]
    for loc in chat_orange_button_candidates:
        try:
            if await loc.count() > 0:
                el = loc.first
                try: await el.scroll_into_view_if_needed(timeout=800)
                except Exception: pass
                async with _ActionScope(page, "click_chat_orange_button"):
                    await el.click(timeout=25000)
                    setattr(page, "_last_chat_click_ts", _now())
                log.info("[NAV] click 'Чат/Chat' через круглую оранжевую кнопку (%s)", str(loc))
                try:
                    await page.wait_for_selector(TEXTAREA_SEL, timeout=35000)
                    return True
                except PlaywrightTimeout:
                    await page.wait_for_timeout(600)
        except Exception as e:
            log.info("[NAV] orange-button 'Чат/Chat' fail: %s", e)

    # 1) До 3 попыток кликнуть ИМЕННО кнопку "Чат" или "Chat"
    for attempt in range(3):
        for t in CHAT_TAB_TEXTS:
            button_candidates = [
                page.get_by_role("button", name=t),
                page.locator(f"button:has-text('{t}')"),
                page.locator(f"button:has(span:has-text('{t}'))"),
            ]
            for loc in button_candidates:
                try:
                    if await loc.count() > 0:
                        el = loc.first
                        try: await el.scroll_into_view_if_needed(timeout=800)
                        except Exception: pass
                        async with _ActionScope(page, "click_chat_button"):
                            await el.click(timeout=25000)
                            setattr(page, "_last_chat_click_ts", _now())
                        log.info("[NAV] click '%s' кнопка (%s), попытка %s", t, str(loc), attempt + 1)
                        try:
                            await page.wait_for_selector(TEXTAREA_SEL, timeout=35000)
                            return True
                        except PlaywrightTimeout:
                            await page.wait_for_timeout(600)
                            continue
                except Exception as e:
                    log.info("[NAV] button-click '%s' fail: %s", t, e)
        await page.wait_for_timeout(400)

    # 2) Fallback: точечный span как в верстке (RU/EN)
    try:
        for t in CHAT_TAB_TEXTS:
            span_css = r"span.text-\[14px\].whitespace-nowrap.text-\[\#8F8F9A\]"+f":has-text('{t}')"
            loc = page.locator(span_css)
            if await loc.count() > 0:
                async with _ActionScope(page, "click_chat_span"):
                    await loc.first.click(timeout=25000)
                    setattr(page, "_last_chat_click_ts", _now())
                log.info("[NAV] click '%s' через span (%s)", t, span_css)
                try:
                    await page.wait_for_selector(TEXTAREA_SEL, timeout=35000)
                    return True
                except PlaywrightTimeout:
                    pass
    except Exception as e:
        log.info("[NAV] span-click 'Чат/Chat' fail: %s", e)

    # 3) Последний резерв: таб, ссылка, общий текст (RU/EN)
    reserve_locs = []
    for t in CHAT_TAB_TEXTS:
        reserve_locs.extend([
            page.get_by_role("tab", name=t),
            page.locator("a[href*='/chat']"),
            page.get_by_text(t, exact=True),
            page.get_by_text(t, exact=False),
        ])
    for loc in reserve_locs:
        try:
            if await loc.count() > 0:
                try: await loc.first.scroll_into_view_if_needed(timeout=800)
                except Exception: pass
                async with _ActionScope(page, "click_chat_reserve"):
                    await loc.first.click(timeout=25000)
                    setattr(page, "_last_chat_click_ts", _now())
                log.info("[NAV] click 'Чат/Chat' через резерв %s", str(loc))
                try:
                    await page.wait_for_selector(TEXTAREA_SEL, timeout=35000)
                except PlaywrightTimeout:
                    pass
                return True
        except Exception as e:
            log.info("[NAV] reserve-click 'Чат/Chat' fail: %s", e)

    log.info("[NAV] 'Чат/Chat' не найден селекторами")
    # Генерируем исключение, чтобы верхний уровень мог переключиться в видимый режим
    raise ChatNotFoundError("'Чат/Chat' не найден селекторами")

# ======= ДЕТЕКТОР БАННЕРА «ПРОДОЛЖИТЬ» (с проверкой балансов) =======
async def is_login_gate_present(page) -> bool:
    """
    Возвращает True, если страница просит логин:
      - есть баннер/кнопка «Продолжить» или "Continue" (кнопка, текст, div в диалоге),
      - явные признаки экрана логина/регистрации,
      - НО если textarea уже видна или по балансам понятно, что пользователь залогинен — False.
    """
    try:
        if (await _textarea_any(page).count()) > 0:
            return False
    except Exception:
        pass

    try:
        tk, at = await _read_balances(page)
        if tk is not None or at is not None:
            return False
    except Exception:
        pass

    url_l = ""
    try:
        url_l = (page.url or "").lower()
    except Exception:
        pass

    has_login_inputs = False
    try:
        has_login_inputs = (await page.locator("input[type='email'], input[type='password']").count()) > 0
    except Exception:
        pass

    has_login_text = False
    try:
        if (await page.get_by_text("Войд", exact=False).count()) > 0:
            has_login_text = True
        elif (await page.get_by_text("Login", exact=False).count()) > 0:
            has_login_text = True
        elif (await page.get_by_text("Sign in", exact=False).count()) > 0:
            has_login_text = True
    except Exception:
        pass

    login_like = (
        ("/login" in url_l) or
        ("/signin" in url_l) or
        ("/register" in url_l) or
        ("/signup" in url_l) or
        has_login_inputs or
        has_login_text
    )

    in_dialog = False
    try:
        in_dialog = (await page.get_by_role("dialog").count()) > 0
    except Exception:
        try:
            in_dialog = (await page.locator("[role='dialog']").count()) > 0
        except Exception:
            in_dialog = False

    gate_texts = LOGIN_GATE_TEXT  # Исправлено: кортеж, но в коде ("Продолжить", "Continue")

    for t in gate_texts:
        try:
            if (await page.get_by_role("button", name=t).count()) > 0:
                return True
        except Exception:
            pass

    for t in gate_texts:
        try:
            if (await page.get_by_text(t, exact=True).count()) > 0:
                return True
        except Exception:
            pass

    if in_dialog:
        for t in gate_texts:
            try:
                if (await page.locator(f"div:has-text('{t}')").count()) > 0:
                    return True
            except Exception:
                pass
        for t in gate_texts:
            try:
                loc = page.locator("button").filter(has_text=t)
                if (await loc.count()) > 0:
                    try:
                        if await loc.first.is_visible():
                            return True
                    except Exception:
                        return True
            except Exception:
                pass

    if login_like:
        return True

    return False

# ======= ОЖИДАНИЕ ЛОГИНА/ЧАТА С МЕТКОЙ АККАУНТА =======
async def wait_until_in_chat(page, *, log_every_s: int = 5, max_wait_s: Optional[float] = None) -> None:
    """
    Ждём появления textarea чата. Если виден логин‑гейт — периодически
    логируем, указывая конкретный аккаунт (метка из chat_url или баннера).
    Если передан max_wait_s (в секундах) и чат не открылся за это время,
    генерирует TimeoutError. Это позволяет внешнему коду перезапустить
    браузер, если открыть чат не удалось в разумные сроки.
    """
    def _label() -> str:
        try:
            cu = getattr(page, "_target_chat_url", None) or ""
            if "/chat/" in cu:
                tail = cu.split("/chat/", 1)[1]
                tail = tail.split("?", 1)[0].split("#", 1)[0].strip("/")
                if tail:
                    return tail
        except Exception:
            pass
        try:
            lbl = getattr(page, "_banner_label", None) \
                  or getattr(page.context, "_banner_label", None) \
                  or getattr(page.context.browser, "_banner_label", None)
            if lbl:
                return lbl
        except Exception:
            pass
        try:
            return page.context._profile_name
        except Exception:
            return page.url or "unknown"

    start_ts = _now()
    while True:
        if max_wait_s is not None:
            try:
                if _now() - start_ts > max_wait_s:
                    raise TimeoutError(f"чат не открыт в течение {max_wait_s} секунд")
            except Exception:
                pass

        try:
            if (await _textarea_any(page).count()) > 0:
                log.info("[LOGIN] Чат открыт — вход подтверждён для аккаунта: %s.", _label())
                return
        except Exception:
            pass

        try:
            on_register = "/register" in (page.url or "")
        except Exception:
            on_register = False

        try:
            login_gate = await is_login_gate_present(page)
        except Exception:
            login_gate = False

        if login_gate or on_register:
            log.info("[LOGIN-GATE] Нужно залогиниться и открыть чат для аккаунта: %s. Жду…", _label())
            await asyncio.sleep(log_every_s)
            continue

        try:
            await escape_register_if_needed(page)
        except Exception:
            pass

        clicked = False
        try:
            clicked = await open_chat_tab(page)
        except ChatNotFoundError:
            raise
        except Exception:
            pass

        try:
            await page.wait_for_selector(TEXTAREA_SEL, timeout=15000)
            log.info("[LOGIN] Чат открыт — вход подтверждён для аккаунта: %s.", _label())
            return
        except Exception:
            if clicked:
                log.info("[LOGIN] 'Чат' не открылся за 15с для аккаунта: %s — попробую ещё…", _label())
            else:
                log.info("[LOGIN] Чат пока недоступен для аккаунта: %s — повторная проверка…", _label())
            await asyncio.sleep(5)

async def smart_goto(page, url: str, retries: int = 2) -> None:
    last_err = None
    for i in range(retries + 1):
        try:
            async with _ActionScope(page, "goto"):
                await page.goto(url, wait_until="domcontentloaded", timeout=NAVIGATION_TIMEOUT_MS)
            if await detect_generic_challenge(page):
                log.warning("[CHALLENGE] после goto → backoff")
                GLOBAL_SEND_COORD.apply_backoff(BACKOFF_BASE_S)
                await asyncio.sleep(BACKOFF_BASE_S)
            return
        except Exception as e:
            last_err = e
            log.warning("[GOTO] ошибка/таймаут (%s/%s): %s", i+1, retries+1, e)
            await asyncio.sleep(15.5)
    raise last_err

async def _click_once_by_text(page, text: str, flag_attr: str,
                              prefer_role_button: bool = True,
                              exact: bool = False, wait_ms: int = 1000) -> bool:
    if getattr(page, flag_attr, False):
        return False
    try:
        loc = None
        if prefer_role_button:
            loc = page.get_by_role("button", name=text)
            if await loc.count() == 0:
                loc = page.get_by_text(text, exact=exact)
        else:
            loc = page.get_by_text(text, exact=exact)
        if await loc.count() == 0:
            loc = page.get_by_text(text, exact=False)
        if await loc.count() > 0:
            if await is_login_gate_present(page):
                return False
            async with _ActionScope(page, f"click_{text}"):
                await loc.first.click(timeout=35000)
            setattr(page, flag_attr, True)
            log.info("[NAV] one-shot click: '%s'", text)
            await page.wait_for_timeout(wait_ms)
            return True
    except Exception as e:
        log.info("[NAV] one-shot click fail '%s': %s", text, e)
    return False

async def _click_once_any_by_texts(page, texts: Tuple[str, ...], flag_attr: str,
                                   prefer_role_button: bool = True,
                                   exact: bool = False, wait_ms: int = 1000) -> bool:
    if getattr(page, flag_attr, False):
        return False
    for t in texts:
        ok = await _click_once_by_text(page, t, flag_attr,
                                       prefer_role_button=prefer_role_button,
                                       exact=exact, wait_ms=wait_ms)
        if ok:
            return True
    return False

async def _click_chat_tab_force(page) -> bool:
    return await open_chat_tab(page)

async def escape_register_if_needed(page) -> None:
    try:
        chat_url = getattr(page, "_target_chat_url", None)

        try:
            create_visible = False
            for t in CREATE_COMPANION_TEXTS:
                loc = page.get_by_text(t, exact=False)
                if await loc.count() > 0:
                    create_visible = True
                    break
            if create_visible and not await is_login_gate_present(page):
                if not getattr(page, "_companion_refreshed_once", False):
                    log.info("[NAV] Видно 'Создать моего компаньона' → один reload")
                    setattr(page, "_companion_refreshed_once", True)
                    await page_reload_governed(page, reason="companion_seen")
                    return
                elif not getattr(page, "_companion_opened_chat_url_once", False) and chat_url:
                    log.info("[NAV] 'Создать…' снова → открываю чат по ссылке аккаунта")
                    setattr(page, "_companion_opened_chat_url_once", True)
                    await smart_goto(page, chat_url, retries=2)
                    return
        except Exception:
            pass

        if "/register" in (page.url or ""):
            reg_btn = page.get_by_role("button", name=START_REG_TEXT)
            if await reg_btn.count() > 0:
                log.info("[REGISTER] Видна 'Начать регистрацию' — жду ручной вход.")
                return
            await _click_once_by_text(page, START_CHAT_TEXT, "_clicked_start_chat")

            await _click_once_any_by_texts(
                page, CHAT_TAB_TEXTS, "_clicked_chat_tab",
                prefer_role_button=True, exact=True
            )

        await _click_once_any_by_texts(page, RETURN_TO_EVENT_TEXTS, "_clicked_return_to_event")

    except Exception:
        pass

# ================== ЛОГИН / ОТКРЫТИЕ ЧАТА ==================
async def wait_manual_login(page) -> None:
    # Ждём открытия чата с тайм‑аутом 5 минут. Если за это время чат не
    # появился, wait_until_in_chat сгенерирует TimeoutError, и внешний
    # вызов сможет перезапустить браузер.
    await wait_until_in_chat(page, log_every_s=5, max_wait_s=300)

async def wait_chat_open_or_reload(page, *, _from_watchdog: bool = False) -> None:
    # количество безуспешных попыток нажать 'Чат' и дождаться появления поля ввода
    chat_click_attempts = 0
    while True:
        if await is_login_gate_present(page):
            # если чат не открыт в течение 5 минут, wait_until_in_chat
            # сгенерирует TimeoutError, которую внешний код может поймать
            await wait_until_in_chat(page, log_every_s=5, max_wait_s=300)
            return

        await escape_register_if_needed(page)
        clicked = await _click_chat_tab_force(page)

        if await detect_ai_unavailable(page):
            await cooldown_and_reload(page, "в ожидании открытия чата")
        try:
            await page.wait_for_selector(TEXTAREA_SEL, timeout=35000)
            if await detect_generic_challenge(page):
                log.warning("[CHALLENGE] во время ожидания чата → backoff")
                GLOBAL_SEND_COORD.apply_backoff(BACKOFF_BASE_S)
                await asyncio.sleep(BACKOFF_BASE_S)
            return
        except PlaywrightTimeout:
            if _from_watchdog:
                return
            if clicked:
                chat_click_attempts += 1
                # если несколько раз пытались нажать 'Чат', но поле ввода так и не появилось,
                # считаем, что кнопка не работает → генерируем ChatNotFoundError (не для watchdog)
                if chat_click_attempts >= 3 and not _from_watchdog:
                    log.info("[NAV] Несколько попыток открыть 'Чат' не привели к появлению textarea")
                    raise ChatNotFoundError("'Чат/Chat' не открылся после многократных кликов")
                await asyncio.sleep(5)
                continue
            await page_reload_governed(page, reason="chat_open_timeout")
            await escape_register_if_needed(page)

# ================== БАЛАНСЫ / EQ / ОТПРАВКА / ДРУГОЕ ==================
_number_re = re.compile(r"\d{1,4}(?:[.,]\d{1,2})?")

def _parse_header_numbers(text: str) -> Tuple[Optional[float], Optional[int]]:
    text = " ".join(text.split())
    nums = [(m.group(0), m.start()) for m in _number_re.finditer(text)]
    if not nums:
        return None, None
    tokens_val, attempts_val = None, None
    for i, (s, _) in enumerate(nums):
        if "." in s or "," in s:
            try:
                tokens_val = float(s.replace(",", "."))
            except ValueError:
                continue
            for j in range(i + 1, len(nums)):
                ss, _ = nums[j]
                if "." not in ss and "," not in ss:
                    try:
                        attempts_val = int(ss)
                    except ValueError:
                        pass
                    break
            break
    if tokens_val is None:
        try:
            tokens_val = float(nums[0][0].replace(",", "."))
        except ValueError:
            pass
        if len(nums) > 1:
            try:
                attempts_val = int(nums[1][0])
            except ValueError:
                pass
    return tokens_val, attempts_val

async def _read_header_balances(page) -> Tuple[Optional[float], Optional[int]]:
    header_text = ""
    try:
        header = page.locator("header")
        if await header.count() > 0:
            header_text = (await header.first.inner_text()).strip()
    except:
        pass
    if not header_text:
        try:
            banner = page.locator("[role='banner']")
            if await banner.count() > 0:
                header_text = (await banner.first.inner_text()).strip()
        except:
            pass
    if not header_text:
        return None, None
    return _parse_header_numbers(header_text)

async def _read_balances_precise(page) -> Tuple[Optional[float], Optional[int]]:
    js = r"""
    (() => {
      function parseFloatSafe(s){
        if(!s) return null;
        const cleaned = String(s).replace(/[^0-9.,-]/g, "").replace(",", ".");
        if(!cleaned) return null;
        const v = parseFloat(cleaned);
        return Number.isFinite(v) ? v : null;
      }
      function parseIntSafe(s){
        if(!s) return null;
        const cleaned = String(s).replace(/[^0-9-]/g, "");
        if(!cleaned) return null;
        const v = parseInt(cleaned, 10);
        return Number.isFinite(v) ? v : null;
      }
      const root = document.querySelector("header,[role='banner']");
      if(!root) return {tk:null, at:null};
      let tk = null, at = null;
      const tokImg = root.querySelector("img[src*='tokens_icon'],img[alt='tokens']");
      if (tokImg && tokImg.nextElementSibling) { tk = parseFloatSafe(tokImg.nextElementSibling.textContent); }
      const moneyImg = root.querySelector("img[src*='money_icon'],img[alt='money'],img[alt='chat_icon']");
      if (moneyImg && moneyImg.nextElementSibling) { at = parseIntSafe(moneyImg.nextElementSibling.textContent); }
      return {tk, at};
    })()
    """
    try:
        res = await page.evaluate(js)
        tk = res.get("tk", None)
        at = res.get("at", None)
        if tk is not None or at is not None:
            return tk, at
    except Exception:
        pass
    return None, None

async def _read_balances(page) -> Tuple[Optional[float], Optional[int]]:
    tk, at = await _read_balances_precise(page)
    if tk is None and at is None:
        tk, at = await _read_header_balances(page)
    return tk, at

async def wait_tokens_and_attempts(page, url: str,
                                   timeout_ms: int = BALANCE_TIMEOUT_MS,
                                   reloads: int = BALANCE_RELOADS) -> Tuple[Optional[float], Optional[int]]:
    step = 300

    def ok_pair(tk, at):
        if tk is None or at is None:
            return False
        if BALANCE_ZERO_BOTH_INVALID and float(tk) == 0.0 and int(at) == 0:
            return False
        return True

    async def stable_read_if_ok(tk1, at1):
        await asyncio.sleep(BALANCE_STABLE_DELAY_MS / 1000)
        tk2, at2 = await _read_balances(page)
        if ok_pair(tk2, at2) and tk1 == tk2 and at1 == at2:
            log.info("[BALANCE] токены=%s, попытки=%s", tk2, at2)
            return tk2, at2
        return None, None

    async def grace_until_attempts_present():
        grace = ATTEMPTS_GRACE_MS
        while grace > 0:
            await asyncio.sleep(min(ATTEMPTS_GRACE_STEP_MS, grace) / 1000)
            tk_g, at_g = await _read_balances(page)
            if at_g is not None and at_g != 0:
                res_tk, res_at = await stable_read_if_ok(tk_g, at_g)
                if res_tk is not None:
                    return res_tk, res_at
            grace -= ATTEMPTS_GRACE_STEP_MS
        return None, None

    async def zero_stability_window():
        remain = ATTEMPTS_ZERO_STABILITY_MS
        last_tk, last_at = None, None
        while remain > 0:
            tk_z, at_z = await _read_balances(page)
            last_tk, last_at = tk_z, at_z
            if at_z is None or at_z != 0:
                return None, None
            await asyncio.sleep(min(ATTEMPTS_GRACE_STEP_MS, remain) / 1000)
            remain -= ATTEMPTS_GRACE_STEP_MS
        res_tk, res_at = await stable_read_if_ok(last_tk, last_at)
        return (res_tk, res_at) if res_tk is not None else (None, None)

    budget = timeout_ms
    seen_reload_count_for_zero = 0

    while budget > 0:
        tk, at = await _read_balances(page)

        if tk is not None and (at is None):
            got = await grace_until_attempts_present()
            if got != (None, None):
                return got

        if tk is not None and at == 0:
            got = await grace_until_attempts_present()
            if got != (None, None):
                return got
            if seen_reload_count_for_zero >= ATTEMPTS_ZERO_ACCEPT_AFTER_RELOADS or reloads == 0:  # Добавлено or reloads==0 для нулевых попыток без reload
                res_tk, res_at = await zero_stability_window()
                if res_tk is not None:
                    log.info("[BALANCE] принимаем стабильные попытки=0 (после reload-ов).")
                    return res_tk, res_at

        if ok_pair(tk, at) and (at is None or at != 0):
            res_tk, res_at = await stable_read_if_ok(tk, at)
            if res_tk is not None:
                return res_tk, res_at

        await asyncio.sleep(step / 1000)
        budget -= step

    for i in range(reloads):
        log.info("[BALANCE] нет валидных чисел → reload %s/%s", i+1, reloads)
        try:
            await page_reload_governed(page, reason="balance_read")
            await escape_register_if_needed(page)
            if await detect_generic_challenge(page):
                log.warning("[CHALLENGE] после reload (баланс) → backoff")
                GLOBAL_SEND_COORD.apply_backoff(BACKOFF_BASE_S)
                await asyncio.sleep(BACKOFF_BASE_S)
        except ReloadTimeoutError:
            raise
        except Exception:
            pass

        seen_reload_count_for_zero += 1
        budget = timeout_ms
        while budget > 0:
            tk, at = await _read_balances(page)

            if tk is not None and (at is None):
                got = await grace_until_attempts_present()
                if got != (None, None):
                    return got

            if tk is not None and at == 0:
                got = await grace_until_attempts_present()
                if got != (None, None):
                    return got
                if seen_reload_count_for_zero >= ATTEMPTS_ZERO_ACCEPT_AFTER_RELOADS:
                    res_tk, res_at = await zero_stability_window()
                    if res_tk is not None:
                        log.info("[BALANCE] принимаем стабильные попытки=0 (после reload-ов).")
                        return res_tk, res_at

            if ok_pair(tk, at) and (at is None or at != 0):
                res_tk, res_at = await stable_read_if_ok(tk, at)
                if res_tk is not None:
                    return res_tk, res_at

            await asyncio.sleep(step / 1000)
            budget -= step

    return None, None

EQ_NUM_RE = re.compile(r"\d+")

async def get_latest_eq_value(page) -> Optional[int]:
    try:
        val_loc = page.locator("div:has(> img[alt='eq_icon']) span.text-white").last
        txt = (await val_loc.inner_text(timeout=800)).strip()
        m = EQ_NUM_RE.search(txt)
        if m:
            return int(m.group(0))
    except Exception:
        pass
    return None

async def click_reeval_if_present(page) -> None:
    try:
        reeval = page.get_by_text(REEVAL_TEXT, exact=False)
        if await reeval.count() > 0:
            async with _ActionScope(page, "click_reeval"):
                await reeval.first.click(timeout=1000)
            log.info("[EQ] нажали 'Переоценка'")
            await asyncio.sleep(1.0)
    except Exception:
        pass

async def wait_eq_for_last_message(page, url: str,
                                   timeout_ms: int = EQ_TIMEOUT_MS,
                                   reloads: int = EQ_RELOADS) -> bool:
    step = 400

    async def once(budget_ms: int) -> Optional[int]:
        budget = budget_ms
        while budget > 0:
            if await detect_ai_unavailable(page):
                await cooldown_and_reload(page, "в ожидании EQ")
                return None
            await click_reeval_if_present(page)
            val = await get_latest_eq_value(page)
            if val is not None:
                log.info("[EQ] получена оценка: %s", val)
                return val
            await asyncio.sleep(step / 1000)
            budget -= step
        return None

    val = await once(timeout_ms)
    if val is not None:
        return True

    for i in range(reloads):
        log.info("[EQ] оценки нет → reload %s/%s", i+1, reloads)
        try:
            await page_reload_governed(page, reason="eq_wait")
            await escape_register_if_needed(page)
            if await detect_generic_challenge(page):
                log.warning("[CHALLENGE] после reload (EQ) → backoff")
                GLOBAL_SEND_COORD.apply_backoff(BACKOFF_BASE_S)
                await asyncio.sleep(BACKOFF_BASE_S)
        except ReloadTimeoutError:
            raise
        except Exception:
            pass
        try:
            await wait_chat_open_or_reload(page)
            await asyncio.sleep(0.6)
        except Exception:
            pass
        val = await once(timeout_ms)
        if val is not None:
            return True

    log.warning("[EQ] оценка так и не появилась.")
    return False

async def is_daily_limit_reached(page) -> bool:
    try:
        elem = page.get_by_text(LIMIT_MESSAGE, exact=False)
        if await elem.count() > 0:
            try:
                await elem.first.wait_for(state="visible", timeout=6000)
                return True
            except PlaywrightTimeout:
                pass
        body_text = (await page.locator("body").inner_text()).strip()
        if LIMIT_MESSAGE in body_text:
            return True
    except:
        pass
    return False

async def send_one(page, text: str) -> Tuple[bool, bool]:
    """
    Возвращает:
      (ok: bool, limit_reached: bool)
    ok = True  -> сообщение отправлено
    limit_reached = True -> упёрлись в дневной лимит
    """
    # 0) Локаторы (опираемся на уже вставленный TEXTAREA_SEL и _textarea_any)
    textarea = _textarea_any(page)
    form = page.locator(f"form:has({TEXTAREA_SEL})")
    send_btn = form.locator("button[type='submit']")

    # 1) Предубеждающие проверки
    if await detect_ai_unavailable(page):
        await cooldown_and_reload(page, "перед отправкой сообщения")

    if await is_daily_limit_reached(page):
        return False, True

    # 2) Гарантируем видимость поля ввода
    try:
        await textarea.first.wait_for(state="visible", timeout=5000)
    except PlaywrightTimeout:
        # если скрыто — на некоторых UI помогает «клик» в область формы
        try:
            await form.click(timeout=1000)
            await textarea.first.wait_for(state="visible", timeout=3000)
        except Exception:
            pass

    # 3) Ввод текста (с безопасным fallback)
    try:
        # очистим на всякий случай
        try:
            await textarea.fill("")
        except Exception:
            pass

        await textarea.click()
        await textarea.fill(text)
    except Exception:
        # крайний случай — печать клавиатурой
        await textarea.click()
        await page.keyboard.type(text, delay=12)

    # 4) Отправка — кнопкой, либо Enter как запасной вариант
    WAIT_ENABLE_SEND_MS = 4000  # используй свой, если он у тебя уже объявлен выше
    clicked = False
    try:
        await page.wait_for_selector("form button[type='submit']:not([disabled])",
                                     timeout=WAIT_ENABLE_SEND_MS)
        await send_btn.click()
        clicked = True
    except PlaywrightTimeout:
        # некоторые чаты отправляют по Enter
        await page.keyboard.press("Enter")

    # 5) Подождём «стабилизацию» сети/UI, но не упираемся жёстко
    try:
        await page.wait_for_load_state("networkidle", timeout=5000)
    except PlaywrightTimeout:
        pass

    # 6) Пост-проверки на лимит (после отправки иногда всплывает баннер лимита)
    if await is_daily_limit_reached(page):
        return False, True

    # Если до сюда дошли без исключений — считаем, что всё ок
    return True, False

async def _last_msg_id_and_text(page) -> Tuple[Optional[str], Optional[str]]:
    try:
        boxes = page.locator("div[id^='msg-']")
        n = await boxes.count()
        if n == 0: return None, None
        node = boxes.nth(n - 1)
        msg_id = await node.get_attribute("id")
        txt = (await node.inner_text(timeout=800)).strip()
        return msg_id, _norm(txt)
    except Exception:
        return None, None

async def wait_new_partner_message(page, after_msg_id: Optional[str], my_sent_text: str,
                                   timeout_ms: int = 60000) -> Optional[str]:
    # Оптимизировано: wait_for_selector вместо поллинга
    try:
        probe = my_sent_text.strip()[:40]
        await page.wait_for_selector(f"div[id^='msg-']:not([id='{after_msg_id or ''}']):not(:has-text('{probe}'))", timeout=timeout_ms)
        _, last_txt = await _last_msg_id_and_text(page)
        return last_txt
    except PlaywrightTimeout:
        return None
    except Exception:
        return None

# ================== PRECHECK ==================
async def precheck_in_order(accounts: List[Account]) -> List[Account]:
    def _account_label(a: Account) -> str:
        try:
            seg = (a.chat_url.rstrip("/").split("/")[-1] or "").strip()
        except Exception:
            seg = ""
        return seg or a.user_data_dir

    good_accounts: List[Account] = []
    async with async_playwright() as p:
        for acc in accounts:
            log.info("[PRECHECK] %s → %s", acc.user_data_dir, BASE_CHAT_URL)
            # Две попытки открыть чат для аккаунта. Если первая завершается
            # TimeoutError (истекло 5 минут ожидания), перезапускаем
            # браузер и пробуем ещё раз. Иные ошибки не ретраим.
            retries_left = 2
            while retries_left > 0:
                browser = None
                page = None
                try:
                    heal_profile_dir(acc.user_data_dir)
                    browser = await launch_persistent_context_adapter(
                        p, user_data_dir=acc.user_data_dir, headless=HEADLESS
                    )
                    try:
                        setattr(browser, "_banner_label", _account_label(acc))
                    except Exception:
                        pass

                    page = await get_or_create_main_page(browser, BASE_CHAT_URL)
                    try:
                        setattr(page, "_banner_label", getattr(browser, "_banner_label", None))
                    except Exception:
                        pass
                    attach_block_watchers(page)
                    # запускаем watchdog только если no_throttle=False (т.е. окно может требовать фокуса)
                    try:
                        no_thr = getattr(page.context, "_no_throttle", False)
                    except Exception:
                        no_thr = False
                    if not no_thr:
                        await start_user_interference_watchdog(page)
                    setattr(page, "_target_chat_url", acc.chat_url)

                    await smart_goto(page, BASE_CHAT_URL)
                    await escape_register_if_needed(page)
                    # ожидание открытия чата с тайм‑аутом 5 минут
                    await wait_manual_login(page)

                    # если мы дошли до этой точки — чат открыт
                    tk, at = await wait_tokens_and_attempts(page, BASE_CHAT_URL)
                    while tk is None or at is None:
                        log.info("[PRECHECK] %s: баланс не найден. Жду появления…", acc.user_data_dir)
                        tk, at = await wait_tokens_and_attempts(page, BASE_CHAT_URL)

                    acc.tokens_cached = float(tk or 0.0)
                    acc.attempts_cached = max(0, int(at or 0))
                    REPORTS[acc.user_data_dir] = AccountReport(
                        user_data_dir=acc.user_data_dir,
                        chat_url=acc.chat_url,
                        start_tokens=acc.tokens_cached,
                        start_attempts=acc.attempts_cached,
                        status="pending"
                    )

                    if acc.attempts_cached > 0:
                        good_accounts.append(acc)
                    else:
                        log.info("[PRECHECK] %s: попыток %s → исключён", acc.user_data_dir, acc.attempts_cached)
                        rep = REPORTS.get(acc.user_data_dir)
                        if rep:
                            rep.status = "skipped_no_attempts"
                            rep.end_tokens = acc.tokens_cached
                            rep.end_attempts = acc.attempts_cached
                    # Успешно завершили предчек — выходим из ретраев
                    break
                except TimeoutError as te:
                    # Чат не открылся в течение 5 минут: закрываем браузер и пробуем ещё раз
                    retries_left -= 1
                    log.warning(
                        "[PRECHECK] %s: чат не открылся за 5 минут (TimeoutError) → перезапуск (%s оставалось)",
                        acc.user_data_dir, retries_left)
                    try:
                        if page:
                            cancel_user_interference_watchdog(page)
                    except Exception:
                        pass
                    try:
                        if browser:
                            await browser.close()
                    except Exception:
                        pass
                    # если попыток больше нет, считаем это ошибкой
                    if retries_left <= 0:
                        log.error("[PRECHECK] %s: не удалось открыть чат после перезапуска", acc.user_data_dir)
                        rep = REPORTS.get(acc.user_data_dir)
                        if rep:
                            rep.status = "error"
                            rep.errors.append("chat_open_timeout")
                    continue
                except Exception as e:
                    # Любая другая критическая ошибка
                    log.error("[PRECHECK] %s критическая ошибка: %s", acc.user_data_dir, e, exc_info=True)
                    rep = REPORTS.get(acc.user_data_dir)
                    if rep:
                        rep.status = "error"
                        rep.errors.append(f"precheck_exception: {e}")
                    # прекращаем попытки для этого аккаунта
                    break
                finally:
                    # закрываем браузер в конце итерации, если он не будет использован дальше
                    try:
                        if page:
                            try:
                                cancel_user_interference_watchdog(page)
                            except Exception:
                                pass
                    except Exception:
                        pass
                    try:
                        if browser:
                            await browser.close()
                    except Exception:
                        pass
                # конец while
            # end of while retry loop
        # end for acc
    return good_accounts

# ================== ОСНОВНОЙ ЦИКЛ ПО АККАУНТУ ==================
async def run_for_account(acc: Account, max_local_limit: int, *, headless_override: Optional[bool] = None) -> Tuple[int, bool, bool, str, bool, float, int]:
    def _account_label(a: Account) -> str:
        try:
            seg = (a.chat_url.rstrip("/").split("/")[-1] or "").strip()
        except Exception:
            seg = ""
        return seg or a.user_data_dir

    sent = 0
    finished_cleanly = False
    reason = ""
    limit_hit = False
    final_tokens = acc.tokens_cached
    final_attempts = acc.attempts_cached

    try:
        async with async_playwright() as p:
            # даём до двух попыток открыть чат для данного аккаунта (цикл вместо рекурсии для избежания stack overflow)
            retries_left = 2
            hflag = HEADLESS if headless_override is None else headless_override
            browser = None
            page = None
            while retries_left > 0:
                try:
                    browser = await launch_persistent_context_adapter(
                        p, user_data_dir=acc.user_data_dir, headless=hflag
                    )
                    try:
                        setattr(browser, "_banner_label", _account_label(acc))
                    except Exception:
                        pass

                    page = await get_or_create_main_page(browser, BASE_CHAT_URL)
                    attach_block_watchers(page)
                    # запускаем watchdog только если no_throttle=False
                    try:
                        no_thr = getattr(page.context, "_no_throttle", False)
                    except Exception:
                        no_thr = False
                    if not no_thr:
                        await start_user_interference_watchdog(page)
                    setattr(page, "_target_chat_url", acc.chat_url)

                    phase = stable_phase(acc.user_data_dir + "|" + acc.chat_url, max_shift_s=10.0)
                    await asyncio.sleep(phase)

                    await smart_goto(page, BASE_CHAT_URL)
                    await escape_register_if_needed(page)
                    # Если в headless‑режиме уже виден логин‑гейт, переключаем аккаунт
                    # в видимый режим сразу, не дожидаясь тайм‑аута
                    if headless_override is None and HEADLESS:
                        try:
                            if await is_login_gate_present(page):
                                log.info(
                                    "[%s] Обнаружен логин‑гейт в headless-режиме → открываю видимый браузер для входа",
                                    acc.user_data_dir
                                )
                                try:
                                    cancel_user_interference_watchdog(page)
                                except Exception:
                                    pass
                                try:
                                    await browser.close()
                                except Exception:
                                    pass
                                hflag = False  # Переключаем в видимый
                                continue  # Retry с новым hflag
                        except Exception:
                            pass
                    # ожидание открытия чата с тайм‑аутом 5 минут
                    await wait_chat_open_or_reload(page)

                    # если дошли до этой точки — чат открыт, выходим из цикла ретраев
                    break
                except TimeoutError:
                    # чат не открылся за 5 минут
                    retries_left -= 1
                    log.warning(
                        "[%s] чат не открылся за 5 минут → перезапуск (%s попыток)",
                        acc.user_data_dir, retries_left
                    )
                    try:
                        if page:
                            cancel_user_interference_watchdog(page)
                    except Exception:
                        pass
                    try:
                        if browser:
                            await browser.close()
                    except Exception:
                        pass
                    # Первая неудача в headless-режиме → переключаем в видимый режим
                    if retries_left <= 0:
                        reason = "chat_open_timeout"
                        return sent, False, False, reason, limit_hit, final_tokens, final_attempts
                    continue

                except ChatNotFoundError:
                    # Кнопка 'Чат/Chat' не найдена — переходим в видимый режим для ручного входа
                    log.warning(
                        "[%s] Кнопка 'Чат/Chat' не найдена селекторами → переключаемся в видимый режим",
                        acc.user_data_dir
                    )
                    try:
                        if page:
                            cancel_user_interference_watchdog(page)
                    except Exception:
                        pass
                    try:
                        if browser:
                            await browser.close()
                    except Exception:
                        pass
                    if headless_override is None and HEADLESS:
                        hflag = False
                        continue
                    else:
                        # если уже в видимом режиме, фиксируем ошибку и завершаем
                        reason = "chat_button_not_found"
                        return sent, False, True, reason, limit_hit, final_tokens, final_attempts

                except ReloadTimeoutError:
                    # Перезагрузка страницы завершилась неудачно — также переходим в видимый режим
                    log.warning(
                        "[%s] Неудачная перезагрузка страницы → переключаемся в видимый режим",
                        acc.user_data_dir
                    )
                    try:
                        if page:
                            cancel_user_interference_watchdog(page)
                    except Exception:
                        pass
                    try:
                        if browser:
                            await browser.close()
                    except Exception:
                        pass
                    if headless_override is None and HEADLESS:
                        hflag = False
                        continue
                    else:
                        reason = "reload_fail"
                        return sent, False, True, reason, limit_hit, final_tokens, final_attempts

                except Exception as e:
                    # Любая другая критическая ошибка до начала отправки сообщений
                    reason = f"pre_send_exception: {e}"
                    log.error("[%s] критическая ошибка перед отправкой: %s", acc.user_data_dir, e, exc_info=True)
                    try:
                        if page:
                            cancel_user_interference_watchdog(page)
                    except Exception:
                        pass
                    try:
                        if browser:
                            await browser.close()
                    except Exception:
                        pass
                    return sent, False, True, reason, limit_hit, final_tokens, final_attempts
            # если мы вышли из цикла потому что чату открылся успешно, продолжаем отправку
            # Считываем баланс: токены и попытки. Без этого precheck не выполняется
            try:
                tk0, at0 = await wait_tokens_and_attempts(page, BASE_CHAT_URL)
                while tk0 is None or at0 is None:
                    log.info("[%s] баланс не найден. Жду появления…", acc.user_data_dir)
                    tk0, at0 = await wait_tokens_and_attempts(page, BASE_CHAT_URL)
                acc.tokens_cached = float(tk0 or 0.0)
                acc.attempts_cached = max(0, int(at0 or 0))
                # обновляем стартовые значения в отчёте
                rep0 = REPORTS.get(acc.user_data_dir)
                if rep0:
                    rep0.start_tokens = acc.tokens_cached
                    rep0.start_attempts = acc.attempts_cached
                # обновляем переменные, используемые в итогах
                final_tokens = acc.tokens_cached
                final_attempts = acc.attempts_cached
            except Exception as e:
                log.warning("[%s] ошибка чтения баланса: %s", acc.user_data_dir, e)
                # если не удалось прочитать баланс, попытки остаются по умолчанию (0)
                acc.tokens_cached = acc.tokens_cached or 0.0
                acc.attempts_cached = acc.attempts_cached or 0
                final_tokens = float(acc.tokens_cached)
                final_attempts = int(acc.attempts_cached)
            attempts = max(0, int(acc.attempts_cached or 0))
            if attempts <= 0:
                reason = "no_attempts_cached"
                try: cancel_user_interference_watchdog(page)
                except Exception: pass
                await browser.close()
                return 0, False, False, reason, False, final_tokens, final_attempts
            remaining = await remaining_cap_for(acc)  # Async call
            if remaining <= 0:
                reason = "hard_cap_reached"
                log.info("[%s] Жёсткий лимит %s уже исчерпан (used=%s).",
                         acc.user_data_dir, GLOBAL_HARD_CAP, await get_sent_so_far(acc))  # Async
                try: cancel_user_interference_watchdog(page)
                except Exception: pass
                await browser.close()
                return 0, True, False, reason, False, final_tokens, final_attempts
            # локальный лимит аккаунта, попытки сайта и остаток по капу
            plan = min(
                max_local_limit,
                attempts,
                remaining
            )
            if plan <= 0:
                reason = "plan_zero"
                try: cancel_user_interference_watchdog(page)
                except Exception: pass
                await browser.close()
                return 0, False, False, reason, False, final_tokens, final_attempts
            last_partner_text_cache: Optional[str] = None
            for _ in range(plan):
                # Перепроверим кап прямо перед попыткой
                if await remaining_cap_for(acc) <= 0:  # Async
                    log.info("[%s] Достигли жёсткого лимита %s перед отправкой.", acc.user_data_dir, GLOBAL_HARD_CAP)
                    limit_hit = True
                    break
                if await is_login_gate_present(page):
                    await wait_until_in_chat(page, log_every_s=5, max_wait_s=300)
                if last_partner_text_cache:
                    partner_text = last_partner_text_cache
                else:
                    _, partner_text = await _last_msg_id_and_text(page)
                    if not partner_text:
                        partner_text = "Продолжай со мной этот тёплый разговор у пруда."
                my_reply = await aopenai_romantic_300(partner_text)
                # Ещё одна защита капа, вдруг параллельно сессия изменила счётчик
                if await remaining_cap_for(acc) <= 0:  # Async
                    log.info("[%s] Достигли жёсткого лимита %s непосредственно перед send_one.",
                             acc.user_data_dir, GLOBAL_HARD_CAP)
                    limit_hit = True
                    break
                ok, limit = await send_one(page, my_reply)
                if limit:
                    log.info("[%s] лимит исчерпан (баннер).", acc.user_data_dir)
                    limit_hit = True
                    break
                if not ok:
                    reason = "send_failed"
                    try: cancel_user_interference_watchdog(page)
                    except Exception: pass
                    await browser.close()
                    return sent, False, True, reason, limit_hit, final_tokens, final_attempts
                # === КРИТИЧЕСКОЕ МЕСТО: фиксируем факт отправки немедленно ===
                used_now = await inc_sent(acc, 1)  # Async
                sent += 1
                log.info("[%s] Персистентный счётчик: %s/%s (локально sent=%s).",
                         acc.user_data_dir, used_now, GLOBAL_HARD_CAP, sent)
                # Дальше можно ждать EQ, но кап уже учтён и сохранён
                got_eq = await wait_eq_for_last_message(page, BASE_CHAT_URL, timeout_ms=EQ_TIMEOUT_MS, reloads=EQ_RELOADS)
                if not got_eq:
                    reason = "eq_not_received"
                    try: cancel_user_interference_watchdog(page)
                    except Exception: pass
                    await browser.close()
                    return sent, False, True, reason, limit_hit, final_tokens, final_attempts
                log.info("[%s] %s → отправлено %s/%s (план=%s, remaining_cap=%s)",
                         acc.user_data_dir, acc.chat_url, sent, plan, plan, await remaining_cap_for(acc))  # Async
                last_id_now, _ = await _last_msg_id_and_text(page)
                partner_new = await wait_new_partner_message(page, last_id_now, my_reply, timeout_ms=60000)
                last_partner_text_cache = partner_new
                if sent < plan:
                    extra = stable_phase(acc.user_data_dir) * 0.4
                    jitter = random.uniform(*JITTER_SECONDS) + random.uniform(0, 1.2) + extra
                    await asyncio.sleep(DELAY_SECONDS + jitter)
            else:
                finished_cleanly = True
            tk_end, at_end = await wait_tokens_and_attempts(page, BASE_CHAT_URL)
            final_tokens = float(tk_end or 0.0)
            final_attempts = max(0, int(at_end or 0))
            if sent > 0 or limit_hit:
                log.info("[%s] Пауза %s c перед закрытием вкладки.", acc.user_data_dir, POST_SWITCH_WAIT_SECONDS)
                await asyncio.sleep(POST_SWITCH_WAIT_SECONDS)
            try: cancel_user_interference_watchdog(page)
            except Exception: pass
            await browser.close()
            return sent, finished_cleanly, False, reason, limit_hit, final_tokens, final_attempts
    except Exception as e:
        reason = f"exception: {e}"
        log.error("[%s] критическая ошибка: %s → requeue", acc.user_data_dir, e, exc_info=True)
        return sent, False, True, reason, limit_hit, final_tokens, final_attempts

# ================== PIPELINE / ОТЧЁТ / MAIN ==================
def split_round_robin(items: List[Account], n: int) -> List[List[Account]]:
    buckets = [[] for _ in range(max(1, n))]
    for i, it in enumerate(items):
        buckets[i % max(1, n)].append(it)
    return buckets

def save_reports():
    with open(REPORT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter=";")
        writer.writerow([
            "user_data_dir", "chat_url",
            "start_tokens", "start_attempts",
            "end_tokens", "end_attempts",
            "sent", "limit_hit", "status", "errors"
        ])
        for rep in REPORTS.values():
            writer.writerow([
                rep.user_data_dir, rep.chat_url,
                f"{rep.start_tokens:.2f}", rep.start_attempts,
                f"{rep.end_tokens:.2f}", rep.end_attempts,
                rep.sent, int(rep.limit_hit), rep.status, " | ".join(rep.errors)
            ])

    total_start_tokens = sum(r.start_tokens for r in REPORTS.values()) if REPORTS else 0.0
    total_end_tokens = sum(r.end_tokens for r in REPORTS.values()) if REPORTS else 0.0
    problems = [r for r in REPORTS.values() if r.status in ("partial", "error") or r.errors]

    lines = []
    lines.append("Vibely Run Report\n")
    lines.append(f"Accounts processed: {len(REPORTS)}\n")
    lines.append(f"Total tokens start: {total_start_tokens:.2f}")
    lines.append(f"Total tokens end  : {total_end_tokens:.2f}\n")

    lines.append("Per-account summary:")
    for r in REPORTS.values():
        lines.append(
            f"- {r.user_data_dir}: start {r.start_tokens:.2f}/{r.start_attempts}, "
            f"end {r.end_tokens:.2f}/{r.end_attempts}, sent={r.sent}, "
            f"limit={r.limit_hit}, status={r.status}, errors={'; '.join(r.errors) if r.errors else '-'}"
        )

    if problems:
        lines.append("\nAccounts with issues:")
        for r in problems:
            lines.append(f"* {r.user_data_dir} → status={r.status}; errors={'; '.join(r.errors) if r.errors else '-'}")
    else:
        lines.append("\nNo problematic accounts detected.")

    Path(REPORT_TXT).write_text("\n".join(lines), encoding="utf-8")
    log.info("[REPORT] Сохранено %s и %s", REPORT_CSV, REPORT_TXT)

async def run_pipeline_for_accounts(worker_id: int, accounts_subset: List[Account]) -> None:
    log.info("[PIPE] Воркер %s: старт", worker_id)
    worker_phase = (worker_id - 1) * 0.9 + random.uniform(0, 0.9)
    await asyncio.sleep(worker_phase)

    # В новом режиме предварительная проверка не выполняется отдельно: аккаунты
    # идут напрямую в основную очередь, где будет выполнен логин, проверка баланса
    # и основной цикл отправки сообщений. Мы не вызываем precheck_in_order().
    ordered_accounts = list(accounts_subset)

    queue = list(ordered_accounts)
    while queue:
        acc = queue.pop(0)
        per_acc_limit = acc.messages_limit if acc.messages_limit is not None else DEFAULT_MESSAGES_PER_ACCOUNT

        # ====== БЛОК 5 (начало) ======
        remaining = await remaining_cap_for(acc)  # Async
        if remaining <= 0:
            log.info("[PIPE] %s: жёсткий лимит уже исчерпан — пропускаем аккаунт.", acc.user_data_dir)
            rep = REPORTS.get(acc.user_data_dir)
            if rep:
                rep.status = "success"
            continue
        per_acc_limit = min(per_acc_limit, remaining)

        # Создаём запись в отчётах, если ещё нет. start_tokens/start_attempts
        # будут обновлены в run_for_account после чтения баланса.
        if acc.user_data_dir not in REPORTS:
            REPORTS[acc.user_data_dir] = AccountReport(
                user_data_dir=acc.user_data_dir,
                chat_url=acc.chat_url,
                start_tokens=0.0,
                start_attempts=0,
                status="pending"
            )

        sent, clean, requeue, reason, limit_hit, tk_end, at_end = await run_for_account(acc, per_acc_limit)

        rep = REPORTS.get(acc.user_data_dir)
        if rep:
            rep.sent = sent
            rep.limit_hit = limit_hit
            rep.end_tokens = tk_end
            rep.end_attempts = at_end
            if requeue:
                rep.status = "partial"
                if reason:
                    rep.errors.append(reason)
            else:
                rep.status = "success" if clean or limit_hit or sent > 0 else "partial"
                if reason:
                    rep.errors.append(reason)

        if requeue:
            log.warning("[PIPE] Воркер %s: %s → переносим в конец (повтор).", worker_id, acc.user_data_dir)
            queue.append(acc)
            continue

        log.info("[PIPE] Воркер %s: готов аккаунт. В очереди %s.", worker_id, len(queue))

    log.info("[PIPE] Воркер %s: завершён.", worker_id)

async def main():
    if not ACCOUNTS:
        log.error("[MAIN] Список ACCOUNTS пуст. Добавьте аккаунты в коде.")
        save_reports()  # Добавлено: save даже если пусто
        return

    groups = split_round_robin(ACCOUNTS, WORKERS)
    for idx, g in enumerate(groups, 1):
        log.info("[MAIN] Воркер %s: %s аккаунтов (AI-циклы от последнего ответа в чате)", idx, len(g))

    await asyncio.gather(*(run_pipeline_for_accounts(i+1, g) for i, g in enumerate(groups)))
    log.info("== Все воркеры завершили работу. ==")
    save_reports()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as e:
        log.critical("[MAIN] Неперехваченное исключение: %s", e, exc_info=True)
        raise

def cancel_user_interference_watchdog(page):
    try:
        t = getattr(page, "_watchdog_task", None)
        if t and hasattr(t, "cancel") and not t.done():
            t.cancel()
    except Exception:
        pass
    try:
        setattr(page, "_watchdog_task", None)
    except Exception:
        pass
```