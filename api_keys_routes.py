"""/config/api-keys — admin UI for per-user API key management.

Endpoints (all admin-only):

  GET    /config/api-keys                       HTML page
  GET    /config/api-keys/api/users             list users + key counts
  POST   /config/api-keys/api/users             { username, is_admin }
  DELETE /config/api-keys/api/users/{uid}       soft-revoke (cascades to keys)
  GET    /config/api-keys/api/users/{uid}/keys  list keys for one user
  POST   /config/api-keys/api/users/{uid}/keys  { label }   -> show-once raw key
  DELETE /config/api-keys/api/users/{uid}/keys/{kid}        soft-revoke

Last-admin guard: revoking the only admin key (or only admin user)
returns 409. Prevents accidental lockout.

The HTML page is a single-file React-less app: vanilla JS + sessionStorage
bearer (same pattern as /config). Generates keys with a show-once modal —
the raw key is copied to the clipboard, then never retrievable.
"""
from __future__ import annotations

import logging
from typing import Any, Literal

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

import api_keys_store
import web_common
from admin_routes import require_admin_host
from auth import require_admin

logger = logging.getLogger("whisper-api")

router = APIRouter(prefix="/config/api-keys")


# ---------------------------------------------------------------------
# Payloads
# ---------------------------------------------------------------------

class CreateUserIn(BaseModel):
    model_config = {"extra": "forbid"}
    username: str = Field(min_length=1, max_length=128)
    is_admin: bool = False


class CreateKeyIn(BaseModel):
    model_config = {"extra": "forbid"}
    label: str = Field(default="", max_length=128)


# ---------------------------------------------------------------------
# HTML page
# ---------------------------------------------------------------------

@router.get(
    "",
    dependencies=[Depends(require_admin_host)],
    response_class=HTMLResponse,
)
async def api_keys_page() -> HTMLResponse:
    return HTMLResponse(
        web_common.render_page(_API_KEYS_HTML, current="api-keys"),
        headers={"Cache-Control": "no-store"},
    )


# ---------------------------------------------------------------------
# JSON APIs
# ---------------------------------------------------------------------

@router.get(
    "/api/users",
    dependencies=[Depends(require_admin_host), Depends(require_admin)],
)
async def list_users_api() -> JSONResponse:
    users = api_keys_store.list_users()
    # Annotate each user with their active key count for the card header.
    out: list[dict[str, Any]] = []
    for u in users:
        keys = api_keys_store.list_keys(user_id=u["id"])
        out.append({**u, "active_key_count": len(keys)})
    return JSONResponse({
        "users": out,
        "open_mode": not api_keys_store.is_locked_down(),
        "active_admins": api_keys_store.count_active_admins(),
        "active_admin_keys": api_keys_store.count_active_admin_keys(),
    })


@router.post(
    "/api/users",
    dependencies=[Depends(require_admin_host), Depends(require_admin)],
)
async def create_user_api(payload: CreateUserIn) -> JSONResponse:
    try:
        uid = api_keys_store.create_user(payload.username, payload.is_admin)
    except ValueError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
    return JSONResponse({"user_id": uid})


@router.delete(
    "/api/users/{uid}",
    dependencies=[Depends(require_admin_host), Depends(require_admin)],
)
async def revoke_user_api(uid: str) -> JSONResponse:
    user = api_keys_store.get_user(uid)
    if user is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "user not found")
    if user["revoked_ts"] is not None:
        return JSONResponse({"ok": True, "already_revoked": True})
    # Last-admin guard: refuse if this would zero out active admins.
    if user["is_admin"] and api_keys_store.count_active_admins() <= 1:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "refusing to revoke the last admin user — create another admin first",
        )
    api_keys_store.revoke_user(uid)
    return JSONResponse({"ok": True})


@router.get(
    "/api/users/{uid}/keys",
    dependencies=[Depends(require_admin_host), Depends(require_admin)],
)
async def list_user_keys_api(uid: str) -> JSONResponse:
    if api_keys_store.get_user(uid) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "user not found")
    return JSONResponse({"keys": api_keys_store.list_keys(user_id=uid)})


@router.post(
    "/api/users/{uid}/keys",
    dependencies=[Depends(require_admin_host), Depends(require_admin)],
)
async def create_user_key_api(uid: str, payload: CreateKeyIn) -> JSONResponse:
    """Show-once raw key on creation. Subsequent reads via list_user_keys
    never return the raw value."""
    try:
        raw_key, rec = api_keys_store.create_key(uid, label=payload.label)
    except ValueError as e:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(e))
    return JSONResponse({"key": raw_key, "record": rec})


@router.delete(
    "/api/users/{uid}/keys/{kid}",
    dependencies=[Depends(require_admin_host), Depends(require_admin)],
)
async def revoke_key_api(uid: str, kid: str) -> JSONResponse:
    key = api_keys_store.get_key(kid)
    if key is None or key["user_id"] != uid:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "key not found")
    if key["revoked_ts"] is not None:
        return JSONResponse({"ok": True, "already_revoked": True})
    user = api_keys_store.get_user(uid)
    # Last-admin-key guard.
    if user and user["is_admin"] and \
            api_keys_store.count_active_admin_keys() <= 1:
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "refusing to revoke the last active admin key — generate another"
            " admin key first",
        )
    api_keys_store.revoke_key(kid)
    return JSONResponse({"ok": True})


# ---------------------------------------------------------------------
# HTML template
# ---------------------------------------------------------------------

_API_KEYS_HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>API keys — WhisperAPI</title>
{{SCALE_BOOTSTRAP_HEAD}}
<style>
  :root {
    --bg: #0d1117; --panel: #161b22; --fg: #c9d1d9; --dim: #6e7681;
    --cyan: #79c0ff; --green: #7ee787; --yellow: #f2cc60;
    --red: #ff7b72; --bold: #f0f6fc; --border: #30363d; --input-bg: #0d1117;
    --help: #8b949e;
  }
  html, body { background: var(--bg); color: var(--fg);
    font-family: var(--font-sans); font-size: var(--fs-base); margin: 0; }
  a { color: var(--cyan); }
  header button { background: var(--panel); border: 1px solid var(--border);
    color: var(--fg); padding: 0.25rem 0.625rem; border-radius: 4px;
    cursor: pointer; font: inherit; font-size: var(--fs-sm); }
  header button.primary { color: var(--green); border-color: var(--green); }
  main { padding: 1rem; max-width: 56rem; margin: 0 auto; }
  .banner-open {
    background: #5a2424; color: #fff; padding: 0.6rem 1rem;
    text-align: center; font-weight: 600;
  }
  .card { background: var(--panel); border: 1px solid var(--border);
    border-radius: 4px; padding: 0.75rem 1rem; margin-bottom: 0.75rem; }
  .card h3 { margin: 0 0 0.25rem 0; font-size: var(--fs-lg);
    color: var(--bold); display: flex; align-items: center; gap: 0.6rem; }
  .pill { font-size: var(--fs-xs); padding: 0.075rem 0.5rem;
    border-radius: 999px; border: 1px solid var(--border); color: var(--dim);
    font-weight: normal; }
  .pill.admin { color: var(--yellow); border-color: #4d3e1f; }
  .pill.revoked { color: var(--red); border-color: #5a2424;
    background: #2d1414; }
  .pill.live { color: var(--green); border-color: #1d4f2c; }
  .key-row { display: grid;
    grid-template-columns: minmax(8rem,1fr) 9rem 9rem 9rem auto;
    gap: 0.6rem; align-items: center; padding: 0.3rem 0;
    border-top: 1px solid var(--border); font-size: var(--fs-sm); }
  .key-row:first-child { border-top: none; }
  .key-row .label { color: var(--fg); }
  .key-row .id { color: var(--dim); font-family: var(--font-mono); }
  .key-row .ts { color: var(--dim); font-size: var(--fs-xs); }
  button.danger { color: var(--red); border-color: #5a2424; }
  .toolbar { display: flex; gap: 0.5rem; margin: 0.5rem 0; flex-wrap: wrap; }
  input[type=text], input[type=password] { box-sizing: border-box;
    width: 100%;
    background: var(--input-bg); color: var(--fg);
    border: 1px solid var(--border); border-radius: 3px;
    padding: 0.3rem 0.5rem; font: inherit; font-size: var(--fs-sm); }
  label.row { display: flex; gap: 0.5rem; align-items: center;
    margin: 0.3rem 0; font-size: var(--fs-sm); }
  .modal { position: fixed; inset: 0; display: none;
    align-items: center; justify-content: center;
    background: rgba(0,0,0,0.7); z-index: 100; }
  .modal.show { display: flex; }
  .modal .box { background: var(--panel); border: 1px solid var(--border);
    border-radius: 4px; padding: 1rem 1.2rem;
    width: 28rem; max-width: 92vw; }
  .modal h3 { margin: 0 0 0.5rem 0; color: var(--bold); }
  .modal .raw-key {
    font-family: var(--font-mono); font-size: var(--fs-sm);
    word-break: break-all; padding: 0.5rem; background: var(--input-bg);
    border: 1px solid var(--border); border-radius: 3px;
    margin: 0.5rem 0; user-select: all;
  }
  .modal .actions {
    display: flex; gap: 0.5rem; justify-content: flex-end;
    margin-top: 0.6rem;
  }
  .err { color: var(--red); font-size: var(--fs-sm); margin: 0.4rem 0 0 0; }
  .hint { color: var(--help); font-size: var(--fs-sm); }
  {{NAV_CSS}}
</style></head>
<body>

<div id="open-banner" class="banner-open" style="display:none">
  &#9888; No admin key set &mdash; the server is in OPEN mode and anyone who can
  reach it can use it. Generate the first admin key below.
</div>

<header><div class="header-inner">
  <span class="title">faster-whisper-backend &middot; API keys</span>
  {{NAV}}
  <span class="spacer"></span>
  <span class="wrap-anchor"></span>
  {{SCALE_PICKER}}
  <button id="logout-btn" title="forget API key in this tab">logout</button>
  <button id="reload-btn">reload</button>
</div></header>

<main>
  <div class="card">
    <h3>Add user</h3>
    <div class="toolbar">
      <input id="new-username" type="text" placeholder="username (e.g., Dr. Mueller)"
             style="flex: 1; max-width: 18rem;">
      <label class="row" style="margin: 0;">
        <input id="new-is-admin" type="checkbox"> admin
      </label>
      <button id="add-user-btn" class="primary">+ add user</button>
    </div>
    <p class="hint">
      Usernames are display names only &mdash; nothing about login. Each user
      can hold any number of API keys (one per device is the standard
      pattern).
    </p>
  </div>

  <div id="users-container"></div>
</main>

<!-- Show-once raw key modal -->
<div id="key-modal" class="modal">
  <div class="box">
    <h3>New API key</h3>
    <p>Save this key now &mdash; it will not be shown again. Anyone with the key
    has the same access as <strong id="key-modal-user"></strong>.</p>
    <div class="raw-key" id="key-modal-raw"></div>
    <div class="actions">
      <button id="key-modal-copy">Copy</button>
      <button id="key-modal-done" class="primary">I've saved it</button>
    </div>
  </div>
</div>

<!-- API key prompt -->
<div id="token-modal" class="modal">
  <div class="box">
    <h3>Admin API key</h3>
    <p>Paste your <code>wk_&hellip;</code> admin key to manage keys. In OPEN
    mode any value works.</p>
    <input id="token-input" type="password" placeholder="wk_&hellip;">
    <div class="actions">
      <button id="token-cancel">Cancel</button>
      <button id="token-save" class="primary">Save</button>
    </div>
    <p id="token-err" class="err"></p>
  </div>
</div>

<div id="toast"></div>

{{SCALE_PICKER_JS}}
{{SEV_POLLER_JS}}
<script>
(function() {
  'use strict';

  var TOKEN_KEY = 'whisper_api_key';
  function getToken() {
    try { return sessionStorage.getItem(TOKEN_KEY) || ''; } catch(_) { return ''; }
  }
  function setToken(v) {
    try { sessionStorage.setItem(TOKEN_KEY, v || ''); } catch(_) {}
  }

  function authHeaders() {
    var t = getToken();
    return t ? { Authorization: 'Bearer ' + t } : {};
  }

  async function api(method, path, body) {
    var h = Object.assign({ Accept: 'application/json' }, authHeaders());
    var opts = { method: method, headers: h };
    if (body !== undefined) {
      h['Content-Type'] = 'application/json';
      opts.body = JSON.stringify(body);
    }
    return fetch(path, opts);
  }

  function showToast(msg, kind) {
    var el = document.getElementById('toast');
    if (!el) {
      el = document.createElement('div');
      el.id = 'toast';
      el.style.position = 'fixed';
      el.style.bottom = '1rem';
      el.style.right = '1rem';
      el.style.padding = '0.6rem 1rem';
      el.style.borderRadius = '4px';
      el.style.zIndex = '200';
      el.style.fontSize = 'var(--fs-sm)';
      document.body.appendChild(el);
    }
    el.textContent = msg;
    el.style.background = kind === 'err' ? '#5a2424'
                       : kind === 'ok'  ? '#1d4f2c'
                       : '#21262d';
    el.style.color = '#fff';
    el.style.display = 'block';
    setTimeout(function(){ el.style.display = 'none'; }, 3000);
  }

  function fmtTs(ts) {
    if (!ts) return '—';
    var d = new Date(ts * 1000);
    return d.toISOString().replace('T', ' ').slice(0, 19) + 'Z';
  }

  function showTokenModal() {
    return new Promise(function(resolve){
      var m = document.getElementById('token-modal');
      var inp = document.getElementById('token-input');
      var err = document.getElementById('token-err');
      err.textContent = '';
      inp.value = '';
      m.classList.add('show');
      setTimeout(function(){ inp.focus(); }, 50);
      function done(v) {
        m.classList.remove('show');
        document.getElementById('token-save').onclick = null;
        document.getElementById('token-cancel').onclick = null;
        inp.onkeydown = null;
        resolve(v);
      }
      document.getElementById('token-save').onclick = function() {
        var v = inp.value.trim();
        if (!v) { err.textContent = 'Empty value'; return; }
        done(v);
      };
      document.getElementById('token-cancel').onclick = function() {
        done(null);
      };
      inp.onkeydown = function(e) {
        if (e.key === 'Enter') document.getElementById('token-save').click();
        if (e.key === 'Escape') document.getElementById('token-cancel').click();
      };
    });
  }

  function showKeyModal(rawKey, username) {
    document.getElementById('key-modal-user').textContent = username;
    document.getElementById('key-modal-raw').textContent = rawKey;
    var m = document.getElementById('key-modal');
    m.classList.add('show');
    document.getElementById('key-modal-copy').onclick = function() {
      // navigator.clipboard requires a secure context (https / localhost).
      // Over LAN HTTP it's undefined, so fall back to a hidden textarea +
      // document.execCommand('copy'). If both fail, select the visible
      // .raw-key span so the user can ctrl-c manually.
      function copyFallback(text) {
        var ta = document.createElement('textarea');
        ta.value = text;
        // Hide off-screen but keep selectable.
        ta.style.position = 'fixed';
        ta.style.top = '-1000px';
        ta.setAttribute('readonly', '');
        document.body.appendChild(ta);
        ta.select();
        ta.setSelectionRange(0, ta.value.length);
        var ok = false;
        try { ok = document.execCommand('copy'); } catch(_) {}
        document.body.removeChild(ta);
        return ok;
      }
      function selectRawSpan() {
        var span = document.getElementById('key-modal-raw');
        var sel = window.getSelection();
        var range = document.createRange();
        range.selectNodeContents(span);
        sel.removeAllRanges();
        sel.addRange(range);
      }
      function onSuccess() { showToast('Copied to clipboard', 'ok'); }
      function onFailure() {
        selectRawSpan();
        showToast('Auto-copy blocked — press Ctrl+C', 'err');
      }
      if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(rawKey).then(onSuccess, function() {
          if (!copyFallback(rawKey)) onFailure();
          else onSuccess();
        });
      } else {
        if (copyFallback(rawKey)) onSuccess();
        else onFailure();
      }
    };
    document.getElementById('key-modal-done').onclick = function() {
      m.classList.remove('show');
      load();
    };
  }

  function renderUser(u) {
    var card = document.createElement('div');
    card.className = 'card';
    var h = document.createElement('h3');
    h.innerHTML = '<span>' + escapeHtml(u.username) + '</span>';
    var pill = document.createElement('span');
    pill.className = 'pill ' + (u.is_admin ? 'admin' : '');
    pill.textContent = u.is_admin ? 'admin' : 'user';
    h.appendChild(pill);
    if (u.revoked_ts) {
      var rp = document.createElement('span');
      rp.className = 'pill revoked';
      rp.textContent = 'revoked';
      h.appendChild(rp);
    }
    var keyCount = document.createElement('span');
    keyCount.className = 'pill';
    keyCount.textContent = u.active_key_count + ' active key' +
      (u.active_key_count === 1 ? '' : 's');
    h.appendChild(keyCount);
    card.appendChild(h);

    var tb = document.createElement('div');
    tb.className = 'toolbar';
    if (!u.revoked_ts) {
      var labelInp = document.createElement('input');
      labelInp.type = 'text';
      labelInp.placeholder = 'label (e.g., desktop)';
      labelInp.style.maxWidth = '14rem';
      labelInp.style.flex = '1';
      var addBtn = document.createElement('button');
      addBtn.className = 'primary';
      addBtn.textContent = '+ generate key';
      addBtn.onclick = function() {
        var label = labelInp.value.trim();
        api('POST', '/config/api-keys/api/users/' + encodeURIComponent(u.id) + '/keys',
            { label: label })
          .then(function(r) {
            if (!r.ok) return r.text().then(function(t) { throw new Error(t); });
            return r.json();
          })
          .then(function(j) {
            showKeyModal(j.key, u.username);
          })
          .catch(function(e) {
            showToast(String(e.message || e), 'err');
          });
      };
      tb.appendChild(labelInp);
      tb.appendChild(addBtn);

      var revBtn = document.createElement('button');
      revBtn.className = 'danger';
      revBtn.textContent = 'revoke user';
      revBtn.onclick = function() {
        if (!confirm('Revoke user "' + u.username +
            '"? This will also revoke all of their keys.')) return;
        api('DELETE', '/config/api-keys/api/users/' + encodeURIComponent(u.id))
          .then(function(r) {
            if (r.status === 409) {
              return r.json().then(function(j) {
                throw new Error(j.detail || 'cannot revoke last admin');
              });
            }
            if (!r.ok) throw new Error('HTTP ' + r.status);
            showToast('User revoked', 'ok');
            load();
          })
          .catch(function(e) {
            showToast(String(e.message || e), 'err');
          });
      };
      tb.appendChild(revBtn);
    }
    card.appendChild(tb);

    // Fetch + render keys
    var listEl = document.createElement('div');
    card.appendChild(listEl);
    api('GET', '/config/api-keys/api/users/' + encodeURIComponent(u.id) + '/keys')
      .then(function(r) { return r.ok ? r.json() : { keys: [] }; })
      .then(function(j) {
        if (!j.keys || j.keys.length === 0) {
          listEl.innerHTML = '<p class="hint">No keys yet — generate one above.</p>';
          return;
        }
        j.keys.forEach(function(k) {
          var row = document.createElement('div');
          row.className = 'key-row';
          row.innerHTML =
            '<div class="label">' + escapeHtml(k.label || '(no label)') + '</div>' +
            '<div class="id">' + escapeHtml(k.key_prefix) + '&hellip;' + escapeHtml(k.key_last4) + '</div>' +
            '<div class="ts">created ' + escapeHtml(fmtTs(k.created_ts)) + '</div>' +
            '<div class="ts">used '    + escapeHtml(fmtTs(k.last_used_ts)) + '</div>';
          var actionCell = document.createElement('div');
          if (k.revoked_ts) {
            var rp = document.createElement('span');
            rp.className = 'pill revoked';
            rp.textContent = 'revoked';
            actionCell.appendChild(rp);
          } else if (!u.revoked_ts) {
            var b = document.createElement('button');
            b.className = 'danger';
            b.textContent = 'revoke';
            b.onclick = function() {
              if (!confirm('Revoke key ' + k.key_prefix + '…' + k.key_last4 + '?')) return;
              api('DELETE', '/config/api-keys/api/users/' + encodeURIComponent(u.id) +
                  '/keys/' + encodeURIComponent(k.id))
                .then(function(r) {
                  if (r.status === 409) {
                    return r.json().then(function(j) {
                      throw new Error(j.detail || 'cannot revoke last admin key');
                    });
                  }
                  if (!r.ok) throw new Error('HTTP ' + r.status);
                  showToast('Key revoked', 'ok');
                  load();
                })
                .catch(function(e) {
                  showToast(String(e.message || e), 'err');
                });
            };
            actionCell.appendChild(b);
          }
          row.appendChild(actionCell);
          listEl.appendChild(row);
        });
      });

    return card;
  }

  function escapeHtml(s) {
    return String(s == null ? '' : s)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;')
      .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }

  async function load() {
    var r = await api('GET', '/config/api-keys/api/users');
    if (r.status === 401) {
      var v = await showTokenModal();
      if (!v) return;
      setToken(v);
      r = await api('GET', '/config/api-keys/api/users');
      if (!r.ok) {
        setToken('');
        document.getElementById('token-err').textContent = 'invalid key';
        return;
      }
    }
    if (!r.ok) {
      showToast('Load failed: HTTP ' + r.status, 'err');
      return;
    }
    var j = await r.json();
    var banner = document.getElementById('open-banner');
    banner.style.display = j.open_mode ? 'block' : 'none';
    var ct = document.getElementById('users-container');
    ct.innerHTML = '';
    if (!j.users.length) {
      ct.innerHTML = '<p class="hint">No users yet. Add one above to create the first admin.</p>';
      return;
    }
    j.users.forEach(function(u) { ct.appendChild(renderUser(u)); });
  }

  document.getElementById('add-user-btn').onclick = function() {
    var username = document.getElementById('new-username').value.trim();
    var is_admin = document.getElementById('new-is-admin').checked;
    if (!username) { showToast('Enter a username', 'err'); return; }
    api('POST', '/config/api-keys/api/users', { username: username, is_admin: is_admin })
      .then(function(r) {
        if (!r.ok) return r.text().then(function(t) { throw new Error(t); });
        return r.json();
      })
      .then(function() {
        document.getElementById('new-username').value = '';
        document.getElementById('new-is-admin').checked = false;
        showToast('User created', 'ok');
        load();
      })
      .catch(function(e) { showToast(String(e.message || e), 'err'); });
  };

  document.getElementById('reload-btn').onclick = load;
  document.getElementById('logout-btn').onclick = function() {
    setToken('');
    location.reload();
  };

  load();
})();
</script>
</body></html>
"""
