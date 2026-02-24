/* app.js — Frontend glue for Premium Editorial Bookstore
   - Correct API endpoints wiring for search, popular, related, book detail
   - Proper rating display using backend rating and backend reviewCount when present
   - Related books request fixed to /api/books/{id}/related
   - Views increment one-per-session via POST /api/books/{id}/view (graceful fallback)
   - Local personal rating + local reviews persisted in localStorage
   - Modal fits (scrollable), initial personal rating is empty (0) when not rated
*/

const API_ROOT = '/api';
const PLACEHOLDER = 'https://picsum.photos/seed/placeholder/600/900';
const PER_PAGE = 9;

/* DOM refs */
const searchInput = document.getElementById('searchInput');
const suggestionsEl = document.getElementById('suggestions');
const searchResults = document.getElementById('searchResults');
const featuredCover = document.getElementById('featuredCover');
const featuredTitle = document.getElementById('featuredTitle');
const featuredDesc = document.getElementById('featuredDesc');
const featuredMeta = document.getElementById('featuredMeta');
const popularList = document.getElementById('popularList');
const userRecList = document.getElementById('userRecList');
const paginationEl = document.getElementById('pagination');
const sortSelect = document.getElementById('sortSelect');
const clearFiltersBtn = document.getElementById('clearFilters');
const modal = document.getElementById('modal');
const modalBackdrop = document.getElementById('modalBackdrop');
const modalContent = document.getElementById('modalContent');
const cartDrawer = document.getElementById('cartDrawer');
const wishDrawer = document.getElementById('wishDrawer');
const openCartBtn = document.getElementById('open-cart');
const openWishBtn = document.getElementById('open-wishlist');
const closeCart = document.getElementById('closeCart');
const closeWish = document.getElementById('closeWish');
const cartItemsEl = document.getElementById('cartItems');
const wishItemsEl = document.getElementById('wishItems');
const cartTotalEl = document.getElementById('cartTotal');
const checkoutBtn = document.getElementById('checkoutBtn');
const cartCount = document.getElementById('cartCount');
const wishCount = document.getElementById('wishCount');

let STATE = {
  books: [],
  featured: null,
  page: 1,
  perPage: PER_PAGE,
  query: '',
  sort: 'featured',
  filters: { author: null, category: null },
  searchResults: null,
  isSearching: false,
  cart: JSON.parse(localStorage.getItem('cart') || '[]'),
  wishlist: JSON.parse(localStorage.getItem('wishlist') || '[]')
};

window._lastResponses = {}; // debug

/* ---------------- fetchVerbose ----------------
   wrapper that records responses for debugging and returns parsed JSON when available
*/
async function fetchVerbose(path, key, opts = {}) {
  const url = path.startsWith('http') ? path : `${API_ROOT}${path}`;
  const info = { url, status: null, ok: false, text: null, json: null, opts };
  window._lastResponses[key || url] = info;
  try {
    const headers = new Headers(opts.headers || {});
    if (!headers.has('Content-Type') && opts.body && !(opts.body instanceof FormData)) {
      headers.set('Content-Type', 'application/json');
    }
    const res = await fetch(url, { ...opts, headers });
    info.status = res.status;
    info.ok = res.ok;
    const txt = await res.text();
    info.text = txt;
    try { info.json = JSON.parse(txt); } catch (e) { info.json = null; }
    window._lastResponses[key || url] = info;
    if (!res.ok) throw new Error(`HTTP ${res.status}: ${txt.slice(0,200)}`);
    return info.json !== null ? info.json : txt;
  } catch (err) {
    window._lastResponses[key || url] = info;
    console.warn('fetchVerbose error', url, err);
    throw err;
  }
}

/* ---------------- normalizeBook ----------------
   Backend returns many shapes; normalize to { id, title, image, description, authors[], categories[], rating, reviewCount, publisher, year, views }
*/
function normalizeBook(raw) {
  if (!raw) return null;
  if (Array.isArray(raw) && raw.length > 0) raw = raw[0];

  // Remove wrapping quotes/brackets from noisy CSV-like values, e.g. "'History'" or "['History']"
  const cleanLabel = (value) => {
    let s = String(value === undefined || value === null ? '' : value).trim();
    if (!s) return '';
    s = s.replace(/^\[+|\]+$/g, '').trim();
    s = s.replace(/^[`"'‘’“”\s]+|[`"'‘’“”\s]+$/g, '').trim();
    return s;
  };

  const toCleanArray = (value) => {
    let parts = [];
    if (Array.isArray(value)) parts = value;
    else if (typeof value === 'string') {
      const s = value.trim();
      if (!s) parts = [];
      else {
        const unwrapped = (s.startsWith('[') && s.endsWith(']')) ? s.slice(1, -1) : s;
        parts = unwrapped.split(/[,;|]+/);
      }
    } else if (value !== undefined && value !== null) {
      parts = [value];
    }

    const out = [];
    const seen = new Set();
    parts.forEach((p) => {
      const cleaned = cleanLabel(p);
      const key = cleaned.toLowerCase();
      if (!cleaned || seen.has(key)) return;
      seen.add(key);
      out.push(cleaned);
    });
    return out;
  };

  const get = (o, ...keys) => {
    for (const k of keys) {
      if (!o) continue;
      if (k in o && o[k] !== undefined && o[k] !== null) return o[k];
      const camel = k.replace(/_([a-z])/g, g => g[1].toUpperCase());
      if (camel in o && o[camel] !== undefined) return o[camel];
      const hyphen = k.replaceAll('_', '-');
      if (hyphen in o && o[hyphen] !== undefined) return o[hyphen];
    }
    return undefined;
  };

  const id = String(get(raw, 'ISBN', 'isbn', 'id') || '');
  const title = get(raw, 'Book_Title', 'Book-Title', 'title') || '';
  const image = get(raw, 'Image', 'image', 'cover') || PLACEHOLDER;
  const description = get(raw, 'Description', 'description', 'page_content', 'desc') || '';
  const rating = Number(get(raw, 'rating', 'score') || 0);
  const publisher = get(raw, 'Publisher', 'publisher') || '';
  const year = get(raw, 'Year_Of_Publication', 'Year-Of-Publication', 'year') || '';

  // authors to array
  let aRaw = get(raw, 'Book_Author', 'Book-Author', 'authors', 'author') || [];
  let authors = [];
  if (Array.isArray(aRaw)) authors = toCleanArray(aRaw);
  else if (typeof aRaw === 'string') authors = toCleanArray(aRaw);
  else authors = toCleanArray(aRaw);

  // categories to array
  let cRaw = get(raw, 'Categories', 'categories', 'tags') || [];
  let categories = [];
  if (Array.isArray(cRaw)) categories = toCleanArray(cRaw);
  else if (typeof cRaw === 'string') categories = toCleanArray(cRaw);
  else categories = toCleanArray(cRaw);

  // backend-provided review count or variations
  let reviewCount = get(raw, 'reviewCount', 'reviews_count', 'review_count', 'num_reviews', 'reviewsCount', 'reviews') || 0;
  if (Array.isArray(reviewCount)) reviewCount = reviewCount.length;
  reviewCount = Number(reviewCount || 0);

  // backend views (if server provides)
  let views = Number(get(raw, 'views', 'view_count', 'viewsCount') || 0);

  return { id, title, authors, categories, image, description, rating, reviewCount, publisher, year, views };
}

function dedupeBooks(list) {
  const out = [];
  const seen = new Set();
  (list || []).forEach((item) => {
    const b = normalizeBook(item);
    if (!b) return;
    const id = String(b.id || '').trim();
    const fallback = [
      String(b.title || '').trim().toLowerCase(),
      String((b.authors || []).join('|') || '').trim().toLowerCase(),
      String(b.year || '').trim().toLowerCase(),
      String(b.publisher || '').trim().toLowerCase()
    ].join('::');
    const key = id ? `id:${id}` : `meta:${fallback}`;
    if (seen.has(key)) return;
    seen.add(key);
    out.push(b);
  });
  return out;
}

/* ---------------- local storage helpers ---------------- */
const PERSONAL_RATINGS_KEY = 'personal_ratings_v2';
const LOCAL_REVIEWS_KEY = 'local_reviews_v2';
const VIEWS_KEY = 'book_views_v1';

function readPersonalRatings() { try { return JSON.parse(localStorage.getItem(PERSONAL_RATINGS_KEY) || '{}'); } catch { return {}; } }
function writePersonalRatings(obj) { localStorage.setItem(PERSONAL_RATINGS_KEY, JSON.stringify(obj)); }
function getPersonalRating(isbn) { return Number(readPersonalRatings()[isbn] || 0); }
function setPersonalRating(isbn, val) { const r = readPersonalRatings(); r[isbn] = Number(val); writePersonalRatings(r); }

function readLocalReviews() { try { return JSON.parse(localStorage.getItem(LOCAL_REVIEWS_KEY) || '{}'); } catch { return {}; } }
function writeLocalReviews(obj) { localStorage.setItem(LOCAL_REVIEWS_KEY, JSON.stringify(obj)); }
function getLocalReviews(isbn) { const all = readLocalReviews(); return all[isbn] || []; }
function addLocalReview(isbn, review) { const all = readLocalReviews(); all[isbn] = all[isbn] || []; all[isbn].unshift(review); writeLocalReviews(all); }

function readViews() { try { return JSON.parse(localStorage.getItem(VIEWS_KEY) || '{}'); } catch { return {}; } }
function writeViews(obj) { localStorage.setItem(VIEWS_KEY, JSON.stringify(obj)); }
function getViewsLocal(isbn) { const v = readViews(); return Number(v[isbn] || 0); }

/* session-level view guard (one increment per session per book) */
function hasViewedInSession(isbn) {
  try { return sessionStorage.getItem(`viewed_${isbn}`) === '1'; } catch { return false; }
}
function markViewedInSession(isbn) {
  try { sessionStorage.setItem(`viewed_${isbn}`, '1'); } catch { }
}

/* increment view once-per-session (server if endpoint exists, else localStorage) */
async function incViewOnce(isbn) {
  if (!isbn) return 0;
  if (hasViewedInSession(isbn)) return getViewsDisplay(isbn);
  markViewedInSession(isbn);

  // try server endpoint
  try {
    await fetchVerbose(`/books/${encodeURIComponent(isbn)}/view`, `view_${isbn}`, { method: 'POST', body: JSON.stringify({ view: 1 }) });
  } catch (e) {
    // ignore if server doesn't have this endpoint
  }

  // also track locally for UI (persist)
  const v = readViews();
  v[isbn] = (Number(v[isbn] || 0) + 1);
  writeViews(v);

  // animate UI count if present
  const el = document.querySelector(`.book-card[data-isbn="${cssEscape(isbn)}"] .views-count`);
  if (el) {
    el.textContent = v[isbn];
    el.classList.remove('animate');
    // reflow to restart animation
    void el.offsetWidth;
    el.classList.add('animate');
  }

  // if modal open and showing views, update it
  const modalViewEl = modalContent.querySelector('.views-inline .views-count');
  if (modalViewEl && modal.getAttribute('aria-hidden') === 'false') {
    modalViewEl.textContent = v[isbn];
  }

  return v[isbn];
}

function getViewsDisplay(isbn) {
  // prefer server-provided (normalized) view value (STATE.books), else local
  const book = STATE.books.find(b => b.id === isbn);
  if (book && Number(book.views) > 0) return Number(book.views);
  return getViewsLocal(isbn);
}

/* utility: css escape for data-isbn selectors */
function cssEscape(s) {
  return String(s).replace(/([ #;?%&,.+*~\':"!^$[\]()=>|\\/])/g, '\\$1');
}

/* ---------------- UI fragments ---------------- */
function escapeHtml(s) { return String(s === undefined || s === null ? '' : s).replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;'); }

function renderPersonalStarButtons(isbn) {
  const cur = getPersonalRating(isbn) || 0; // 0 when not rated -> empty stars
  let html = '';
  for (let i = 1; i <= 5; i++) {
    html += `<button class="star-btn ${i <= cur ? 'filled' : ''}" data-value="${i}" data-isbn="${escapeHtml(isbn)}" aria-label="Rate ${i}">★</button>`;
  }
  return html;
}

function formatAvgDisplay(avg, reviewCount) {
  const a = Number(avg) || 0;
  const one = Math.round(a * 10) / 10;
  return `<div class="rating-block"><div class="avg-number">${one.toFixed(1)} <span class="emoji-star">⭐</span></div><div class="muted small">(${reviewCount} reviews)</div></div>`;
}

// inline svg eye
function eyeSvg() {
  return `<svg class="eye-icon" width="18" height="18" viewBox="0 0 24 24" preserveAspectRatio="xMidYMid meet" xmlns="http://www.w3.org/2000/svg" aria-hidden="true" style="width:18px;height:18px;min-width:18px;min-height:18px;max-width:18px;max-height:18px;display:inline-block;vertical-align:middle;flex:0 0 18px;fill:currentColor"><path d="M12 5C7 5 2.73 8.11 1 12c1.73 3.89 6 7 11 7s9.27-3.11 11-7c-1.73-3.89-6-7-11-7zm0 12a5 5 0 1 1 0-10 5 5 0 0 1 0 10z"/><circle cx="12" cy="12" r="2.5"/></svg>`;
}

/* ---------- computeCombinedAverage ----------
   - If backend provides reviewCount (>0) and rating, treat backend rating as authoritative (optionally combine with local reviews by weighting)
   - If backend reviewCount is 0, prefer backend rating if present; otherwise local reviews average.
*/
function computeCombinedAverage(backendRating, backendCount, localReviews) {
  const backendR = Number(backendRating || 0);
  const backendC = Number(backendCount || 0);
  const localCount = (localReviews || []).length;
  const localSum = (localReviews || []).reduce((s, r) => s + (Number(r.rating || 0)), 0);
  const localAvg = localCount ? (localSum / localCount) : 0;

  if (backendC > 0 && backendR > 0) {
    // weighted average: weight backend by its count, local by its count
    const totalCount = backendC + localCount;
    const totalSum = backendR * backendC + (localAvg * localCount);
    return totalCount ? (totalSum / totalCount) : backendR;
  }
  // no backend counts — prefer backend rating if available
  if (backendR > 0) return backendR;
  return localCount ? localAvg : 0;
}

/* ------------------- Render grid / cards ------------------- */
function renderGrid(items) {
  searchResults.innerHTML = '';
  if (!items || items.length === 0) {
    searchResults.innerHTML = `<div class="muted">No results</div>`;
    return;
  }

  items.forEach((raw, idx) => {
    const b = normalizeBook(raw);
    const localReviews = getLocalReviews(b.id);
    const avg = computeCombinedAverage(b.rating, b.reviewCount, localReviews);
    const views = getViewsDisplay(b.id);

    const authorsHtml = (b.authors || []).map(a => `<span class="tag tag-author" data-author="${escapeHtml(a)}">${escapeHtml(a)}</span>`).join(' ');
    const categoriesHtml = (b.categories || []).map(c => `<span class="tag tag-category" data-category="${escapeHtml(c)}">${escapeHtml(c)}</span>`).join(' ');

    const el = document.createElement('div');
    el.className = 'book-card';
    el.dataset.isbn = b.id || '';
    el.innerHTML = `
      <div class="cover" style="background-image:url('${escapeHtml(b.image || PLACEHOLDER)}')"></div>
      <div class="card-body">
        <div class="book-title">${escapeHtml(b.title)}</div>
        <div class="publisher">${escapeHtml(b.publisher || '')}</div>
        <div class="excerpt">${escapeHtml((b.description || '').slice(0, 140))}</div>
        <div class="tags">${authorsHtml}</div>
        <div class="tags">${categoriesHtml}</div>

        <div style="display:flex;align-items:center;justify-content:space-between;margin-top:8px">
          <div>
            ${formatAvgDisplay(avg, Math.max(b.reviewCount || 0, localReviews.length))}
            <div class="views-inline">${eyeSvg()} <span class="views-count muted small">${views}</span></div>
          </div>

          <div style="min-width:120px;text-align:right" class="small muted"></div>
        </div>
      </div>
    `;

    // Clicking cover/title -> increment view once-per-session then open modal
    const openHandler = async (e) => {
      e && e.stopPropagation && e.stopPropagation();
      try { await incViewOnce(b.id); } catch (err) { /* ignore */ }
      await openBook(b.id);
      // re-render grid so cards show updated view counts & weighted ratings
      applySearchAndRender();
    };

    el.querySelector('.cover')?.addEventListener('click', openHandler);
    el.querySelector('.book-title')?.addEventListener('click', openHandler);

    // tag click filters
    el.querySelectorAll('.tag-author').forEach(t => t.addEventListener('click', (e) => {
      e.stopPropagation();
      STATE.filters.author = t.dataset.author;
      STATE.page = 1;
      applySearchAndRender();
    }));
    el.querySelectorAll('.tag-category').forEach(t => t.addEventListener('click', (e) => {
      e.stopPropagation();
      STATE.filters.category = t.dataset.category;
      STATE.page = 1;
      applySearchAndRender();
    }));

    searchResults.appendChild(el);
    setTimeout(() => el.classList.add('visible'), idx * 40);
  });

  observeReveal();
}

/* ---------------- pagination ---------------- */
function renderPagination(pages) {
  paginationEl.innerHTML = '';
  if (!pages || pages <= 1) return;
  for (let i = 1; i <= pages; i++) {
    const btn = document.createElement('button');
    btn.className = 'btn-ghost page-btn';
    btn.textContent = i;
    if (i === STATE.page) btn.disabled = true;
    btn.addEventListener('click', () => { STATE.page = i; applySearchAndRender(); });
    paginationEl.appendChild(btn);
  }
}

/* ---------------- debounced search ---------------- */
function debounce(fn, wait = 260) { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), wait); }; }

const doSearch = debounce(async (q) => {
  q = (q || '').trim();
  STATE.query = q;
  if (!q) { STATE.isSearching = false; STATE.searchResults = null; applySearchAndRender(); suggestionsEl.style.display = 'none'; return; }
  STATE.isSearching = true;
  try {
    const res = await fetchVerbose(`/search?q=${encodeURIComponent(q)}`, 'search');
    const arr = Array.isArray(res) ? res : (res && (res.results || res.items || res.data) ? (res.results || res.items || res.data) : []);
    STATE.searchResults = dedupeBooks(arr || []);
    renderSuggestions(STATE.searchResults.slice(0, 6));
    STATE.page = 1;
    applySearchAndRender();
  } catch (e) {
    console.warn('search failed', e);
    STATE.searchResults = [];
    renderSuggestions([]);
    applySearchAndRender();
  }
}, 180);

searchInput?.addEventListener('input', (e) => doSearch(e.target.value));
searchInput?.addEventListener('focus', () => { if (suggestionsEl && suggestionsEl.children.length) suggestionsEl.style.display = 'block'; });
searchInput?.addEventListener('blur', () => setTimeout(() => suggestionsEl && (suggestionsEl.style.display = 'none'), 150));

function renderSuggestions(list) {
  if (!suggestionsEl) return;
  suggestionsEl.innerHTML = '';
  if (!list || list.length === 0) { suggestionsEl.style.display = 'none'; return; }
  list.forEach(b => {
    const s = document.createElement('div');
    s.className = 'sugg';
    s.textContent = b.title || b.Book_Title || '';
    s.addEventListener('click', () => {
      searchInput.value = b.title || b.Book_Title || '';
      suggestionsEl.style.display = 'none';
      STATE.isSearching = true;
      STATE.searchResults = [b];
      STATE.page = 1;
      applySearchAndRender();
    });
    suggestionsEl.appendChild(s);
  });
  suggestionsEl.style.display = 'block';
}

/* ---------------- filters / sort / apply ---------------- */
sortSelect?.addEventListener('change', () => { STATE.sort = sortSelect.value; STATE.page = 1; applySearchAndRender(); });
clearFiltersBtn?.addEventListener('click', () => {
  STATE.filters = { author: null, category: null };
  STATE.query = '';
  searchInput.value = '';
  STATE.isSearching = false;
  STATE.searchResults = null;
  applySearchAndRender();
});

function applySearchAndRender() {
  let arr = (STATE.isSearching && Array.isArray(STATE.searchResults)) ? [...STATE.searchResults] : [...STATE.books];
  arr = dedupeBooks(arr);

  if (STATE.filters.author) arr = arr.filter(b => (b.authors || []).includes(STATE.filters.author));
  if (STATE.filters.category) arr = arr.filter(b => (b.categories || []).includes(STATE.filters.category));

  if (STATE.sort === 'rating') arr.sort((a, b) => (Number(b.rating || 0) - Number(a.rating || 0)));
  else if (STATE.sort === 'newest') arr.sort((a, b) => (Number(b.year || 0) - Number(a.year || 0)));
  else if (STATE.sort === 'az') arr.sort((a, b) => (a.title || '').localeCompare(b.title || ''));
  else arr.sort((a, b) => (Number(b.rating || 0) - Number(a.rating || 0)));

  const total = arr.length;
  const start = (STATE.page - 1) * STATE.perPage;
  const pageItems = arr.slice(start, start + STATE.perPage);

  renderGrid(pageItems);
  renderPagination(Math.ceil(total / STATE.perPage));
  renderActiveFilters();
}

function renderActiveFilters() {
  const el = document.getElementById('activeFilters');
  const parts = [];
  if (STATE.filters.author) parts.push(`Author: ${STATE.filters.author}`);
  if (STATE.filters.category) parts.push(`Category: ${STATE.filters.category}`);
  el && (el.textContent = parts.join(' • '));
}

/* ---------------- modal / book detail / reviews ---------------- */
async function openBook(isbn) {
  if (!isbn) return;
  let b = STATE.books.find(x => x.id === isbn) || (STATE.searchResults || []).find(x => x.id === isbn);
  try {
    const r = await fetchVerbose(`/books/${encodeURIComponent(isbn)}`, 'get_book');
    b = normalizeBook(r);
  } catch (e) {
    // fallback to local if server fails
    if (!b) { alert('Book not found'); return; }
  }

  // fetch related
  let related = [];
  try {
    const rr = await fetchVerbose(`/books/${encodeURIComponent(isbn)}/related`, 'related');
    related = Array.isArray(rr) ? dedupeBooks(rr) : [];
  } catch (e) {
    // ignore if server doesn't return related
  }

  await incViewOnce(isbn); // increment view only once per session
  showBookModal(b, related);
  // update grid UI to reflect new views
  applySearchAndRender();
}

function showBookModal(book, related = []) {
  if (!book) return;
  const localReviews = getLocalReviews(book.id);
  const avg = computeCombinedAverage(book.rating, book.reviewCount, localReviews);
  const views = getViewsDisplay(book.id);

  modal.classList.remove('hidden');
  modal.setAttribute('aria-hidden', 'false');

  modalContent.innerHTML = `
    <div style="display:flex;gap:18px" class="modal-grid">
      <div>
        <div class="detail-cover" style="background-image:url('${escapeHtml(book.image || PLACEHOLDER)}')"></div>
      </div>
      <div class="detail-body">
        <h2>${escapeHtml(book.title)}</h2>
        <div class="muted">${escapeHtml((book.authors || []).join(', '))} · ${escapeHtml(book.year || '')}</div>
        <div style="margin-top:8px">
          <strong style="font-size:18px">${avg.toFixed(1)} ⭐</strong>
          <span class="views-inline muted" style="margin-left:8px">${eyeSvg()} <span class="views-count">${views}</span> views</span>
          <span class="muted" style="margin-left:10px">(${Math.max(book.reviewCount || 0, localReviews.length)} reviews)</span>
        </div>

        <p style="margin-top:10px;color:#dbeafe">${escapeHtml(book.description || 'No description available.')}</p>
        <p style="margin-top:8px"><strong>Publisher:</strong> ${escapeHtml(book.publisher || '')}</p>

        <div style="margin-top:14px">
          <h4>Leave a review</h4>
          <div id="modalPersonalStars" style="display:flex;gap:8px;align-items:center">
            ${renderPersonalStarButtons(book.id)}
          </div>
          <textarea id="reviewText" placeholder="Write a short review (max 400 chars)" style="width:100%;height:86px;margin-top:8px;padding:8px;border-radius:8px;background:#07101a;border:none;color:var(--text)"></textarea>
          <div style="display:flex;gap:8px;justify-content:flex-end;margin-top:8px">
            <button id="submitReview" class="btn-primary">Submit Review</button>
            <button id="closeModalBtn" class="btn-ghost">Close</button>
          </div>
        </div>

        <div style="margin-top:12px">
          <h4>Reviews</h4>
          <div class="reviews" id="reviewsList">
            ${localReviews.map(rv => `<div class="review"><div class="meta">${escapeHtml(rv.name || 'You')} · <span class="muted">${escapeHtml(rv.date)}</span></div><div class="text">${escapeHtml(rv.text)}</div></div>`).join('')}
          </div>
        </div>
      </div>
    </div>

    <div style="margin-top:12px">
      <h4>Related</h4>
      <div class="related-row">
        ${related.length ? related.map(r => `
          <button class="related-item" type="button" data-isbn="${escapeHtml(r.id || '')}" aria-label="Open related book ${escapeHtml(r.title || '')}">
            <div class="related-cover" style="background-image:url('${escapeHtml(r.image || PLACEHOLDER)}')"></div>
            <div class="related-title">${escapeHtml(r.title)}</div>
          </button>
        `).join('') : `<div class="muted small">No related books found.</div>`}
      </div>
    </div>
  `;

  // close handlers
  const closeModalBtn = document.getElementById('closeModalBtn');
  closeModalBtn && closeModalBtn.addEventListener('click', closeModal);
  modalBackdrop && modalBackdrop.addEventListener('click', closeModal);

  // personal stars in modal: initial state is empty if user hasn't rated
  const modalStars = modalContent.querySelectorAll('#modalPersonalStars .star-btn');
  modalStars.forEach(btn => {
    btn.addEventListener('click', async (e) => {
      e.stopPropagation();
      const val = Number(btn.dataset.value);
      const isbn = btn.dataset.isbn;
      // optimistic UI
      setPersonalRating(isbn, val);
      modalContent.querySelectorAll(`#modalPersonalStars .star-btn`).forEach(s => s.classList.toggle('filled', Number(s.dataset.value) <= val));
      // POST to backend if endpoint exists
      try {
        await fetchVerbose(`/books/${encodeURIComponent(isbn)}/rate`, `rate_${isbn}`, { method: 'POST', body: JSON.stringify({ rating: val }) });
      } catch (err) { /* ignore */ }
      applySearchAndRender(); // refresh cards to reflect possible new rating
    });
  });

  // submit review
  const submitReviewEl = document.getElementById('submitReview');
  submitReviewEl && submitReviewEl.addEventListener('click', async () => {
    const text = (document.getElementById('reviewText') || {}).value || '';
    const personal = getPersonalRating(book.id) || 0;
    if (!text.trim()) { alert('Please write a review'); return; }
    const review = { text: text.trim(), rating: personal, name: 'You', date: (new Date()).toLocaleString() };
    addLocalReview(book.id, review);
    // attempt server POST
    try {
      await fetchVerbose(`/books/${encodeURIComponent(book.id)}/review`, `review_${book.id}`, { method: 'POST', body: JSON.stringify({ text: review.text, rating: review.rating }) });
    } catch (err) { /* ignore */ }
    // add to reviews list
    const reviewsList = document.getElementById('reviewsList');
    reviewsList && reviewsList.insertAdjacentHTML('afterbegin', `<div class="review"><div class="meta">${escapeHtml(review.name)} · <span class="muted">${escapeHtml(review.date)}</span></div><div class="text">${escapeHtml(review.text)}</div></div>`);
    // clear textarea
    (document.getElementById('reviewText') || {}).value = '';
    applySearchAndRender();
  });

  // related click handlers
  modalContent.querySelectorAll('.related-item').forEach(el => el.addEventListener('click', async () => {
    const isbn = el.dataset.isbn;
    closeModal();
    await incViewOnce(isbn);
    applySearchAndRender();
    openBook(isbn);
  }));
}

function closeModal() {
  modal.classList.add('hidden');
  modal.setAttribute('aria-hidden', 'true');
  modalContent.innerHTML = '';
}

/* ---------------- cart & wishlist (unchanged) ---------------- */
function saveCart() { localStorage.setItem('cart', JSON.stringify(STATE.cart)); }
function saveWishlist() { localStorage.setItem('wishlist', JSON.stringify(STATE.wishlist)); }

function addToCart(book) {
  const id = book.id || book.title;
  const found = STATE.cart.find(i => i.id === id);
  if (found) found.qty = (found.qty || 1) + 1; else STATE.cart.push({ id, title: book.title, qty: 1, image: book.image });
  saveCart(); renderCartItems(); updateCounts();
}
function toggleWishlist(book) {
  const id = book.id || book.title;
  const idx = STATE.wishlist.findIndex(i => i.id === id);
  if (idx >= 0) STATE.wishlist.splice(idx, 1); else STATE.wishlist.push({ id, title: book.title, image: book.image });
  saveWishlist(); renderWishItems(); updateCounts();
}
function renderCartItems() {
  cartItemsEl.innerHTML = '';
  if (!STATE.cart.length) { cartItemsEl.innerHTML = '<div class="muted">Cart empty</div>'; cartTotalEl.textContent = 'Total: $0.00'; return; }
  let total = 0;
  STATE.cart.forEach(i => {
    total += (i.price || 0) * (i.qty || 1);
    const row = document.createElement('div');
    row.innerHTML = `<div style="display:flex;gap:12px;align-items:center"><img src="${escapeHtml(i.image || PLACEHOLDER)}" style="width:60px;height:80px;object-fit:cover;border-radius:6px"/><div>${escapeHtml(i.title)}<div class="muted">Qty: ${i.qty}</div></div></div><div><button class="btn-ghost btn-remove" data-id="${escapeHtml(i.id)}">Remove</button></div>`;
    cartItemsEl.appendChild(row);
  });
  cartTotalEl.textContent = `Total: $${total.toFixed(2)}`;
  cartItemsEl.querySelectorAll('.btn-remove').forEach(b => b.addEventListener('click', () => { STATE.cart = STATE.cart.filter(x => x.id !== b.dataset.id); saveCart(); renderCartItems(); updateCounts(); }));
}
function renderWishItems() {
  wishItemsEl.innerHTML = '';
  if (!STATE.wishlist.length) { wishItemsEl.innerHTML = '<div class="muted">No wishlist items</div>'; return; }
  STATE.wishlist.forEach(i => {
    const el = document.createElement('div');
    el.innerHTML = `<div style="display:flex;gap:12px;align-items:center"><img src="${escapeHtml(i.image || PLACEHOLDER)}" style="width:60px;height:80px;object-fit:cover;border-radius:6px"/><div>${escapeHtml(i.title)}</div></div><div><button class="btn-ghost btn-wish-remove" data-id="${escapeHtml(i.id)}">Remove</button></div>`;
    wishItemsEl.appendChild(el);
  });
  wishItemsEl.querySelectorAll('.btn-wish-remove').forEach(b => b.addEventListener('click', () => { STATE.wishlist = STATE.wishlist.filter(x => x.id !== b.dataset.id); saveWishlist(); renderWishItems(); updateCounts(); }));
}
openCartBtn && openCartBtn.addEventListener('click', () => { cartDrawer.classList.remove('hidden'); renderCartItems(); });
openWishBtn && openWishBtn.addEventListener('click', () => { wishDrawer.classList.remove('hidden'); renderWishItems(); });
closeCart && closeCart.addEventListener('click', () => cartDrawer.classList.add('hidden'));
closeWish && closeWish.addEventListener('click', () => wishDrawer.classList.add('hidden'));
checkoutBtn && checkoutBtn.addEventListener('click', () => alert('Checkout placeholder'));
function updateCounts() { cartCount && (cartCount.textContent = STATE.cart.reduce((s, i) => s + (i.qty || 0), 0)); wishCount && (wishCount.textContent = STATE.wishlist.length); }

/* ---------------- reveal observer ---------------- */
function observeReveal() {
  document.querySelectorAll('.book-card').forEach(el => {
    const obs = new IntersectionObserver(entries => { entries.forEach(en => { if (en.isIntersecting) el.classList.add('visible'); }); }, { threshold: 0.12 });
    obs.observe(el);
  });
}

/* ---------------- load home ---------------- */
async function loadHome() {
  try {
    // popular
    const popRaw = await fetchVerbose('/books/popular', 'popular');
    const popular = Array.isArray(popRaw) ? popRaw : (popRaw && (popRaw.results || popRaw.data || popRaw.items) ? (popRaw.results || popRaw.data || popRaw.items) : []);
    const normalized = dedupeBooks(popular || []);

    // store into STATE.books as a base dataset (server likely returns limited list; but we use it)
    STATE.books = normalized;
    STATE.featured = normalized.length ? normalized[0] : null;
    renderFeatured();
    renderCarousel(popularList, normalized.slice(0, 6));

    // recommendations
    try {
      const recRaw = await fetchVerbose('/user/recommendations', 'recs');
      const recs = Array.isArray(recRaw) ? dedupeBooks(recRaw) : [];
      renderCarousel(userRecList, recs);
    } catch (e) {
      // ignore recs error
    }

    // If user wants a broader catalog, your backend should expose a /api/books endpoint; we try to fetch it
    try {
      const all = await fetchVerbose('/books?per_page=500', 'books_all');
      const arr = Array.isArray(all) ? all : (all && (all.results || all.data || all.items) ? (all.results || all.data || all.items) : (all ? [all] : []));
      if (arr && arr.length) {
        STATE.books = dedupeBooks(arr || []);
        // keep featured if not provided by popular
        STATE.featured = STATE.featured || (STATE.books.length ? STATE.books[0] : null);
        renderFeatured();
      }
    } catch (e) {
      // optional endpoint - ignore if missing
    }

    applySearchAndRender();
  } catch (e) {
    console.warn('loadHome failed — falling back to demo data', e);
    STATE.books = dedupeBooks(demoBooks());
    STATE.featured = STATE.books[0] || null;
    renderFeatured();
    renderCarousel(popularList, STATE.books.slice(0, 6));
    applySearchAndRender();
  }
}

function renderFeatured() {
  const f = STATE.featured;
  if (!f) { featuredTitle.textContent = '—'; featuredDesc.textContent = ''; featuredCover.style.backgroundImage = ''; featuredMeta.innerHTML = ''; return; }
  featuredCover.style.backgroundImage = `url('${escapeHtml(f.image || PLACEHOLDER)}')`;
  featuredTitle.textContent = f.title;
  featuredDesc.textContent = (f.description || '').slice(0, 160);
  featuredMeta.innerHTML = `<span class="muted">${escapeHtml((f.authors || []).join(', '))}</span> · <span class="muted">${escapeHtml(f.publisher || '')}</span> · <strong>${(f.rating || 0).toFixed(1)}</strong>`;
  document.getElementById('featuredView').onclick = async () => { if (f.id) { await incViewOnce(f.id); applySearchAndRender(); openBook(f.id); } };
  document.getElementById('featuredAddCart').onclick = () => addToCart(f);
}

function renderCarousel(container, items) {
  if (!container) return;
  container.innerHTML = '';
  const uniqueItems = dedupeBooks(items || []);
  if (!uniqueItems.length) { container.innerHTML = '<div class="muted">No items</div>'; return; }
  uniqueItems.forEach(it => {
    const b = normalizeBook(it);
    const el = document.createElement('div');
    el.className = 'side-item';
    el.innerHTML = `<div class="thumb" style="background-image:url('${escapeHtml(b.image || PLACEHOLDER)}')"></div><div class="meta"><div class="title">${escapeHtml(b.title)}</div><div class="muted">${escapeHtml((b.authors || []).join(', '))}</div></div>`;
    el.addEventListener('click', async () => { if (b.id) { await incViewOnce(b.id); applySearchAndRender(); openBook(b.id); } });
    container.appendChild(el);
  });
}

/* demo fallback */
function demoBooks() {
  return [
    { ISBN: 'demo-1', 'Book_Title': 'The Complete Illustrated Guide to Chinese Medicine', 'Book-Author': 'Tom Williams', 'Image': 'https://picsum.photos/400/600?1', 'Description': 'Comprehensive guide to Chinese medicine.', 'Publisher': 'HarperThorsons', 'rating': 4.2, 'Categories': ['Health & Fitness'], 'reviews_count': 12, 'views': 54 },
    { ISBN: 'demo-2', 'Book_Title': 'How to Discipline Kids without Losing Their Love', 'Book-Author': 'Jim Fay', 'Image': 'https://picsum.photos/400/600?2', 'Description': 'Parenting guide.', 'Publisher': 'Love & Logic Press', 'rating': 3.8, 'Categories': ['Family & Relationships'], 'reviews_count': 5, 'views': 12 }
  ];
}

/* ---------------- init ---------------- */
(async function init() {
  STATE.cart = JSON.parse(localStorage.getItem('cart') || '[]');
  STATE.wishlist = JSON.parse(localStorage.getItem('wishlist') || '[]');
  updateCounts();
  await loadHome();
  searchInput.addEventListener('input', (e) => doSearch(e.target.value));
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeModal(); });
  modalBackdrop && modalBackdrop.addEventListener('click', closeModal);
})();

/* alias doSearch for earlier listeners */
const doSearchAlias = doSearch;

/* ---------------- utilities ---------------- */
function updateCounts() { cartCount && (cartCount.textContent = STATE.cart.reduce((s, i) => s + (i.qty || 0), 0)); wishCount && (wishCount.textContent = STATE.wishlist.length); }
