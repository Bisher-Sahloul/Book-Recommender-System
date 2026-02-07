// FILE: app.js
/*
  Robust frontend glue for FastAPI Book Recommender
  - Adds verbose debug info when fetch fails (shows HTTP status + body)
  - Keeps tolerant normalization
  - Improves error messages in the UI so you can see backend responses directly
*/

const API_ROOT = '/api';
const PLACEHOLDER_IMAGE = 'https://picsum.photos/seed/placeholder/400/600';

// ---------- Helpers ----------
const el = id => document.getElementById(id);
const debounce = (fn,wait)=>{let t; return (...args)=>{clearTimeout(t);t=setTimeout(()=>fn(...args),wait);};};
function escapeHtml(s){return String(s===undefined||s===null?'':s).replaceAll('&','&amp;').replaceAll('<','&lt;').replaceAll('>','&gt;');}

// Debug store so you can inspect last raw responses in console
window._lastResponses = {};

// Normalize book object from different shapes/keys
function normalizeBook(raw){
  if(!raw) return null;
  if(Array.isArray(raw) && raw.length>0) raw = raw[0];
  const get = (o, ...keys) => {
    for(const k of keys){
      if(o == null) continue;
      if(k in o) return o[k];
      const camel = k.replace(/_([a-z])/g, g=>g[1].toUpperCase());
      if(camel in o) return o[camel];
      const hyphen = k.replaceAll('_','-');
      if(hyphen in o) return o[hyphen];
    }
    return undefined;
  };
  return {
    ISBN: get(raw, 'ISBN', 'isbn'),
    Book_Title: get(raw, 'Book_Title', 'Book-Title', 'Book Title', 'title'),
    Book_Author: get(raw, 'Book_Author', 'Book-Author', 'author'),
    Year_Of_Publication: get(raw, 'Year_Of_Publication', 'Year-Of-Publication', 'year'),
    Publisher: get(raw, 'Publisher', 'publisher'),
    Description: get(raw, 'Description', 'description', 'desc', 'page_content'),
    Categories: get(raw, 'Categories', 'categories'),
    Image: get(raw, 'Image', 'image', 'cover') || PLACEHOLDER_IMAGE,
    rating: get(raw, 'rating', 'score') || 0
  };
}

// ---------- Render helpers ----------
function createBookCard(bookRaw){
  const book = normalizeBook(bookRaw);
  const div = document.createElement('div');
  div.className = 'book-card';
  div.innerHTML = `
    <div class="cover" style="background-image:url('${book.Image}')"></div>
    <div class="title">${escapeHtml(book.Book_Title)}</div>
    <div class="meta">${escapeHtml(book.Book_Author)} · ⭐ ${escapeHtml(book.rating)}</div>
  `;
  div.addEventListener('click', ()=>{
    if(!book.ISBN){ console.warn('Missing ISBN for book', book); return; }
    openBook(book.ISBN);
  });
  return div;
}

function renderGrid(listEl,books){
  listEl.innerHTML='';
  if(!books || books.length===0){ listEl.innerHTML = '<div style="color:var(--muted)">No items</div>'; return }
  books.forEach(b=> listEl.appendChild(createBookCard(b)));
}

// ---------- API calls with enhanced debug ----------
async function fetchJsonVerbose(url, label){
  const info = { url, label, status:null, ok:false, text:null, json:null };
  window._lastResponses[label||url] = info;
  try{
    const r = await fetch(url);
    info.status = r.status; info.ok = r.ok;
    const txt = await r.text();
    info.text = txt;
    // store raw text for debugging
    window._lastResponses[label||url] = info;
    try{
      info.json = JSON.parse(txt);
      window._lastResponses[label||url] = info;
      if(!r.ok){
        throw new Error(`HTTP ${r.status} — ${JSON.stringify(info.json).slice(0,200)}`);
      }
      return info.json;
    }catch(parseErr){
      if(!r.ok){
        throw new Error(`HTTP ${r.status} — ${txt}`);
      }
      // if parse failed but status ok, still try to return text as fallback
      return txt;
    }
  }catch(e){
    console.error('FetchVerbose error', url, e, window._lastResponses[label||url]);
    throw e;
  }
}

async function apiSearch(q){ return await fetchJsonVerbose(`${API_ROOT}/search?q=${encodeURIComponent(q)}`,'search'); }
async function apiPopular(){ return await fetchJsonVerbose(`${API_ROOT}/books/popular`,'popular'); }
async function apiUserRecs(){ return await fetchJsonVerbose(`${API_ROOT}/user/recommendations`,'user_recs'); }
async function apiGetBook(isbn){ return await fetchJsonVerbose(`${API_ROOT}/books/${encodeURIComponent(isbn)}`,'get_book'); }
async function apiRelated(isbn){ return await fetchJsonVerbose(`${API_ROOT}/books/${encodeURIComponent(isbn)}/related`,'related'); }

// ---------- UI ----------
const searchInput = el('searchInput');
const searchResults = el('searchResults');
const popularList = el('popularList');
const userRecList = el('userRecList');
const bookDetail = el('bookDetail');
const placeholderDetail = el('placeholderDetail');
const userArea = el('userArea');

async function init(){
  userArea.innerHTML = `<span class="user-badge">Current User</span>`;
  try{
    const [pop, recs] = await Promise.all([apiPopular(), apiUserRecs()]);
    renderGrid(popularList, Array.isArray(pop)?pop:[]);
    renderGrid(userRecList, Array.isArray(recs)?recs:[]);
  }catch(e){
    popularList.innerHTML = `<div style="color:var(--muted)">Failed to load popular books — ${escapeHtml(e.message||'error')}</div>`;
    userRecList.innerHTML = `<div style="color:var(--muted)">Failed to load recommendations — ${escapeHtml(e.message||'error')}</div>`;
  }
}

const doSearch = debounce(async (q)=>{
  if(!q){searchResults.innerHTML='';return}
  searchResults.innerHTML = '<div style="color:var(--muted)">Searching...</div>';
  try{
    const res = await apiSearch(q);
    // if response is an object that contains an array under common keys, try to extract
    const candidates = Array.isArray(res) ? res : (res && (res.results || res.items || res.data) ? (res.results||res.items||res.data) : []);
    searchResults.innerHTML='';
    renderGrid(searchResults, candidates);
  }catch(e){
    const raw = window._lastResponses['search']?.text || e.message;
    searchResults.innerHTML = `<div style="color:var(--muted)">Search failed — ${escapeHtml(String(e.message).slice(0,200))}</div><pre style="color:var(--muted);font-size:11px;max-height:160px;overflow:auto">${escapeHtml(String(raw).slice(0,800))}</pre>`;
  }
},300);

searchInput.addEventListener('input', e=> doSearch(e.target.value));
searchInput.addEventListener('keydown', async e=>{ if(e.key==='Enter') doSearch(e.target.value); });

async function openBook(isbn){
  placeholderDetail.classList.add('hidden');
  bookDetail.classList.remove('hidden');
  bookDetail.innerHTML = '<div style="color:var(--muted)">Loading...</div>';

  try{
    let bookResp = await apiGetBook(isbn);
    window._lastResponses['get_book_parsed'] = bookResp;
    // bookResp can be list or object; try common extraction
    let rawBook = null;
    if(Array.isArray(bookResp)) rawBook = bookResp[0];
    else if(bookResp && (bookResp.ISBN || bookResp.Book_Title || bookResp.title)) rawBook = bookResp;
    else if(bookResp && (bookResp.results || bookResp.data || bookResp.items)) rawBook = bookResp.results?.[0] || bookResp.data?.[0] || bookResp.items?.[0];
    else rawBook = bookResp;

    const book = normalizeBook(rawBook);

    const relatedResp = await apiRelated(isbn);
    const relatedArr = Array.isArray(relatedResp) ? relatedResp : (relatedResp && (relatedResp.results||relatedResp.items||relatedResp.data) ? (relatedResp.results||relatedResp.items||relatedResp.data) : []);
    const related = relatedArr.map(normalizeBook);

    bookDetail.innerHTML = `
      <div style="display:flex;gap:16px">
        <div style="width:150px"><div class="cover" style="height:220px;background-image:url('${book.Image}')"></div></div>
        <div>
          <h2>${escapeHtml(book.Book_Title)}</h2>
          <div class="meta">${escapeHtml(book.Book_Author)} (${escapeHtml(book.Year_Of_Publication)})</div>
          <p>${escapeHtml(book.Description)}</p>
          <p><b>Publisher:</b> ${escapeHtml(book.Publisher)}</p>
          <p><b>Categories:</b> ${escapeHtml(book.Categories)}</p>
          <p><b>Rating:</b> ⭐ ${escapeHtml(book.rating)}</p>
        </div>
      </div>
      <div class="section">
        <h4>Related Books</h4>
        <div class="related-list" id="relatedList"></div>
      </div>
    `;

    const rEl = document.getElementById('relatedList');
    rEl.innerHTML='';
    related.forEach(b=>{
      const row = document.createElement('div');
      row.className='related-item';
      row.innerHTML = `<div class="cover" style="background-image:url('${b.Image}')"></div><div>${escapeHtml(b.Book_Title)}</div>`;
      row.addEventListener('click', ()=>{ if(b.ISBN) openBook(b.ISBN); });
      rEl.appendChild(row);
    });

  }catch(e){
    const raw = window._lastResponses['get_book']?.text || e.message;
    bookDetail.innerHTML = `<div style="color:var(--muted)">Failed to load book details — ${escapeHtml(String(e.message).slice(0,200))}</div><pre style="color:var(--muted);font-size:11px;max-height:160px;overflow:auto">${escapeHtml(String(raw).slice(0,800))}</pre>`;
  }
}

// expose a manual debug function
window._app_debug = { normalizeBook, _lastResponses: window._lastResponses };

init();
