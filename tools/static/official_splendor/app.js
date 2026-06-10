const gemOrder = ['white', 'blue', 'green', 'red', 'black', 'gold'];
const gemLabel = {white: 'W', blue: 'U', green: 'G', red: 'R', black: 'B', gold: 'Au'};
let currentState = null;
let legalMoves = [];
let hintsByAction = new Map();

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>'"]/g, ch => ({'&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;'}[ch]));
}

function getHints(state) {
  return (state?.board?.hints || state?.hints || []);
}

function makeHintMap(hints) {
  hintsByAction = new Map(hints.map(hint => [Number(hint.action), hint]));
}

function hintBadge(action) {
  if (action === null || action === undefined || !hintsByAction.has(Number(action))) return '';
  const hint = hintsByAction.get(Number(action));
  const win = hint.win_pct === null || hint.win_pct === undefined ? 'n/a' : `${hint.win_pct}%`;
  return `<span class="hint-badge">#${hint.rank} ${win}</span>`;
}

function costTokens(cost) {
  return gemOrder.slice(0, 5)
    .filter(color => Number(cost?.[color] || 0) > 0)
    .map(color => `<span class="cost-token ${color}" title="${color}">${Number(cost[color])}</span>`)
    .join('');
}

function gemDeltaLabel(delta) {
  if (!delta) return '';
  return gemOrder
    .filter(color => Number(delta[color] || 0) !== 0)
    .map(color => `${Number(delta[color]) > 0 ? '+' : ''}${Number(delta[color])}${gemLabel[color]}`)
    .join(' ');
}

function legalMoveForCard(tier, index, mode = 'buy') {
  const kind = mode === 'reserve' ? 'reserve' : 'buy_visible';
  return legalMoves.find(move => move.kind === kind && Number(move.tier) === Number(tier) && Number(move.index) === Number(index));
}

function legalMoveForReserved(index) {
  return legalMoves.find(move => move.kind === 'buy_reserved' && Number(move.index) === Number(index));
}

function tokenMovesForColor(color) {
  return legalMoves.filter(move => ['take_gems', 'give_gems'].includes(move.kind) && Number(move.gem_delta?.[color] || 0) !== 0);
}

function renderStatus(state) {
  document.getElementById('status').innerHTML = (state.status || []).map(item => `<span class="hint-badge">${escapeHtml(item)}</span>`).join(' ');
}

function renderNobles(nobles) {
  document.getElementById('nobles').innerHTML = (nobles || []).map(noble => `
    <div class="noble">
      <strong>${noble.points} pts</strong>
      <div class="costs">${costTokens(noble.cost)}</div>
    </div>`).join('') || '<p class="small">No nobles.</p>';
}

function renderBank(board) {
  const bank = board.bank_gems || {};
  document.getElementById('bankTokens').innerHTML = gemOrder.map(color => `
    <button class="token ${color}" title="${color}" onclick="playFirstTokenMove('${color}')">${Number(bank[color] || 0)}</button>`).join('');
  const tokenMoves = legalMoves.filter(move => ['take_gems', 'give_gems'].includes(move.kind));
  document.getElementById('tokenMoves').innerHTML = tokenMoves.map(move => `
    <button type="button" class="secondary" onclick="playMove(${move.action})" title="${escapeHtml(move.label)}">
      ${escapeHtml(gemDeltaLabel(move.gem_delta) || move.label)} ${hintBadge(move.action)}
    </button>`).join('') || '<p class="small">No legal token moves.</p>';
}

function renderCard(card) {
  const buy = legalMoveForCard(card.tier, card.index, 'buy');
  const reserve = legalMoveForCard(card.tier, card.index, 'reserve');
  const playable = buy ? 'playable' : '';
  const buyAttr = buy ? `onclick="playMove(${buy.action})" title="Buy: ${escapeHtml(buy.label)}"` : '';
  return `
    <article class="card card-${card.color} ${playable} ${card.color === 'empty' ? 'empty' : ''}" ${buyAttr}>
      ${hintBadge(buy?.action)}
      <div class="points">${card.points || ''}</div>
      <div>${escapeHtml(card.color)}</div>
      <div class="costs">${costTokens(card.cost)}</div>
      ${reserve ? `<button type="button" class="secondary reserve" onclick="event.stopPropagation(); playMove(${reserve.action})" title="Reserve: ${escapeHtml(reserve.label)}">Reserve ${hintBadge(reserve.action)}</button>` : ''}
    </article>`;
}

function renderBoard(board) {
  const tiers = [...(board.visible_cards_by_tier || [])].sort((a, b) => Number(b.tier) - Number(a.tier));
  document.getElementById('board').innerHTML = tiers.map(tier => `
    <section class="tier">
      <div class="deck"><div><strong>Tier ${Number(tier.tier) + 1}</strong><br>${tier.deck_count} cards</div></div>
      ${(tier.cards || []).map(renderCard).join('')}
    </section>`).join('');
}

function renderPlayers(state) {
  const active = Number(state.current_player);
  document.getElementById('players').innerHTML = (state.board.players || []).map(player => `
    <article class="player ${Number(player.id) === active ? 'active' : ''}">
      <strong>P${player.id}${player.is_human ? ' (you)' : ' (AI)'}</strong> — ${player.score} pts
      <div class="owned">${gemOrder.map(color => `<span class="cost-token ${color}" title="${color}">${Number(player.gems?.[color] || 0)}</span>`).join('')}</div>
      <div class="owned">${gemOrder.slice(0, 5).map(color => `<span>${gemLabel[color]}:${Number(player.cards?.[color] || 0)}</span>`).join(' ')}</div>
      <div class="reserved">${(player.reserved || []).map(card => {
        const move = player.is_human ? legalMoveForReserved(card.index) : null;
        return `<div class="card card-${card.color} ${move ? 'playable' : ''}" ${move ? `onclick="playMove(${move.action})"` : ''} style="min-height:92px; width:104px;"><strong>${card.points || ''}</strong>${hintBadge(move?.action)}<div class="costs">${costTokens(card.cost)}</div></div>`;
      }).join('') || '<span class="small">No reserved cards</span>'}</div>
    </article>`).join('');
}

function renderHints(hints) {
  document.getElementById('hints').innerHTML = hints.map(hint => `
    <div class="hint-row">
      <strong>#${hint.rank}</strong> action ${hint.action} · win ${hint.win_pct ?? 'n/a'}% · visits ${hint.visits}<br>
      <span>${escapeHtml(hint.label)}</span><br>
      <button type="button" onclick="playMove(${hint.action})">Play suggestion</button>
    </div>`).join('') || '<p class="small">No suggestions yet.</p>';
}

function renderLegalMoves(moves) {
  document.getElementById('legalMoves').innerHTML = moves.map(move => `
    <button type="button" class="secondary" onclick="playMove(${move.action})" title="${escapeHtml(move.label)}">${move.action} ${hintBadge(move.action)}</button>`).join('') || '<p class="small">No legal moves.</p>';
}

function renderLog(log) {
  document.getElementById('log').innerHTML = (log || []).map(item => `<li>${escapeHtml(item)}</li>`).join('');
}

async function loadState() {
  const response = await fetch('/api/state', {headers: {'Accept': 'application/json'}});
  currentState = await response.json();
  legalMoves = currentState?.board?.legal_moves || currentState?.legal_moves || [];
  const hints = getHints(currentState);
  makeHintMap(hints);
  renderStatus(currentState);
  renderNobles(currentState.board.nobles);
  renderBank(currentState.board);
  renderBoard(currentState.board);
  renderPlayers(currentState);
  renderHints(hints);
  renderLegalMoves(legalMoves);
  renderLog(currentState.log);
}

async function playMove(action) {
  await fetch('/api/move', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({action})});
  await loadState();
}

async function playFirstTokenMove(color) {
  const move = tokenMovesForColor(color)[0];
  if (move) await playMove(move.action);
}

async function undoMove() {
  await fetch('/api/undo', {method: 'POST'});
  await loadState();
}

document.getElementById('refresh').addEventListener('click', loadState);
document.getElementById('undo').addEventListener('click', undoMove);
loadState();
